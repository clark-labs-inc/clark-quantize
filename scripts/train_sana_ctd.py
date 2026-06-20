"""Stage 11A — Code-Transition Descent (CTD / Bonsai-RDP), Sana ternary, FRESH from PTQ.
The PrismLock-v1 failure was that support never moved (mask-logit STE too weak; mean-density grad
diluted ~1/N). CTD instead moves support EXPLICITLY: per round, accumulate dL/dw on the dequant weight
(scoring pass with grad-requiring leaves), score the 3 code transitions per weight
  activate 0->sign(base)   deactivate +-1->0   flip +1<->-1
by dL ~= g*Delta_w (Delta_w = s*(c'-c)) + per-type penalty (flip HIGH, activate LOW, deactivate MED),
apply the most-improving ~budget fraction, then retrain ONLY group scales + islands (velocity + m8).
NO scale-shrink prior, NO density loss, NO forced zero fraction. Scales free. Soft-land alpha from PTQ;
alpha advances only when the alpha=1 hard eval improves (adaptive, not time-scheduled). Init = optimal
PTQ ternary on FP16 teacher (fresh; never the archived q1024t). 4-GPU DDP: leaf grads all-reduced so all
ranks apply identical transitions. Sana islands: caption_norm high-LR (norm_out is parameter-free), embed
islands low-LR, proj_out frozen, conv_depth stays FP16 (not in trunk predicate). Forensic gate prints
support movement + scale stats each round so we can SEE the mechanism fire.
ENV: POOL=sana_calib_256.jsonl ROUNDS=6 STEPS_PER_ROUND=200 BUDGET=0.002 KSCORE=8 TAG=ctd
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, time, random, json
import torch.distributed as dist
os.environ["HF_HUB_OFFLINE"]="1"; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from diffusers import SanaPipeline
from PIL import Image
WORLD=int(os.environ.get("WORLD_SIZE","1")); RANK=int(os.environ.get("RANK","0")); LRANK=int(os.environ.get("LOCAL_RANK","0")); DDP=WORLD>1
if DDP: dist.init_process_group("nccl"); torch.cuda.set_device(LRANK)
dev=f"cuda:{LRANK}"; MAIN=(RANK==0)
def log(*a):
    if MAIN: print(*a,flush=True)
MODEL="models/Sana_1600M_512px"; K=20; G=112; GUID=4.5; M8=8
POOL=os.environ.get("POOL","sana_calib_256.jsonl"); TAG=os.environ.get("TAG","ctd")
ROUNDS=int(os.environ.get("ROUNDS","6")); SPR=int(os.environ.get("STEPS_PER_ROUND","200")); KSCORE=int(os.environ.get("KSCORE","8"))
BUDGET=float(os.environ.get("BUDGET","0.002")); KBPTD=int(os.environ.get("KBPTD","2"))
PEN_ACT=float(os.environ.get("PEN_ACT","0.0")); PEN_DEACT=float(os.environ.get("PEN_DEACT","0.5")); PEN_FLIP=float(os.environ.get("PEN_FLIP","2.5"))
HARDFILE=os.environ.get("HARDFILE","shared_hard.json"); NHARD=4; DBG=int(os.environ.get("DBG","0")); GCLIP=float(os.environ.get("GCLIP","1.0"))
log(f"[cfg] POOL={POOL} ROUNDS={ROUNDS} SPR={SPR} BUDGET={BUDGET} KSCORE={KSCORE} pen(a{PEN_ACT}/d{PEN_DEACT}/f{PEN_FLIP}) TAG={TAG} | DDP world={WORLD} rank={RANK}")

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)  # conv_depth excluded -> stays FP16
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def island_lr(n):   # caption_norm carries the high "final-norm compensator" LR (Sana norm_out is parameter-free)
    if "proj_out" in n: return 0.0
    if "caption_norm" in n: return 1e-4
    if any(k in n for k in ("time_embed","caption_projection","patch_embed","scale_shift_table")): return 1e-6
    return 0.0

ALPHA=[1.0]; _TOK=[0]
class CTDTernary(nn.Module):
    """ternary codes c in {-1,0,+1} (mutated by transitions), free per-group log_scale, frozen base sign.
    forward weight = (1-ALPHA)*W_fp16 + ALPHA * (scale_g * code). Scoring mode exposes a grad-leaf = forward weight."""
    def __init__(self, base_w, conv_shape=None):
        super().__init__()
        W=base_w.reshape(base_w.shape[0],-1).float(); o,i=W.shape; ng=i//G; self.o,self.i,self.ng,self.G,self.cs=o,i,ng,G,conv_shape
        Wg=W.view(o,ng,G); d=0.7*Wg.abs().mean(-1,keepdim=True); m0=(Wg.abs()>d).float()   # optimal-scale PTQ ternary
        s0=(Wg.abs()*m0).sum(-1,keepdim=True)/m0.sum(-1,keepdim=True).clamp_min(1.)
        sign=torch.where(Wg>=0,1.,-1.)
        self.register_buffer("code",(sign*m0).to(torch.int8))                 # [o,ng,G] {-1,0,1}, PTQ init
        self.register_buffer("base_sign",sign.to(torch.int8))                 # sign(W_fp16) for activation candidate
        self.register_buffer("Wfp",W.view(o,ng,G).to(torch.bfloat16))         # FP16 ref for soft landing
        self.log_scale=nn.Parameter(s0.squeeze(-1).clamp_min(1e-6).log().clone())   # [o,ng] free
        self._score=False; self._leaf=None; self._ck=-1; self._cw=None
    def _wq(self):   # [o,i] bf16, alpha-blended ternary
        s=self.log_scale.exp().clamp_min(1e-6).unsqueeze(-1)                  # [o,ng,1]
        wt=s*self.code.float()                                                # [o,ng,G] ternary
        a=ALPHA[0]; w=wt if a>=1.0 else (1.0-a)*self.Wfp.float()+a*wt
        return w.reshape(self.o,self.i).to(torch.bfloat16)
    def make_leaf(self):   # detached leaf = current forward weight, requires grad -> backward deposits dL/dw
        w=self._wq().detach().clone().requires_grad_(True); self._leaf=w
    def W(self):
        if self._score: w=self._leaf
        elif torch.is_grad_enabled(): w=self._wq()
        else:
            if self._ck!=_TOK[0]:
                with torch.no_grad(): self._cw=self._wq()
                self._ck=_TOK[0]
            w=self._cw
        return w.reshape(self.cs) if self.cs else w
    @torch.no_grad()
    def recompute_scale(self):   # closed-form optimal per-group scale for the CURRENT support (PTQ scale becomes wrong once support moves; gradient can't fix it -> reset analytically)
        active=(self.code!=0).float()
        num=(self.Wfp.float().abs()*active).sum(-1); den=active.sum(-1).clamp_min(1.)
        self.log_scale.data=(num/den).clamp_min(1e-6).log()
    @torch.no_grad()
    def zero_frac(self): return (self.code==0).float().mean().item()

pool=[json.loads(l) for l in open(POOL)]
calibP=[r["prompt"] for r in pool if r["split"]=="calib"]; valP=[r["prompt"] for r in pool if r["split"]=="val"]
calibP=calibP[RANK::WORLD]; valP=valP[RANK::WORLD]
hardP=json.load(open(HARDFILE))[:NHARD]
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer
pipe.scheduler.set_timesteps(K,device=dev); SIG=pipe.scheduler.sigmas.float().to(dev)
ehs_pool={}; traj={}; cur=[0]; ti0={"ti":0}
def hook(m,a,k,o):
    out=o[0] if isinstance(o,(tuple,list)) else o.sample; ti=ti0["ti"]
    if ti not in ehs_pool:
        ehs=k.get("encoder_hidden_states",a[1] if len(a)>1 else None)
        ehs_pool[ti]=(ehs.detach(),{kk:v.detach() if torch.is_tensor(v) else v for kk,v in k.items() if kk not in("hidden_states","encoder_hidden_states","timestep")})
    hs=k.get("hidden_states",a[0] if a else None); ts=k.get("timestep",a[2] if len(a)>2 else None)
    traj[(ti,cur[0])]={"hs":hs.detach(),"ts":ts.detach(),"vT":out.detach()}; cur[0]+=1
h=tf.register_forward_hook(hook,with_kwargs=True)
calib_tis=[]; val_tis=[]
with torch.no_grad():
    ti=0
    for split,prompts,sink in (("calib",calibP,calib_tis),("val",valP,val_tis)):
        for p in prompts:
            ti0["ti"]=ti; cur[0]=0; sink.append(ti)
            pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,output_type="latent",generator=torch.Generator(dev).manual_seed(ti*7)); ti+=1
h.remove()
calib_chunks=[(ti,k) for ti in calib_tis for k in range(K-M8)]; val_chunks=[(ti,k) for ti in val_tis for k in range(K-M8)]
calib_states=[(ti,k) for ti in calib_tis for k in range(K)]
EVALC=[val_chunks[i] for i in sorted(random.Random(0).sample(range(len(val_chunks)),min(24,len(val_chunks))))]
refs=[]
if MAIN:
    with torch.no_grad():
        refs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
log(f"[bank] rank{RANK} {len(traj)} states | calib {len(calib_tis)} val {len(val_tis)} | refs {len(refs)}")

def setattr_path(root,name,mod):
    *par,ch=name.split("."); o=root
    for p in par: o=getattr(o,p)
    setattr(o,ch,mod)
slt=[]
for n,m in [(n,m) for n,m in tf.named_modules() if predicate(n,m)]:
    cs=tuple(m.weight.shape) if isinstance(m,nn.Conv2d) else None
    mod=CTDTernary(m.weight.data, cs).to(dev)
    def mkfwd(slmod,is_conv,bias):
        def f(x):
            w=slmod.W().to(x.dtype); return F.conv2d(x,w,bias) if is_conv else F.linear(x,w,bias)
        return f
    m.forward=mkfwd(mod, cs is not None, m.bias)
    m.weight.data=torch.empty(0,device=dev); setattr(m,"_slt",mod); slt.append((n,mod))
for p in tf.parameters(): p.requires_grad_(False)
scale_params=[]; island_params=[]
for n,m in slt: m.log_scale.requires_grad_(True); scale_params.append(m.log_scale)
for n,p in tf.named_parameters():
    lr=island_lr(n)
    if lr>0: p.requires_grad_(True); island_params.append((p,lr))
import bitsandbytes as bnb
SCALE_LR=float(os.environ.get("SCALE_LR","2e-4"))   # scales were starved at 5e-5 (islands dwarf them); give them room
groups=[{"params":scale_params,"lr":SCALE_LR}]+[{"params":[p],"lr":lr} for p,lr in island_params]
opt=bnb.optim.AdamW8bit(groups,betas=(0.9,0.95))
trainable=scale_params+[p for p,_ in island_params]
try: tf.enable_gradient_checkpointing()
except Exception as e: log("[warn] no grad-ckpt "+str(e))
log(f"[ctd] {len(slt)} ternary trunk | islands {len(island_params)} | GPU {torch.cuda.memory_allocated(LRANK)/1e9:.1f}GB")

def tf_call(hs,ehs,ts,ex):
    o=tf(hidden_states=hs,encoder_hidden_states=ehs,timestep=ts,**ex); return (o[0] if isinstance(o,(tuple,list)) else o.sample)
def roll8(ti,k,grad_last):
    ehs,ex=ehs_pool[ti]; x=traj[(ti,k)]["hs"][:1].float()
    for j in range(1,M8+1):
        r=k+j-1; d=traj[(ti,r)]; lat=torch.cat([x,x],0).to(d["hs"].dtype)
        ug=j>(M8-grad_last); ctx=torch.enable_grad() if ug else torch.no_grad()
        with ctx:
            o=tf_call(lat,ehs,d["ts"],ex).float(); vu,vc=o.chunk(2,0); v=vu+GUID*(vc-vu); x=x+(SIG[r+1]-SIG[r])*v
        if not ug: x=x.detach()
    return x
def vel_loss(ti,k):
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; v=tf_call(d["hs"],ehs,d["ts"],ex); return F.mse_loss(v.float(),d["vT"].float())
def task_loss():   # one velocity + one m8 chunk
    ti,k=random.choice(calib_states); Lv=vel_loss(ti,k)
    cti,ck=random.choice(calib_chunks); xS=roll8(cti,ck,KBPTD); xT=traj[(cti,ck+M8)]["hs"][:1].float(); Lm=F.mse_loss(xS,xT)
    return Lm+0.25*Lv, Lm.item(), Lv.item()
@torch.no_grad()
def eval_c8():
    bk={'E':[0.,0.],'M':[0.,0.],'L':[0.,0.]}
    for (ti,k) in EVALC:
        xS=roll8(ti,k,0); xT=traj[(ti,k+M8)]["hs"][:1].float(); e=(xS-xT).pow(2).sum().item(); t=xT.pow(2).sum().item()
        b='E' if k<4 else('M' if k<8 else 'L'); bk[b][0]+=e; bk[b][1]+=t
    return {b:(v[0]/v[1])**0.5 if v[1]>0 else 0 for b,v in bk.items()}
def fmt(d): return f"E {d['E']:.3f} M {d['M']:.3f} L {d['L']:.3f}"
@torch.no_grad()
def global_zero(): return sum(m.zero_frac()*m.code.numel() for _,m in slt)/sum(m.code.numel() for _,m in slt)
@torch.no_grad()
def scale_stats():
    ls=torch.cat([m.log_scale.flatten() for _,m in slt]); return ls.exp().mean().item(), ls.std().item()
@torch.no_grad()
def render(tag):
    qs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
    W=512; M=Image.new("RGB",(W*len(hardP),W*2),(20,20,20))
    for j in range(len(hardP)): M.paste(refs[j],(j*W,0)); M.paste(qs[j],(j*W,W))
    M.save(f"render_{TAG}_{tag}.png"); log(f"[render] render_{TAG}_{tag}.png")
@torch.no_grad()
def save_ckpt(tag):
    torch.save({"code":[m.code.cpu() for _,m in slt],"log_scale":[m.log_scale.detach().cpu() for _,m in slt],
                "cs":[m.cs for _,m in slt],"islands":{n:p.detach().cpu() for n,p in tf.named_parameters() if island_lr(n)>0},
                "group_size":G,"round":tag}, f"ckpt_{TAG}_{tag}.pt")

@torch.no_grad()
def apply_transitions(budget):
    moved=0; tot=0; nact=ndeact=nflip=0
    for _,m in slt:
        g=m._leaf.grad.reshape(m.o,m.ng,m.G).float()                  # dL/dw (DDP: already all-reduced)
        s=m.log_scale.exp().unsqueeze(-1)                             # [o,ng,1]
        code=m.code.float(); base=m.base_sign.float(); BIG=1e30
        sc_act=torch.where(code==0, g*(s*base),       torch.full_like(g,BIG))   # 0 -> base sign
        sc_de =torch.where(code!=0, g*(s*(-code)),    torch.full_like(g,BIG))   # active -> 0
        sc_fl =torch.where(code!=0, g*(s*(-2*code)),  torch.full_like(g,BIG))   # flip
        valid=torch.cat([sc_act[code==0],sc_de[code!=0],sc_fl[code!=0]])
        sd=valid.std().clamp_min(1e-12) if valid.numel() else torch.tensor(1.,device=g.device)
        sc_act=sc_act/sd+PEN_ACT; sc_de=sc_de/sd+PEN_DEACT; sc_fl=sc_fl/sd+PEN_FLIP
        stack=torch.stack([sc_act,sc_de,sc_fl],0); best,arg=stack.min(0)        # best transition per weight
        k=max(1,int(budget*best.numel())); flat=best.flatten()
        vals,idx=flat.topk(k,largest=False); sel=idx[vals<0.0]                  # only predicted improvements
        if sel.numel()==0: continue
        a=arg.flatten()[sel]; cflat=m.code.view(-1); bflat=m.base_sign.reshape(-1).float()
        newc=torch.where(a==0, bflat[sel], torch.where(a==1, torch.zeros_like(bflat[sel]), -cflat[sel].float()))
        cflat[sel]=newc.to(torch.int8)
        nact+=int((a==0).sum()); ndeact+=int((a==1).sum()); nflip+=int((a==2).sum()); moved+=sel.numel(); tot+=best.numel()
    return moved,tot,nact,ndeact,nflip

def score_pass():   # accumulate dL/dw on leaves over KSCORE minibatches
    for _,m in slt: m._score=True; m.make_leaf()
    for _ in range(KSCORE):
        L,_,_=task_loss(); L.backward()
    if DDP:
        for _,m in slt: dist.all_reduce(m._leaf.grad, op=dist.ReduceOp.AVG)   # identical transitions across ranks
    for _,m in slt: m._score=False

# ---- run ----
best=[1e9]; A0=float(os.environ.get("ALPHA0","0.7")); ASTEP=float(os.environ.get("ALPHA_STEP","0.15"))
_TOK[0]+=1
if MAIN:
    zm,zs=scale_stats(); log(f"[init] zero {global_zero():.3f} scale(mean {zm:.3f} std {zs:.3f}) PTQ ternary | scale_lr {SCALE_LR}")
t0=time.time()
for rnd in range(1,ROUNDS+1):
    ALPHA[0]=min(1.0, A0+ASTEP*(rnd-1)); _TOK[0]+=1   # SCHEDULED soft-land: reach alpha=1 by ~round 3, then train PURE ternary
    # retrain scales + islands at this alpha (codes frozen)
    for step in range(SPR):
        _TOK[0]+=1; opt.zero_grad(set_to_none=True)
        L,lm,lv=task_loss(); L.backward()
        if DDP:
            for p in trainable:
                if p.grad is None: p.grad=torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        gn=torch.nn.utils.clip_grad_norm_(trainable,GCLIP)
        if torch.isfinite(gn): opt.step()
        if MAIN and DBG and step%50==0:
            sg=sum((p.grad.norm().item()**2 for p in scale_params if p.grad is not None))**0.5
            log(f"  [dbg r{rnd} s{step}] L{L.item():.3e} gn{gn:.3f} smean{torch.cat([m.log_scale.flatten() for _,m in slt]).exp().mean().item():.5f}")
    # score + mutate support (DDP: leaf grads already all-reduced -> identical transitions)
    score_pass(); moved,tot,na,nd,nf=apply_transitions(BUDGET)
    for _,m in slt: m.recompute_scale()   # re-fit scales closed-form to the new support
    _TOK[0]+=1
    # alpha=1 hard verdict (the only render/eval that counts; rank0 only)
    sv=ALPHA[0]; ALPHA[0]=1.0; _TOK[0]+=1
    if MAIN:
        hd=eval_c8(); hard=hd['M']+hd['L']; zm,zs=scale_stats()
        log(f"[r{rnd}] train_a{sv:.2f} moved {moved} (act{na} de{nd} fl{nf}) zero {global_zero():.3f} scale(m{zm:.3f} s{zs:.3f}) | alpha=1 {fmt(hd)} ({time.time()-t0:.0f}s)")
        render(f"r{rnd}")
        if hard<best[0]: best[0]=hard; save_ckpt("best")
if MAIN: render("final"); save_ckpt("final"); log("[done]")
if DDP: dist.barrier(); dist.destroy_process_group()
