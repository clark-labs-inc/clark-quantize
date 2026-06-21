"""Stage S3 — Mirror-CTD. Greedy CTD (hard, irreversible, snapshot-scored code edits) collapsed Sana pure
ternary (ctd3 r3 = black at alpha=1) even after 12% support movement + closed-form scale refit. Mirror-CTD
replaces hard greedy edits with a CONTINUOUS, REVERSIBLE support: per-weight mask logit a, soft mask
p = sigmoid(a/tau), forward weight = scale_g * sign(W_fp16) * p. Trained by the FULL teacher distillation
gradient (velocity + m8) over many steps, so support emerges from optimization instead of greedy selection;
temperature anneals tau 1.0 -> 0.03 to harden p into a discrete {0, +-1} code. Signs FROZEN (image policy:
flips rare). Scales free + periodic closed-form refit to the hard support. NO mean-density loss, NO forced
zero fraction, NO endpoint prior. Init = optimal PTQ ternary on FP16 teacher. Hard-snap (a>0) for every
alpha=1 eval/render (the only quality verdict). 4-GPU DDP. Sana islands: caption_norm high-LR, embedders
low, proj_out frozen, conv_depth stays FP16.
ENV: POOL=sana_calib_256.jsonl STEPS=4000 TAU0=1.0 TAUMIN=0.03 SCALE_LR=2e-4 REFIT_EVERY=400 TAG=mir
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
POOL=os.environ.get("POOL","sana_calib_256.jsonl"); TAG=os.environ.get("TAG","mir"); STEPS=int(os.environ.get("STEPS","4000"))
TAU0=float(os.environ.get("TAU0","1.0")); TAUMIN=float(os.environ.get("TAUMIN","0.03")); KBPTD=int(os.environ.get("KBPTD","2"))
SCALE_LR=float(os.environ.get("SCALE_LR","2e-4")); REFIT_EVERY=int(os.environ.get("REFIT_EVERY","400")); GCLIP=float(os.environ.get("GCLIP","1.0"))
RENDER_EVERY=int(os.environ.get("RENDER_EVERY","500")); CKPT_EVERY=int(os.environ.get("CKPT_EVERY","500")); NHARD=4
HARDFILE=os.environ.get("HARDFILE","shared_hard.json"); DBG=int(os.environ.get("DBG","0"))
log(f"[cfg] POOL={POOL} STEPS={STEPS} TAU {TAU0}->{TAUMIN} SCALE_LR={SCALE_LR} REFIT_EVERY={REFIT_EVERY} TAG={TAG} | DDP world={WORLD} rank={RANK}")

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def island_lr(n):
    if "proj_out" in n: return 0.0
    if "caption_norm" in n: return 1e-4
    if any(k in n for k in ("time_embed","caption_projection","patch_embed","scale_shift_table")): return 1e-6
    return 0.0

TAU=[1.0]; HARD=[False]; _TOK=[0]
class MirrorMask(nn.Module):
    """soft support: p = sigmoid(a/tau); weight = scale_g * sign(W_fp16) * p (soft) or *(a>0) (hard snap).
    a trained by teacher loss -> support moves continuously + reversibly; tau anneal hardens p into a code."""
    def __init__(self, base_w, conv_shape=None):
        super().__init__()
        W=base_w.reshape(base_w.shape[0],-1).float(); o,i=W.shape; ng=i//G; self.o,self.i,self.ng,self.G,self.cs=o,i,ng,G,conv_shape
        Wg=W.view(o,ng,G); d=0.7*Wg.abs().mean(-1,keepdim=True); m0=(Wg.abs()>d)              # PTQ support
        s0=(Wg.abs()*m0.float()).sum(-1,keepdim=True)/m0.float().sum(-1,keepdim=True).clamp_min(1.)
        self.register_buffer("sign",torch.where(Wg>=0,1,-1).to(torch.int8).view(o,i))
        self.register_buffer("Wfp",Wg.abs().to(torch.bfloat16))                                # |W_fp16| for closed-form scale
        self.a=nn.Parameter(torch.where(m0.view(o,i),2.0,-2.0))                                 # mask logit, init +-2 by PTQ
        self.log_scale=nn.Parameter(s0.squeeze(-1).clamp_min(1e-6).log().clone())
        self._ck=-1; self._cw=None
    def _w(self):
        s=self.log_scale.exp().clamp_min(1e-6).unsqueeze(-1).expand(self.o,self.ng,self.G).reshape(self.o,self.i).to(torch.bfloat16)
        m=(self.a>0).to(torch.bfloat16) if HARD[0] else torch.sigmoid(self.a/TAU[0]).to(torch.bfloat16)
        w=s*self.sign.to(torch.bfloat16)*m
        return w.reshape(self.cs) if self.cs else w
    def W(self):
        if torch.is_grad_enabled(): return self._w()
        if self._ck!=_TOK[0]:
            with torch.no_grad(): self._cw=self._w()
            self._ck=_TOK[0]
        return self._cw
    @torch.no_grad()
    def refit_scale(self):   # closed-form optimal scale on the CURRENT hard support
        active=(self.a>0).float().view(self.o,self.ng,self.G)
        num=(self.Wfp.float()*active).sum(-1); den=active.sum(-1).clamp_min(1.)
        self.log_scale.data=(num/den).clamp_min(1e-6).log()
    @torch.no_grad()
    def zero_frac(self): return (self.a<=0).float().mean().item()

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

slt=[]
for n,m in [(n,m) for n,m in tf.named_modules() if predicate(n,m)]:
    cs=tuple(m.weight.shape) if isinstance(m,nn.Conv2d) else None
    mod=MirrorMask(m.weight.data, cs).to(dev)
    def mkfwd(slmod,is_conv,bias):
        def f(x):
            w=slmod.W().to(x.dtype); return F.conv2d(x,w,bias) if is_conv else F.linear(x,w,bias)
        return f
    m.forward=mkfwd(mod, cs is not None, m.bias)
    m.weight.data=torch.empty(0,device=dev); setattr(m,"_slt",mod); slt.append((n,mod))
for p in tf.parameters(): p.requires_grad_(False)
mask_params=[]; scale_params=[]; island_params=[]
for n,m in slt: m.a.requires_grad_(True); m.log_scale.requires_grad_(True); mask_params.append(m.a); scale_params.append(m.log_scale)
for n,p in tf.named_parameters():
    lr=island_lr(n)
    if lr>0: p.requires_grad_(True); island_params.append((p,lr))
import bitsandbytes as bnb
MASK_LR=float(os.environ.get("MASK_LR","1e-3"))
groups=[{"params":mask_params,"lr":MASK_LR},{"params":scale_params,"lr":SCALE_LR}]+[{"params":[p],"lr":lr} for p,lr in island_params]
opt=bnb.optim.AdamW8bit(groups,betas=(0.9,0.95))
trainable=mask_params+scale_params+[p for p,_ in island_params]
try: tf.enable_gradient_checkpointing()
except Exception as e: log("[warn] no grad-ckpt "+str(e))
log(f"[mirror] {len(slt)} soft-mask trunk | islands {len(island_params)} | mask_lr {MASK_LR} | GPU {torch.cuda.memory_allocated(LRANK)/1e9:.1f}GB")

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
def task_loss():
    ti,k=random.choice(calib_states); Lv=vel_loss(ti,k)
    cti,ck=random.choice(calib_chunks); xS=roll8(cti,ck,KBPTD); xT=traj[(cti,ck+M8)]["hs"][:1].float(); Lm=F.mse_loss(xS,xT)
    return Lm+0.25*Lv, Lm.item(), Lv.item()
@torch.no_grad()
def eval_c8():   # HARD snap (the deliverable)
    HARD[0]=True; _TOK[0]+=1
    bk={'E':[0.,0.],'M':[0.,0.],'L':[0.,0.]}
    for (ti,k) in EVALC:
        xS=roll8(ti,k,0); xT=traj[(ti,k+M8)]["hs"][:1].float(); e=(xS-xT).pow(2).sum().item(); t=xT.pow(2).sum().item()
        b='E' if k<4 else('M' if k<8 else 'L'); bk[b][0]+=e; bk[b][1]+=t
    HARD[0]=False; _TOK[0]+=1
    return {b:(v[0]/v[1])**0.5 if v[1]>0 else 0 for b,v in bk.items()}
def fmt(d): return f"E {d['E']:.3f} M {d['M']:.3f} L {d['L']:.3f}"
@torch.no_grad()
def global_zero(): return sum(m.zero_frac()*m.a.numel() for _,m in slt)/sum(m.a.numel() for _,m in slt)
@torch.no_grad()
def softness():   # mean |p-0.5|*2 over mask: 1=crisp, 0=fully soft. tracks hardening.
    v=[];
    for _,m in slt: v.append((torch.sigmoid(m.a/TAU[0])-0.5).abs().mean()*2)
    return torch.stack(v).mean().item()
@torch.no_grad()
def scale_stats():
    ls=torch.cat([m.log_scale.flatten() for _,m in slt]); return ls.exp().mean().item(), ls.std().item()
@torch.no_grad()
def render(tag):
    HARD[0]=True; _TOK[0]+=1
    qs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
    HARD[0]=False; _TOK[0]+=1
    W=512; M=Image.new("RGB",(W*len(hardP),W*2),(20,20,20))
    for j in range(len(hardP)): M.paste(refs[j],(j*W,0)); M.paste(qs[j],(j*W,W))
    M.save(f"render_{TAG}_{tag}.png"); log(f"[render] render_{TAG}_{tag}.png")
@torch.no_grad()
def save_ckpt(tag):
    torch.save({"code":[((m.a>0).to(torch.int8)*m.sign.view(m.o,m.ng,m.G)).cpu() for _,m in slt],
                "log_scale":[m.log_scale.detach().cpu() for _,m in slt],"cs":[m.cs for _,m in slt],
                "islands":{n:p.detach().cpu() for n,p in tf.named_parameters() if island_lr(n)>0},"group_size":G,"step":tag}, f"ckpt_{TAG}_{tag}.pt")

# ---- run ----
import math
_TOK[0]+=1
if MAIN:
    zm,zs=scale_stats(); log(f"[init] tau {TAU[0]:.3f} zero {global_zero():.3f} soft {softness():.3f} scale(m {zm:.3f}) | a=1 {fmt(eval_c8())}"); render("s0")
best=[1e9]; t0=time.time()
for step in range(1,STEPS+1):
    TAU[0]=max(TAUMIN, TAU0*(TAUMIN/TAU0)**(step/STEPS)); _TOK[0]+=1   # anneal tau 1.0 -> 0.03 (harden)
    opt.zero_grad(set_to_none=True)
    L,lm,lv=task_loss(); L.backward()
    if DDP:
        for p in trainable:
            if p.grad is None: p.grad=torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    gn=torch.nn.utils.clip_grad_norm_(trainable,GCLIP)
    if torch.isfinite(gn): opt.step()
    if step%REFIT_EVERY==0:
        for _,m in slt: m.refit_scale(); _TOK[0]+=1
    if (step%100==0 or step==1) and MAIN and DBG:
        zm,zs=scale_stats(); log(f"  [dbg s{step}] tau{TAU[0]:.3f} L{L.item():.3e} gn{gn:.2f} zero{global_zero():.3f} soft{softness():.3f} sm{zm:.4f}")
    if step%RENDER_EVERY==0 and MAIN:
        hd=eval_c8(); hard=hd['M']+hd['L']; zm,zs=scale_stats()
        log(f"[s{step}] tau{TAU[0]:.3f} zero {global_zero():.3f} soft {softness():.3f} scale(m{zm:.3f}) | a=1 {fmt(hd)} ({time.time()-t0:.0f}s)")
        render(f"s{step}")
        if hard<best[0]: best[0]=hard; save_ckpt("best")
if MAIN: render("final"); save_ckpt("final"); log("[done]")
if DDP: dist.barrier(); dist.destroy_process_group()
