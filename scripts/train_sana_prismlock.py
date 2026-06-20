"""Stage 10B — PrismLock (universal sign-locked QAT, Sana policy instance).
Sign-locked mask STE: weight = scale_g * sign(W_fp16) * mask(a). Only mask logits + log_scale + Sana
islands trained; SIGNS FROZEN from FP16. Support moves ONLY via task loss through the STE (no forced
model-specific density). Scale-shrink prior nudges scales toward 0.625x PTQ + homogenizes. Soft landing:
forward blends (1-alpha)*W_fp16 + alpha*ternary, alpha 0->1 over SOFT_LAND steps, so the velocity/m8 loss
adapts before full quantization. Init FROM FP16+PTQ (clean replica — never the QAT-damaged checkpoints).
Sana islands: caption_norm/embedders trainable; norm_out is parameter-free SanaModulatedNorm; proj_out +
scale_shift_table frozen. Loss = velocity + m8 + scale priors. NO LPIPS.
Data-parallel over N GPUs via `torchrun --nproc_per_node=N` (manual grad all-reduce; rank0 renders/ckpts).
ENV: POOL=sana_calib_256.jsonl STEPS=6000 SCALE_RATIO=0.625 SOFT_LAND=800 TAG=plg4
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time, random, json
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
POOL=os.environ.get("POOL","sana_calib_256.jsonl"); TAG=os.environ.get("TAG","plock"); STEPS=int(os.environ.get("STEPS","6000"))
SCALE_RATIO=float(os.environ.get("SCALE_RATIO","0.625")); SOFT_LAND=int(os.environ.get("SOFT_LAND","800"))   # soft-landing ramp steps (FP16->ternary)
KBPTD=int(os.environ.get("KBPTD","2"))
LAM_SM=float(os.environ.get("LAM_SCALE_MEAN","0.05")); LAM_SV=float(os.environ.get("LAM_SCALE_VAR","0.02"))
RENDER_EVERY=int(os.environ.get("RENDER_EVERY","1000")); CKPT_EVERY=int(os.environ.get("CKPT_EVERY","1000")); NHARD=4
HARDFILE=os.environ.get("HARDFILE","shared_hard.json")
log(f"[cfg] POOL={POOL} STEPS={STEPS} SCALE_RATIO={SCALE_RATIO} SOFT_LAND={SOFT_LAND} TAG={TAG} | DDP world={WORLD} rank={RANK} dev={dev}")

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def _blk(n):
    try: return int(n.split("transformer_blocks.")[1].split(".")[0])
    except: return -1
# island LR policy (norm_out is parameter-free in Sana): train caption_norm + embedders; freeze modulation/proj_out/patch low
def island_lr(n):
    if "proj_out" in n or "scale_shift_table" in n: return 0.0
    if "caption_norm" in n: return 2e-5
    if "patch_embed" in n: return 2e-6
    if any(k in n for k in ("time_embed","caption_projection")): return 5e-6
    return 0.0

ALPHA=[1.0]   # soft-landing blend: W_fwd = (1-ALPHA)*W_fp16 + ALPHA*ternary (ramp 0->1 from cold PTQ start)
_TOK=[0]      # step token: bumped each step so no_grad weight cache invalidates when params/ALPHA change
class SignLockedTernary(nn.Module):
    """weight = scale_g * sign(W_fp16) * mask(a). Only mask logits 'a' and log_scale trained. Signs frozen.
    init: PTQ support/scale by default; if init_mask/init_scale given (warm-start from q1024t), use those."""
    def __init__(self, base_w, conv_shape=None, init_mask=None, init_scale=None):
        super().__init__()
        W=base_w.reshape(base_w.shape[0],-1).float(); o,i=W.shape; self.o,self.i,self.ng=o,i,i//G; self.cs=conv_shape
        self.register_buffer("sign", torch.where(W>=0,1,-1).to(torch.int8))      # int8 frozen signs
        self.register_buffer("Wfp", W.to(torch.bfloat16))                        # FP16 ref for soft landing (frozen)
        Wg=W.view(o,self.ng,G); d=0.7*Wg.abs().mean(-1,keepdim=True); m0=(Wg.abs()>d).float()
        s0=(Wg.abs()*m0).sum(-1,keepdim=True)/m0.sum(-1,keepdim=True).clamp_min(1.)
        self.register_buffer("ptq_logs", s0.squeeze(-1).clamp_min(1e-6).log())   # PTQ log scale [o,ng] (prior target ref)
        mask=init_mask if init_mask is not None else m0                          # warm-start support or PTQ support
        scl=init_scale if init_scale is not None else s0.squeeze(-1)             # warm-start scale or PTQ scale
        self.a=nn.Parameter(torch.where(mask.view(o,i).bool(),2.0,-2.0))         # mask logits (sigmoid>0.5 active)
        self.log_scale=nn.Parameter(scl.clamp_min(1e-6).log().clone())
        self._ck=-1; self._cw=None                                              # no_grad weight cache (token, weight)
    def mask(self):   # FREE support mask: support moves via task loss through STE (universal: don't force a model-specific density)
        hard=(self.a>0).float(); soft=torch.sigmoid(self.a)
        return hard+(soft-soft.detach())
    def _build(self):
        m=self.mask().view(self.o,self.i).to(torch.bfloat16)
        s=self.log_scale.exp().clamp_min(1e-6).to(torch.bfloat16).unsqueeze(-1).expand(self.o,self.ng,G).reshape(self.o,self.i)
        wq=s*self.sign.to(torch.bfloat16)*m
        a=ALPHA[0]; w=wq if a>=1.0 else (1.0-a)*self.Wfp+a*wq   # soft landing: blend FP16->ternary over SOFT_LAND steps
        return w.reshape(self.cs) if self.cs else w
    def W(self):   # grad path reconstructs (autograd through mask/scale); no_grad path reuses per-step cache
        if torch.is_grad_enabled(): return self._build()
        if self._ck!=_TOK[0]:
            with torch.no_grad(): self._cw=self._build()
            self._ck=_TOK[0]
        return self._cw
    @torch.no_grad()
    def zero_frac(self): return ((self.a<=0).float().mean()).item()

pool=[json.loads(l) for l in open(POOL)]
calibP=[r["prompt"] for r in pool if r["split"]=="calib"]; valP=[r["prompt"] for r in pool if r["split"]=="val"]
calibP=calibP[RANK::WORLD]; valP=valP[RANK::WORLD]   # data parallel: each rank banks its own prompt shard
hardP=json.load(open(HARDFILE))[:NHARD]
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer
pipe.scheduler.set_timesteps(K,device=dev); SIG=pipe.scheduler.sigmas.float().to(dev)
# bank
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
log(f"[bank] rank{RANK} {len(traj)} states | calib {len(calib_tis)} val {len(val_tis)} | fp16 refs {len(refs)}")

# replace trunk with SignLockedTernary
def setattr_path(root,name,mod):
    *par,ch=name.split("."); o=root
    for p in par: o=getattr(o,p)
    setattr(o,ch,mod)
RESUME=os.environ.get("RESUME","")   # warm-start support+scale from a trained ternary checkpoint (q1024t)
rck=torch.load(RESUME,map_location="cpu") if RESUME else None
if rck is not None: log(f"[warm-start] sign-locked masks from {RESUME} step {rck['step']}")
slt=[]; depthmult=[]
qmods_list=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
for idx,(n,m) in enumerate(qmods_list):
    cs=tuple(m.weight.shape) if isinstance(m,nn.Conv2d) else None
    im=isc=None
    if rck is not None:
        code=rck["codes"][idx]; im=(code!=0).float().to(dev); isc=torch.exp(rck["log_scales"][idx].float()).to(dev)
    mod=SignLockedTernary(m.weight.data, cs, im, isc).to(dev)
    orig=m   # monkeypatch forward to use the sign-locked weight
    def mkfwd(slmod,is_conv,bias):
        def f(x):
            w=slmod.W().to(x.dtype)
            return F.conv2d(x,w,bias) if is_conv else F.linear(x,w,bias)
        return f
    orig.forward=mkfwd(mod, cs is not None, orig.bias)
    orig.weight.data=torch.empty(0,device=dev)   # free unused FP16 weight (forward uses sign-locked W())
    setattr(orig,"_slt",mod); slt.append((n,mod)); depthmult.append(_blk(n))
for p in tf.parameters(): p.requires_grad_(False)
mask_params=[]; scale_params=[]; island_params=[]
for n,m in slt: m.a.requires_grad_(True); m.log_scale.requires_grad_(True); mask_params.append(m.a); scale_params.append(m.log_scale)
for n,p in tf.named_parameters():
    lr=island_lr(n)
    if lr>0: p.requires_grad_(True); island_params.append((p,lr))
import bitsandbytes as bnb
groups=[{"params":mask_params,"lr":2e-5},{"params":scale_params,"lr":5e-5}]
for p,lr in island_params: groups.append({"params":[p],"lr":lr})
opt=bnb.optim.AdamW8bit(groups,betas=(0.9,0.95))
trainable=mask_params+scale_params+[p for p,_ in island_params]   # for grad all-reduce (DDP) + clip
try: tf.enable_gradient_checkpointing()   # recompute W() in backward instead of storing 200 reconstructions
except Exception as e: log("[warn] no grad-ckpt "+str(e))
log(f"[prismlock] {len(slt)} sign-locked trunk | islands {len(island_params)} | GPU {torch.cuda.memory_allocated(LRANK)/1e9:.1f}GB")

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
@torch.no_grad()
def eval_c8():
    bk={'E':[0.,0.],'M':[0.,0.],'L':[0.,0.]}
    for (ti,k) in EVALC:
        xS=roll8(ti,k,0); xT=traj[(ti,k+M8)]["hs"][:1].float(); e=(xS-xT).pow(2).sum().item(); t=xT.pow(2).sum().item()
        b='E' if k<4 else('M' if k<8 else 'L'); bk[b][0]+=e; bk[b][1]+=t
    return {b:(v[0]/v[1])**0.5 if v[1]>0 else 0 for b,v in bk.items()}
def fmt(d): return f"E {d['E']:.3f} M {d['M']:.3f} L {d['L']:.3f}"
def vel_loss(ti,k):
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; v=tf_call(d["hs"],ehs,d["ts"],ex); return F.mse_loss(v.float(),d["vT"].float())
@torch.no_grad()
def global_zero(): return sum((m.a.data<=0).float().mean().item()*m.a.numel() for _,m in slt)/sum(m.a.numel() for _,m in slt)
@torch.no_grad()
def render(tag):
    qs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
    W=512; M=Image.new("RGB",(W*len(hardP),W*2),(20,20,20))
    for j in range(len(hardP)): M.paste(refs[j],(j*W,0)); M.paste(qs[j],(j*W,W))
    M.save(f"render_{TAG}_{tag}.png"); print(f"[render] render_{TAG}_{tag}.png",flush=True)
@torch.no_grad()
def save_ckpt():
    torch.save({"a":[m.a.detach().cpu() for _,m in slt],"log_scale":[m.log_scale.detach().cpu() for _,m in slt],
                "sign":[m.sign.detach().cpu() for _,m in slt],"cs":[m.cs for _,m in slt],
                "islands":{n:p.detach().cpu() for n,p in tf.named_parameters() if island_lr(n)>0},"step":step}, f"ckpt_{TAG}.pt")

ALPHA[0]=0.0 if SOFT_LAND>0 else 1.0   # start of soft landing (FP16) or straight ternary
if MAIN: log(f"[init] alpha {ALPHA[0]:.2f} zero {global_zero():.3f} chunk8 {fmt(eval_c8())}  (soft-land {SOFT_LAND} steps; support via task loss; scale-shrink->{SCALE_RATIO}x)"); render("s0")
t0=time.time()
for step in range(1,STEPS+1):
    ALPHA[0]=min(1.0, step/float(SOFT_LAND)) if SOFT_LAND>0 else 1.0
    _TOK[0]+=1   # invalidate no_grad weight cache (params + ALPHA change each step)
    opt.zero_grad(set_to_none=True)
    ti,k=random.choice(calib_states); Lv=vel_loss(ti,k)
    cti,ck=random.choice(calib_chunks); xS=roll8(cti,ck,KBPTD); xT=traj[(cti,ck+M8)]["hs"][:1].float(); Lm=F.mse_loss(xS,xT)
    Lsm=Lsv=0.0   # scale-shrink + homogenization priors (toward 0.625x PTQ, narrow)
    for _,m in slt:
        Lsm=Lsm+(m.log_scale.mean()-(m.ptq_logs.mean()+torch.log(torch.tensor(SCALE_RATIO,device=dev)))).pow(2); Lsv=Lsv+m.log_scale.var()
    Lsm/=len(slt); Lsv/=len(slt)
    L=Lm+0.25*Lv+LAM_SM*Lsm+LAM_SV*Lsv
    L.backward()
    if DDP:   # average grads across replicas (effective batch = world_size); fixed order + materialized zeros = no collective divergence
        for p in trainable:
            if p.grad is None: p.grad=torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    gn=torch.nn.utils.clip_grad_norm_(trainable,0.5)
    if torch.isfinite(gn): opt.step(); _TOK[0]+=1   # bump again: eval/render below must see post-update weights
    if (step%100==0 or step==1) and MAIN:
        log(f"[s{step}] a{ALPHA[0]:.2f} Lm{Lm.item():.3e} Lv{Lv.item():.3e} Lsm{Lsm.item():.3e} zero{global_zero():.3f} gn{gn:.1f} chunk8 {fmt(eval_c8())} ({time.time()-t0:.0f}s)")
    if step%RENDER_EVERY==0 and MAIN: render(f"s{step}")
    if step%CKPT_EVERY==0 and MAIN: save_ckpt()
if MAIN: render("final"); save_ckpt(); log("[done]")
if DDP: dist.barrier(); dist.destroy_process_group()
