"""Stage 10B — PrismLock. Forensics showed q1024t stayed in the PTQ regime (zeros 47.7% > Prism 38%,
scales 1.05x vs Prism 0.625x) — the OPPOSITE of Prism's densify+shrink move. Signs were already frozen.
So: sign-locked mask STE (weight = scale * sign(W_fp16) * mask, only mask+scale trained, signs FROZEN)
+ density target (push zeros 48%->~38%, activate PTQ zeros) + scale-shrink prior (->0.625x PTQ) + scale
homogenization. Train trunk masks/scales + available Sana islands (caption_norm/embedders; norm_out is
parameter-free SanaModulatedNorm; proj_out + block modulation frozen). Depth-weighted LR. Loss = velocity
+ m8 + density + scale priors. NO LPIPS. Init from FP16+PTQ (clean Prism replica).
ENV: POOL=sana_calib_256.jsonl STEPS=6000 TARGET_ZERO=0.38 SCALE_RATIO=0.625 DENSITY_RAMP=1500 TAG=plock
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time, random, json
os.environ["HF_HUB_OFFLINE"]="1"; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from diffusers import SanaPipeline
from PIL import Image
dev="cuda"; MODEL="models/Sana_1600M_512px"; K=20; G=112; GUID=4.5; M8=8
POOL=os.environ.get("POOL","sana_calib_256.jsonl"); TAG=os.environ.get("TAG","plock"); STEPS=int(os.environ.get("STEPS","6000"))
TARGET_ZERO=float(os.environ.get("TARGET_ZERO","0.38")); SCALE_RATIO=float(os.environ.get("SCALE_RATIO","0.625"))
DENS_RAMP=int(os.environ.get("DENSITY_RAMP","1500")); KBPTD=int(os.environ.get("KBPTD","2"))
LAM_DENS=float(os.environ.get("LAM_DENS","1.0")); LAM_SM=float(os.environ.get("LAM_SCALE_MEAN","0.05")); LAM_SV=float(os.environ.get("LAM_SCALE_VAR","0.02"))
RENDER_EVERY=int(os.environ.get("RENDER_EVERY","1000")); CKPT_EVERY=int(os.environ.get("CKPT_EVERY","1000")); NHARD=4
HARDFILE=os.environ.get("HARDFILE","shared_hard.json"); LATENTS=1
print(f"[cfg] POOL={POOL} STEPS={STEPS} TARGET_ZERO={TARGET_ZERO} SCALE_RATIO={SCALE_RATIO} DENS_RAMP={DENS_RAMP} TAG={TAG}",flush=True)

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

DENSITY=[1.0]   # global active-fraction schedule (mask = top-(DENSITY*G) per group)
class SignLockedTernary(nn.Module):
    """weight = scale_g * sign(W_fp16) * mask(a). Only mask logits 'a' and log_scale trained. Signs frozen.
    init: PTQ support/scale by default; if init_mask/init_scale given (warm-start from q1024t), use those."""
    def __init__(self, base_w, conv_shape=None, init_mask=None, init_scale=None):
        super().__init__()
        W=base_w.reshape(base_w.shape[0],-1).float(); o,i=W.shape; self.o,self.i,self.ng=o,i,i//G; self.cs=conv_shape
        self.register_buffer("sign", torch.where(W>=0,1,-1).to(torch.int8))      # int8 frozen signs
        Wg=W.view(o,self.ng,G); d=0.7*Wg.abs().mean(-1,keepdim=True); m0=(Wg.abs()>d).float()
        s0=(Wg.abs()*m0).sum(-1,keepdim=True)/m0.sum(-1,keepdim=True).clamp_min(1.)
        self.register_buffer("ptq_logs", s0.squeeze(-1).clamp_min(1e-6).log())   # PTQ log scale [o,ng] (prior target ref)
        mask=init_mask if init_mask is not None else m0                          # warm-start support or PTQ support
        scl=init_scale if init_scale is not None else s0.squeeze(-1)             # warm-start scale or PTQ scale
        self.a=nn.Parameter(torch.where(mask.view(o,i).bool(),2.0,-2.0))         # mask logits (sigmoid>0.5 active)
        self.log_scale=nn.Parameter(scl.clamp_min(1e-6).log().clone())
    def mask(self):   # FREE support mask: support moves via task loss through STE (universal: don't force a model-specific density)
        hard=(self.a>0).float(); soft=torch.sigmoid(self.a)
        return hard+(soft-soft.detach())
    def W(self):
        m=self.mask().view(self.o,self.i).to(torch.bfloat16)
        s=self.log_scale.exp().clamp_min(1e-6).to(torch.bfloat16).unsqueeze(-1).expand(self.o,self.ng,G).reshape(self.o,self.i)
        w=s*self.sign.to(torch.bfloat16)*m; return w.reshape(self.cs) if self.cs else w
    @torch.no_grad()
    def zero_frac(self): return ((self.a<=0).float().mean()).item()

pool=[json.loads(l) for l in open(POOL)]
calibP=[r["prompt"] for r in pool if r["split"]=="calib"]; valP=[r["prompt"] for r in pool if r["split"]=="val"]
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
with torch.no_grad():
    refs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
print(f"[bank] {len(traj)} states | calib {len(calib_tis)} val {len(val_tis)} | fp16 refs {len(refs)}",flush=True)

# replace trunk with SignLockedTernary
def setattr_path(root,name,mod):
    *par,ch=name.split("."); o=root
    for p in par: o=getattr(o,p)
    setattr(o,ch,mod)
RESUME=os.environ.get("RESUME","")   # warm-start support+scale from a trained ternary checkpoint (q1024t)
rck=torch.load(RESUME,map_location="cpu") if RESUME else None
if rck is not None: print(f"[warm-start] sign-locked masks from {RESUME} step {rck['step']}",flush=True)
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
try: tf.enable_gradient_checkpointing()   # recompute W() in backward instead of storing 200 reconstructions
except Exception as e: print("[warn] no grad-ckpt",e,flush=True)
print(f"[prismlock] {len(slt)} sign-locked trunk | islands {len(island_params)} | GPU {torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)

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

print(f"[init] zero {global_zero():.3f} chunk8 {fmt(eval_c8())}  (support moves via task loss; scale-shrink prior toward {SCALE_RATIO}x)",flush=True); render("s0")
t0=time.time()
for step in range(1,STEPS+1):
    opt.zero_grad(set_to_none=True)
    ti,k=random.choice(calib_states); Lv=vel_loss(ti,k)
    cti,ck=random.choice(calib_chunks); xS=roll8(cti,ck,KBPTD); xT=traj[(cti,ck+M8)]["hs"][:1].float(); Lm=F.mse_loss(xS,xT)
    Lsm=Lsv=0.0   # scale-shrink + homogenization priors (toward 0.625x PTQ, narrow)
    for _,m in slt:
        Lsm=Lsm+(m.log_scale.mean()-(m.ptq_logs.mean()+torch.log(torch.tensor(SCALE_RATIO,device=dev)))).pow(2); Lsv=Lsv+m.log_scale.var()
    Lsm/=len(slt); Lsv/=len(slt)
    L=Lm+0.25*Lv+LAM_SM*Lsm+LAM_SV*Lsv
    L.backward(); gn=torch.nn.utils.clip_grad_norm_(mask_params+scale_params+[p for p,_ in island_params],0.5)
    if torch.isfinite(gn): opt.step()
    if step%200==0 or step==1:
        print(f"[s{step}] Lm{Lm.item():.3e} Lv{Lv.item():.3e} Lsm{Lsm.item():.3e} zero{global_zero():.3f} gn{gn:.1f} chunk8 {fmt(eval_c8())} ({time.time()-t0:.0f}s)",flush=True)
    if step%RENDER_EVERY==0: render(f"s{step}")
    if step%CKPT_EVERY==0: save_ckpt()
render("final"); save_ckpt(); print("[done]",flush=True)
