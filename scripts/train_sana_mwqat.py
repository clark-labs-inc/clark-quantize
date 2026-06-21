"""MW-QAT v2 — fresh master-weight QAT from FP16 Sana. Support is a DERIVED consequence of continuous
master-weight movement crossing the ternary threshold (the ONLY mechanism that produced coherent Sana
pure-ternary; explicit mask/code/soft-mask parameterizations all froze or moved support wrongly).
  W_master = W_fp16 ; W_fwd = Q(W_master) ; grad to master via STE (BM._GroupwiseSTEBase).
KEY change vs q1024t: DIFFERENTIATED island LR (caption_norm = Sana's final-norm compensator, HIGH LR;
proj_out FROZEN; embedders low; conv_depth stays FP16/frozen). Closed-form scale refresh every REFRESH
steps (reset_quant_scales) + trainable log_scales. NO density loss, NO scale prior, NO endpoint, NO mask
logits. alpha-ramp soft landing for optimization; alpha=1 hard renders are the ONLY quality verdict.
Proximal pull toward Q(master) on a schedule. Forensics every 500 (zero frac, sign flips, scale by depth,
proj_out delta MUST be 0, caption_norm delta should grow). 4-GPU DDP. Control: START_HARD=1 ALPHA_RAMP=0.
ENV: POOL=sana_calib_256.jsonl STEPS=6000 RAMP=1200 REFRESH=200 TAG=mwq1
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time, random, json
import torch.distributed as dist
os.environ["HF_HUB_OFFLINE"]="1"; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from diffusers import SanaPipeline
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize import bonsai_muon as BM
WORLD=int(os.environ.get("WORLD_SIZE","1")); RANK=int(os.environ.get("RANK","0")); LRANK=int(os.environ.get("LOCAL_RANK","0")); DDP=WORLD>1
if DDP: dist.init_process_group("nccl"); torch.cuda.set_device(LRANK)
dev=f"cuda:{LRANK}"; MAIN=(RANK==0)
def log(*a):
    if MAIN: print(*a,flush=True)
MODEL="models/Sana_1600M_512px"; K=20; G=int(os.environ.get("GSIZE","112")); GUID=4.5; M8=8
MODE=os.environ.get("MODE","ternary"); STEPS=int(os.environ.get("STEPS","6000")); TAG=os.environ.get("TAG","mwq1")
POOL=os.environ.get("POOL","sana_calib_256.jsonl"); LATENTS=int(os.environ.get("LATENTS","2"))
RAMP=int(os.environ.get("RAMP","1200")); P2START=int(os.environ.get("P2START",str(RAMP))); KBPTD=int(os.environ.get("KBPTD","4"))
REFRESH=int(os.environ.get("REFRESH","200")); START_HARD=int(os.environ.get("START_HARD","0"))
CKPT_EVERY=int(os.environ.get("CKPT_EVERY","1000")); RENDER_EVERY=int(os.environ.get("RENDER_EVERY","500")); FORENSE=int(os.environ.get("FORENSE","500"))
NHARD=int(os.environ.get("NHARD","4")); HARDFILE=os.environ.get("HARDFILE","shared_hard.json")
log(f"[cfg] MODE={MODE} POOL={POOL} STEPS={STEPS} RAMP={RAMP} REFRESH={REFRESH} START_HARD={START_HARD} TAG={TAG} | DDP world={WORLD} rank={RANK}")

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)   # conv_depth (3x3 depthwise) excluded -> stays FP16, frozen
def _blk(n):
    try: return int(n.split("transformer_blocks.")[1].split(".")[0])
    except: return -1
def island_lr(n):   # differentiated: caption_norm = Sana final-norm compensator (HIGH); proj_out frozen; norm_out parameter-free
    if "proj_out" in n: return 0.0
    if "caption_norm" in n or "norm_out" in n: return 2e-4
    if any(k in n for k in ("time_embed","caption_projection")): return 2e-5
    if "patch_embed" in n: return 2e-6
    if "scale_shift_table" in n: return 1e-6
    return 0.0
def is_island(n): return island_lr(n)>0 or "proj_out" in n

pool=[json.loads(l) for l in open(POOL)]
calibP=[r["prompt"] for r in pool if r["split"]=="calib"]; valP=[r["prompt"] for r in pool if r["split"]=="val"]
calibP=calibP[RANK::WORLD]; valP=valP[RANK::WORLD]
hardP=json.load(open(HARDFILE))[:NHARD]
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer
pipe.scheduler.set_timesteps(K,device=dev); SIG=pipe.scheduler.sigmas.float().to(dev)
ehs_pool={}; traj={}; cur=[0]; tinfo={"ti":0}
def hook(m,a,k,o):
    out=o[0] if isinstance(o,(tuple,list)) else o.sample; ti=tinfo["ti"]
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
        nlat=LATENTS if split=="calib" else 1
        for p in prompts:
            for s in range(nlat):
                tinfo["ti"]=ti; cur[0]=0; sink.append(ti)
                pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,output_type="latent",generator=torch.Generator(dev).manual_seed(ti*7+s)); ti+=1
h.remove()
def bucket(k): return 'E' if k<4 else ('M' if k<8 else 'L')
calib_chunks=[(ti,k) for ti in calib_tis for k in range(K-M8)]; val_chunks=[(ti,k) for ti in val_tis for k in range(K-M8)]
calib_states=[(ti,k) for ti in calib_tis for k in range(K)]
cs_b={'E':[],'M':[],'L':[]};
for s in calib_states: cs_b[bucket(s[1])].append(s)
cc_b={'E':[],'M':[],'L':[]}
for c in calib_chunks: cc_b[bucket(c[1])].append(c)
EVALC=[val_chunks[i] for i in sorted(random.Random(0).sample(range(len(val_chunks)),min(24,len(val_chunks))))]
refs=[]
if MAIN:
    with torch.no_grad():
        refs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
log(f"[bank] rank{RANK} {len(traj)} states | calib_tis {len(calib_tis)} val_tis {len(val_tis)} | refs {len(refs)}")

# ---- quantize (master-weight STE) ----
BM.replace_modules_with_bonsai_quant(tf, predicate, mode=MODE, group_size=G, init=("mse_ternary" if MODE=="ternary" else "absmean"), forward_blend_alpha=1.0)
for p in tf.parameters(): p.requires_grad_(False)
ste_mods=[m for m in tf.modules() if isinstance(m,BM._GroupwiseSTEBase)]
for m in ste_mods: m.weight_master.requires_grad_(True); m.log_scales.requires_grad_(True)
island_params=[]
for n,p in tf.named_parameters():
    lr=island_lr(n)
    if lr>0: p.requires_grad_(True); island_params.append((n,p,lr))
import bitsandbytes as bnb
masters=[m.weight_master for m in ste_mods]; scales=[m.log_scales for m in ste_mods]
groups=[{"params":masters,"lr":float(os.environ.get("TRUNK_LR","2e-5"))},{"params":scales,"lr":float(os.environ.get("SCALE_LR","1e-4"))}]
for n,p,lr in island_params: groups.append({"params":[p],"lr":lr})
opt=bnb.optim.AdamW8bit(groups,betas=(0.9,0.95),weight_decay=0.0)
trainable=masters+scales+[p for _,p,_ in island_params]
@torch.no_grad()
def prox_pull(rho):
    if rho<=0: return
    for m in ste_mods: m.weight_master.data.lerp_(m._quantized_matrix_detached(), rho)
@torch.no_grad()
def refresh_scales():
    for m in ste_mods: m.reset_quant_scales("mse_ternary" if MODE=="ternary" else "absmean")
# forensics snapshots (init = PTQ / FP16)
init_codesign=[torch.sign(m._codes_from_matrix_detached(m.weight_master).float()).cpu() for m in ste_mods]
init_island={n:p.detach().float().norm().item() for n,p,_ in island_params}
init_island_val={n:p.detach().clone() for n,p,_ in island_params}
log(f"[mwqat] {len(ste_mods)} STE trunk | islands {len(island_params)} | GPU {torch.cuda.memory_allocated(LRANK)/1e9:.1f}GB")

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
        b=bucket(k); bk[b][0]+=e; bk[b][1]+=t
    return {b:(v[0]/v[1])**0.5 if v[1]>0 else 0 for b,v in bk.items()}
def fmt(d): return f"E {d['E']:.3f} M {d['M']:.3f} L {d['L']:.3f}"
def vel_loss(ti,k):
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; v=tf_call(d["hs"],ehs,d["ts"],ex); return F.mse_loss(v.float(),d["vT"].float())
@torch.no_grad()
def forensics():   # the Prism-pattern check: support moves, sign flips low, proj_out frozen, caption_norm compensates
    zf=0.;n=0;sf=0.;sa=0.; sc=[]
    for i,m in enumerate(ste_mods):
        code=m._codes_from_matrix_detached(m.weight_master).float(); nz=code.numel()
        zf+=(code==0).float().sum().item(); n+=nz
        cs=torch.sign(code).cpu(); init_act=init_codesign[i]!=0; both=(code!=0).cpu()&init_act   # flip = active at BOTH, sign differs (not an activation)
        sf+=((cs!=init_codesign[i])&both).float().sum().item(); sa+=both.float().sum().item()
        sc.append(m._scales().mean().item())
    isl={}
    for n2,p,_ in island_params:
        key="caption_norm" if "caption_norm" in n2 else ("proj_out" if "proj_out" in n2 else ("scale_shift" if "scale_shift" in n2 else "other"))
        isl.setdefault(key,0.); isl[key]+=(p.detach()-init_island_val[n2]).norm().item()
    smean=sum(sc)/len(sc)
    return zf/n, (sf/max(sa,1)), smean, isl
@torch.no_grad()
def render(tag):
    asv=[m.forward_blend_alpha for m in ste_mods]; BM.set_bonsai_forward_blend_alpha(tf,1.0)
    qs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
    for m,a in zip(ste_mods,asv): m.forward_blend_alpha=a
    W=512; M=Image.new("RGB",(W*len(hardP),W*2),(20,20,20))
    for j in range(len(hardP)): M.paste(refs[j],(j*W,0)); M.paste(qs[j],(j*W,W))
    M.save(f"render_{TAG}_{tag}.png"); log(f"[render] render_{TAG}_{tag}.png")
@torch.no_grad()
def save_ckpt():
    torch.save({"codes":[m._codes_from_matrix_detached(m.weight_master).cpu() for m in ste_mods],
                "log_scales":[m.log_scales.detach().cpu().clone() for m in ste_mods],
                "shapes":[tuple(m.weight_master.shape) for m in ste_mods],
                "islands":{n:p.detach().cpu().clone() for n,p,_ in island_params},"mode":MODE,"group_size":G,"step":step}, f"ckpt_{TAG}.pt")

best=[9e9]; t0=time.time()
if MAIN:
    BM.set_bonsai_forward_blend_alpha(tf,1.0); zf,sf,sm,isl=forensics()
    log(f"[init a=1] chunk8 {fmt(eval_c8())} | zero {zf:.3f} flip {sf:.4f} scale {sm:.3f}"); render("s0")
for step in range(1,STEPS+1):
    a=1.0 if START_HARD else min(1.0, step/float(RAMP)); BM.set_bonsai_forward_blend_alpha(tf,a)
    rho=0.0 if (a<1.0 or step<P2START) else (0.005 if step<3000 else (0.02 if step<8000 else 0.05))
    opt.zero_grad(set_to_none=True)
    r=random.random(); cb='E' if r<0.3 else ('M' if r<0.6 else 'L')
    ti,k=random.choice(cs_b[cb] or calib_states); Lv=vel_loss(ti,k); L=0.7*Lv; logm=0.0
    if step>=P2START:
        cs=cc_b[cb] or calib_chunks; cti,ck=random.choice(cs); xS=roll8(cti,ck,KBPTD); xT=traj[(cti,ck+M8)]["hs"][:1].float(); Lm=F.mse_loss(xS,xT); L=0.25*Lv+1.0*Lm; logm=Lm.item()
    L.backward()
    if DDP:
        for p in trainable:
            if p.grad is None: p.grad=torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    gn=torch.nn.utils.clip_grad_norm_(trainable,0.5)
    if torch.isfinite(gn): opt.step()
    prox_pull(rho)
    if step%REFRESH==0: refresh_scales()
    if (step%FORENSE==0 or step==1) and MAIN:
        zf,sf,sm,isl=forensics(); c8=eval_c8(); mx=max(c8.values()); best[0]=min(best[0],mx) if a>=1.0 else best[0]
        isls=" ".join(f"{kk}{vv:.2f}" for kk,vv in isl.items())
        log(f"[s{step}] a{a:.2f} rho{rho} Lv{Lv.item():.2e} Lm{logm:.2e} gn{gn:.1f} chunk8 {fmt(c8)} | zero {zf:.3f} flip {sf:.4f} scale {sm:.3f} [{isls}] best {best[0]:.3f} ({time.time()-t0:.0f}s)")
    if step%RENDER_EVERY==0 and a>=1.0 and MAIN: render(f"s{step}")
    if step%CKPT_EVERY==0 and a>=1.0 and MAIN: save_ckpt()
if MAIN: render("final"); save_ckpt(); log("[done]")
if DDP: dist.barrier(); dist.destroy_process_group()
