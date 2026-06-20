"""Sana QAT distillation (BonsaiMuon: Muon for 2D masters + AdamW scales/islands + proximal pull
+ flip trust region) with DIVERSE calibration: prompt-pool JSONL (calib/val/hard splits) x multiple
random latents x stratified timesteps. Fixes 8-prompt overfitting / held-out collapse.
- ehs-dedup bank: text embedding stored ONCE per prompt (constant across 20 steps) -> scales to 1024 prompts.
- train on CALIB, eval chunk8 on VAL (held-out), render on HARD (never trained / never selected on).
- frequent checkpoints (CKPT_EVERY) + frequent held-out renders (RENDER_EVERY).
ENV: POOL=sana_calib_256.jsonl MODE=ternary|binary LATENTS=2 STEPS=12000 RAMP=1500 KBPTD=4
     CKPT_EVERY=1000 RENDER_EVERY=2000 TAG=q256t
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time, random, json
os.environ["HF_HUB_OFFLINE"]="1"; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from diffusers import SanaPipeline
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize import bonsai_muon as BM
dev="cuda"; MODEL="models/Sana_1600M_512px"; K=20; G=112; GUID=4.5; M8=8
MODE=os.environ.get("MODE","ternary"); STEPS=int(os.environ.get("STEPS","12000")); TAG=os.environ.get("TAG","q256t")
POOL=os.environ.get("POOL","sana_calib_256.jsonl"); LATENTS=int(os.environ.get("LATENTS","2"))
RAMP=int(os.environ.get("RAMP","1500")); P2START=int(os.environ.get("P2START",str(RAMP))); KBPTD=int(os.environ.get("KBPTD","4"))
CKPT_EVERY=int(os.environ.get("CKPT_EVERY","1000")); RENDER_EVERY=int(os.environ.get("RENDER_EVERY","2000"))
NHARD=int(os.environ.get("NHARD","4"))
print(f"[cfg] MODE={MODE} POOL={POOL} LATENTS={LATENTS} STEPS={STEPS} RAMP={RAMP} KBPTD={KBPTD} TAG={TAG}",flush=True)

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))

pool=[json.loads(l) for l in open(POOL)]
calibP=[r["prompt"] for r in pool if r["split"]=="calib"]; valP=[r["prompt"] for r in pool if r["split"]=="val"]; hardP=[r["prompt"] for r in pool if r["split"]=="hard"][:NHARD]
print(f"[pool] calib {len(calibP)} val {len(valP)} hard {len(hardP)}",flush=True)
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer
pipe.scheduler.set_timesteps(K,device=dev); SIG=pipe.scheduler.sigmas.float().to(dev)

# ---- teacher bank with ehs-dedup (ehs stored ONCE per trajectory; states keep hs/ts/vT) ----
ehs_pool={}; traj={}; cur=[0]; tinfo={"ti":0}
def hook(m,a,k,o):
    out=o[0] if isinstance(o,(tuple,list)) else o.sample
    ti=tinfo["ti"]
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
        nlat = LATENTS if split=="calib" else 1
        for pi,p in enumerate(prompts):
            for s in range(nlat):
                tinfo["ti"]=ti; cur[0]=0; sink.append(ti)
                pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(ti*7+s))
                ti+=1
h.remove()
def chunks_of(tis): return [(ti,k) for ti in tis for k in range(K-M8)]
calib_chunks=chunks_of(calib_tis); val_chunks=chunks_of(val_tis); calib_states=[(ti,k) for ti in calib_tis for k in range(K)]
# bucket calib for stratified sampling (E:k<7, M:7<=k<13, L:k>=13 by sigma position) -> here k<4/8/rest
def bucket(k): return 'E' if k<4 else ('M' if k<8 else 'L')
cs_b={'E':[],'M':[],'L':[]};
for s in calib_states: cs_b[bucket(s[1])].append(s)
cc_b={'E':[],'M':[],'L':[]}
for c in calib_chunks: cc_b[bucket(c[1])].append(c)
EVALC=[val_chunks[i] for i in sorted(random.Random(0).sample(range(len(val_chunks)), min(24,len(val_chunks))))]
print(f"[bank] {len(traj)} states | calib_tis {len(calib_tis)} val_tis {len(val_tis)} | ehs_pool {len(ehs_pool)} | GPU {torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)
# FP16 hard-set refs (never trained)
with torch.no_grad():
    refs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
print(f"[fp16] {len(refs)} hard refs rendered",flush=True)

# ---- quantize ----
BM.replace_modules_with_bonsai_quant(tf, predicate, mode=MODE, group_size=G, init=("mse_ternary" if MODE=="ternary" else "absmean"), forward_blend_alpha=0.0)
for p in tf.parameters(): p.requires_grad_(False)
for n,m in tf.named_modules():
    if isinstance(m,BM._GroupwiseSTEBase): m.weight_master.requires_grad_(True); m.log_scales.requires_grad_(True)
island_p=[]
for n,p in tf.named_parameters():
    if is_island(n): p.requires_grad_(True); island_p.append(p)
opt=BM.BonsaiMuon(BM.collect_bonsai_param_groups(tf, muon_lr=2e-5, scale_lr=5e-5, island_lr=2e-6, prox_rho=0.0, max_flip_rate=1e-4))
ste_mods=[m for m in tf.modules() if isinstance(m,BM._GroupwiseSTEBase)]
print(f"[replace] {len(ste_mods)} STE modules | GPU {torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)

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
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; v=tf_call(d["hs"],ehs,d["ts"],ex); return F.mse_loss(v.float(), d["vT"].float())
def samp(buckets):  # stratified 0.3 E / 0.3 M / 0.4 L
    r=random.random(); b='E' if r<0.3 else ('M' if r<0.6 else 'L'); pool_b=buckets[b] or buckets['M'] or calib_states
    return random.choice(pool_b)
@torch.no_grad()
def render(tag):
    asv=[m.forward_blend_alpha for m in ste_mods]; BM.set_bonsai_forward_blend_alpha(tf,1.0)
    qs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
    for m,a in zip(ste_mods,asv): m.forward_blend_alpha=a
    W=512; M=Image.new("RGB",(W*len(hardP),W*2),(20,20,20))
    for j in range(len(hardP)): M.paste(refs[j],(j*W,0)); M.paste(qs[j],(j*W,W))
    M.save(f"render_{TAG}_{tag}.png"); print(f"[render] render_{TAG}_{tag}.png",flush=True)
@torch.no_grad()
def save_ckpt():   # compact: int8 hard codes + scales + islands (deployable artifact; reconstruct q=ungroup(codes*exp(log_scales)))
    torch.save({"codes":[m._codes_from_matrix_detached(m.weight_master).cpu() for m in ste_mods],
                "log_scales":[m.log_scales.detach().cpu().clone() for m in ste_mods],
                "shapes":[tuple(m.weight_master.shape) for m in ste_mods],
                "islands":[p.detach().cpu().clone() for p in island_p],"mode":MODE,"group_size":G,"step":step}, f"ckpt_{TAG}.pt")

# ---- Phase 0 ----
BM.set_bonsai_forward_blend_alpha(tf,0.0); print(f"[P0 a=0] vel {vel_loss(*calib_states[0]).item():.2e} chunk8 {fmt(eval_c8())}",flush=True)
BM.set_bonsai_forward_blend_alpha(tf,1.0); print(f"[P0 a=1] vel {vel_loss(*calib_states[0]).item():.2e} chunk8 {fmt(eval_c8())} | zeros {BM.bonsai_code_stats(tf)['zero_frac']:.2f}",flush=True)

best=[9e9]; t0=time.time()
for step in range(1,STEPS+1):
    a=min(1.0, step/float(RAMP)); BM.set_bonsai_forward_blend_alpha(tf,a)
    rho=0.0 if step<P2START else (0.01 if step<3000 else (0.03 if step<8000 else 0.05)); opt.set_prox_rho(rho)
    opt.set_max_flip_rate(1e-4 if step<RAMP else None)
    opt.zero_grad(set_to_none=True)
    ti,k=samp(cs_b); Lv=vel_loss(ti,k); L=0.7*Lv; logm=0.0
    if step>=P2START:
        r=random.random(); cb='E' if r<0.3 else ('M' if r<0.6 else 'L'); cs=cc_b[cb] or calib_chunks
        cti,ck=random.choice(cs); xS=roll8(cti,ck,KBPTD); xT=traj[(cti,ck+M8)]["hs"][:1].float(); Lm=F.mse_loss(xS,xT); L=0.25*Lv+1.0*Lm; logm=Lm.item()
    L.backward(); gn=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],0.5)
    if torch.isfinite(gn): opt.step()
    if step%500==0 or step==1:
        c8=eval_c8(); mx=max(c8.values()); best[0]=min(best[0],mx) if a>=1.0 else best[0]
        print(f"[s{step}] a{a:.2f} rho{rho} Lv{Lv.item():.2e} Lm{logm:.2e} gn{gn:.1f} val-chunk8 {fmt(c8)} | best {best[0]:.3f} ({time.time()-t0:.0f}s)",flush=True)
    if step%CKPT_EVERY==0 and a>=1.0: save_ckpt()
    if step%RENDER_EVERY==0 and a>=1.0: render(f"s{step}")
render("final"); save_ckpt(); print("[done]",flush=True)
