"""Sana-Bonsai-Muon quick test: BonsaiMuon (Muon for 2D masters + AdamW scales/islands +
proximal pull + flip trust region) on a LARGE diverse PartiPrompts calibration set (fixes the
8-prompt overfitting / held-out collapse). Renders held-out prompts DURING training (alpha=1
hard-quant forward). Reuses teacher-bank/roll8/chunk8 from the proven pipeline.
ENV: MODE=ternary|binary NPROMPTS=128 STEPS=8000 RAMP=1500 P2START=1500 KBPTD=8 TAG=BM
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time, random, json
os.environ["HF_HUB_OFFLINE"]="1"
from diffusers import SanaPipeline
from PIL import Image
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root for clark_quantize
from clark_quantize import bonsai_muon as BM
dev="cuda"; MODEL="models/Sana_1600M_512px"; K=20; G=112; GUID=4.5; M8=8
MODE=os.environ.get("MODE","ternary"); STEPS=int(os.environ.get("STEPS","8000")); TAG=os.environ.get("TAG","BM")
NPROMPTS=int(os.environ.get("NPROMPTS","128")); RAMP=int(os.environ.get("RAMP","1500")); P2START=int(os.environ.get("P2START","1500"))
KBPTD=int(os.environ.get("KBPTD","8")); RENDER_EVERY=int(os.environ.get("RENDER_EVERY","4000"))
print(f"[cfg] MODE={MODE} NPROMPTS={NPROMPTS} STEPS={STEPS} RAMP={RAMP} KBPTD={KBPTD} TAG={TAG}",flush=True)

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)   # quantize attn Q/K/V/O + FFN 1x1 convs; protect rest+conv_depth
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))

CALIB=json.load(open("calib_prompts_big.json"))[:NPROMPTS]
HOLD=json.load(open("holdout_prompts_big.json"))[:4]
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer
pipe.scheduler.set_timesteps(K,device=dev); SIG=pipe.scheduler.sigmas.float().to(dev)

# ---- teacher bank (FP16) ----
traj={}; kc=[0]; pidx=[0]
def hook(m,a,k,o):
    out=o[0] if isinstance(o,(tuple,list)) else o.sample
    hs=k.get("hidden_states",a[0] if a else None); ehs=k.get("encoder_hidden_states",a[1] if len(a)>1 else None); ts=k.get("timestep",a[2] if len(a)>2 else None)
    traj[(pidx[0],kc[0])]={"hs":hs.detach(),"ehs":ehs.detach(),"ts":ts.detach(),"vT":out.detach(),"extra":{kk:v.detach() if torch.is_tensor(v) else v for kk,v in k.items() if kk not in("hidden_states","encoder_hidden_states","timestep")}}; kc[0]+=1
h=tf.register_forward_hook(hook,with_kwargs=True)
with torch.no_grad():
    for j,p in enumerate(CALIB):
        pidx[0]=j; kc[0]=0; pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(j))
h.remove()
chunks=[(pi,k) for (pi,k) in traj if k+M8<K]; states=list(traj.keys())
EVALC=[chunks[i] for i in sorted(random.Random(0).sample(range(len(chunks)), min(24,len(chunks))))]
print(f"[bank] {len(states)} states, {len(chunks)} chunks ({NPROMPTS} prompts)",flush=True)
# ---- FP16 holdout refs (pre-quant) ----
with torch.no_grad():
    refs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(1000+j)).images[0] for j,p in enumerate(HOLD)]
print(f"[fp16] {len(refs)} holdout refs rendered",flush=True)

# ---- replace modules with bonsai STE, set trainability ----
replaced=BM.replace_modules_with_bonsai_quant(tf, predicate, mode=MODE, group_size=G, init=("mse_ternary" if MODE=="ternary" else "absmean"), forward_blend_alpha=0.0)
for p in tf.parameters(): p.requires_grad_(False)
for n,m in tf.named_modules():
    if isinstance(m, BM._GroupwiseSTEBase): m.weight_master.requires_grad_(True); m.log_scales.requires_grad_(True)
for n,p in tf.named_parameters():
    if is_island(n): p.requires_grad_(True)
nq=sum(1 for m in tf.modules() if isinstance(m,BM._GroupwiseSTEBase))
print(f"[replace] {len(replaced)} modules -> STE ({nq} quantized) | GPU {torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)
opt=BM.BonsaiMuon(BM.collect_bonsai_param_groups(tf, muon_lr=2e-5, scale_lr=5e-5, island_lr=2e-6, prox_rho=0.0, max_flip_rate=1e-4))

def tf_call(hs,ehs,ts,ex):
    o=tf(hidden_states=hs,encoder_hidden_states=ehs,timestep=ts,**ex); return (o[0] if isinstance(o,(tuple,list)) else o.sample)
def roll8(pi,k,grad_last):
    x=traj[(pi,k)]["hs"][:1].float()
    for j in range(1,M8+1):
        r=k+j-1; d=traj[(pi,r)]; lat=torch.cat([x,x],0).to(d["hs"].dtype)
        ug=j>(M8-grad_last); ctx=torch.enable_grad() if ug else torch.no_grad()
        with ctx:
            o=tf_call(lat,d["ehs"],d["ts"],d["extra"]).float(); vu,vc=o.chunk(2,0); v=vu+GUID*(vc-vu); x=x+(SIG[r+1]-SIG[r])*v
        if not ug: x=x.detach()
    return x
@torch.no_grad()
def eval_c8():
    bk={'E':[0.,0.],'M':[0.,0.],'L':[0.,0.]}
    for (pi,k) in EVALC:
        xS=roll8(pi,k,0); xT=traj[(pi,k+M8)]["hs"][:1].float(); e=(xS-xT).pow(2).sum().item(); t=xT.pow(2).sum().item()
        b='E' if k<4 else('M' if k<8 else 'L'); bk[b][0]+=e; bk[b][1]+=t
    return {b:(v[0]/v[1])**0.5 if v[1]>0 else 0 for b,v in bk.items()}
def fmt(d): return f"E {d['E']:.3f} M {d['M']:.3f} L {d['L']:.3f}"
def vel_loss(st):
    d=traj[st]; v=tf_call(d["hs"],d["ehs"],d["ts"],d["extra"]); return F.mse_loss(v.float(), d["vT"].float())
@torch.no_grad()
def render(tag):
    a_save=[m.forward_blend_alpha for m in tf.modules() if isinstance(m,BM._GroupwiseSTEBase)]
    BM.set_bonsai_forward_blend_alpha(tf,1.0)
    qs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(1000+j)).images[0] for j,p in enumerate(HOLD)]
    for m,a in zip([m for m in tf.modules() if isinstance(m,BM._GroupwiseSTEBase)],a_save): m.forward_blend_alpha=a
    W=512; M=Image.new("RGB",(W*len(HOLD),W*2),(20,20,20))
    for j in range(len(HOLD)): M.paste(refs[j],(j*W,0)); M.paste(qs[j],(j*W,W))
    M.save(f"render_{TAG}_{tag}.png"); print(f"[render] saved render_{TAG}_{tag}.png (top FP16 / bottom {MODE})",flush=True)

# ---- Phase 0 sanity ----
BM.set_bonsai_forward_blend_alpha(tf,0.0); print(f"[P0 a=0] vel {vel_loss(states[0]).item():.2e} chunk8 {fmt(eval_c8())} (~FP16)",flush=True)
BM.set_bonsai_forward_blend_alpha(tf,1.0); print(f"[P0 a=1] vel {vel_loss(states[0]).item():.2e} chunk8 {fmt(eval_c8())} | codes {BM.bonsai_code_stats(tf)}",flush=True)

best=[9e9,None]; t0=time.time()
for step in range(1,STEPS+1):
    a=min(1.0, step/float(RAMP)); BM.set_bonsai_forward_blend_alpha(tf,a)
    rho=0.0 if step<P2START else (0.01 if step<3000 else (0.03 if step<6000 else 0.05)); opt.set_prox_rho(rho)
    opt.set_max_flip_rate(1e-4 if step<P2START else (1e-3 if step<6000 else None))
    phase2=step>=P2START
    opt.zero_grad(set_to_none=True)
    st=random.choice(states); Lv=vel_loss(st); L=0.7*Lv; logm=0.0
    if phase2:
        pi,k=random.choice(chunks); xS=roll8(pi,k,KBPTD); xT=traj[(pi,k+M8)]["hs"][:1].float(); Lm=F.mse_loss(xS,xT); L=0.25*Lv+1.0*Lm; logm=Lm.item()
    L.backward(); gn=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],0.5)
    if torch.isfinite(gn): opt.step()
    if step%1000==0 or step==1:
        c8=eval_c8(); mx=max(c8.values())
        if a>=1.0 and mx<best[0]: best[0]=mx
        print(f"[s{step}] a{a:.2f} rho{rho} Lv{Lv.item():.2e} Lm{logm:.2e} gn{gn:.1f} chunk8 {fmt(c8)} | best {best[0]:.3f} ({time.time()-t0:.0f}s)",flush=True)
    if step%RENDER_EVERY==0: render(f"s{step}")
render("final"); print("[done]",flush=True)
