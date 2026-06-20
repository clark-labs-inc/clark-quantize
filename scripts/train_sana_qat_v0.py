"""Sana-Bonsai-8x: distill FP16 Sana -> codebook-native student (binary/ternary g=112)
under teacher rollout loss, data-free. Wrapper: fp32 master u per group, STE forward with
soft-landing alpha-ramp, proximal pull u<-(1-rho)u+rho*What after each step. Protected island
(adaLN modulation/conditioning/final proj) + conv_depth stay bf16. Loss = velocity + m=8
(truncated BPTD). Phase0 sanity -> Phase1 soft landing -> Phase2 hard codebook.
ENV: MODE=ternary|binary  STEPS=30000  TAG=A1
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time, random
os.environ["HF_HUB_OFFLINE"]="1"
from diffusers import SanaPipeline
try: import bitsandbytes as bnb; BNB=True
except Exception: BNB=False
dev="cuda"; MODEL="models/Sana_1600M_512px"; K=20; G=112; GUID=4.5; M8=8
MODE=os.environ.get("MODE","ternary"); STEPS=int(os.environ.get("STEPS","30000")); TAG=os.environ.get("TAG","A1")
RAMP=int(os.environ.get("RAMP","3000")); P2START=int(os.environ.get("P2START","3000"))
KBPTD=int(os.environ.get("KBPTD","8"))          # backprop steps through m=8 rollout (8=full; 80GB affords it)
M8EVERY=int(os.environ.get("M8EVERY","1"))      # m=8 rollout every Nth step (1=every step, dense; 4=throughput)
GRADCKPT=os.environ.get("GRADCKPT","0")=="1"    # gradient checkpointing (off on 80GB: avoids 2x recompute)
print(f"[cfg] MODE={MODE} STEPS={STEPS} TAG={TAG} RAMP={RAMP} P2START={P2START} KBPTD={KBPTD} M8EVERY={M8EVERY} GRADCKPT={GRADCKPT} bnb={BNB}",flush=True)

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))

def quant_groups(u, mode):
    out,inf=u.shape; ug=u.view(out,inf//G,G)
    if mode=="binary":
        t=torch.where(ug>=0,1.,-1.); s=ug.abs().mean(-1,keepdim=True)
    else:
        tau=0.7*ug.abs().mean(-1,keepdim=True); t=torch.where(ug>tau,1.,torch.where(ug<-tau,-1.,0.))
        s=(ug.abs()*(t!=0).float()).sum(-1,keepdim=True)/((t!=0).float().sum(-1,keepdim=True).clamp_min(1.))
    return (s*t).view(out,inf)

ALPHA=[1.0]
def patch(m, conv):
    w=m.weight.data; w2=w.reshape(w.shape[0],-1) if conv else w
    m._u=nn.Parameter(w2.clone().float()); m._conv=conv
    m._ksh=tuple(w.shape); m.weight.data=torch.empty(0,device=w.device)  # free dead original weight (forward uses _u)
    def fwd(x,_m=m):
        u=_m._u; What=quant_groups(u,MODE); Wste=u+(What-u).detach(); Wf=(1-ALPHA[0])*u+ALPHA[0]*Wste
        if _m._conv: return F.conv2d(x, Wf.view(_m._ksh).to(x.dtype), _m.bias, _m.stride,_m.padding,_m.dilation,_m.groups)
        return F.linear(x, Wf.to(x.dtype), _m.bias)
    m.forward=fwd
@torch.no_grad()
def proximal_pull(rho):
    if rho<=0: return
    for m in QMODS:
        What=quant_groups(m._u.data,MODE); m._u.data.mul_(1-rho).add_(What,alpha=rho)

pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer
pipe.scheduler.set_timesteps(K,device=dev); SIG=pipe.scheduler.sigmas.float().to(dev)
# ---- teacher trajectory bank (data-free: fixed + null + synthetic prompts) ----
traj={}; kc=[0]; pidx=[0]
PROMPTS_SMALL=["a red panda astronaut, studio photo","an oil painting of a lighthouse at sunset","a bowl of ramen, food photography",
         "a cyberpunk street at night, neon","a watercolor fox in a forest","","a portrait of a woman, soft light","a mountain landscape, golden hour"]
PROMPTS_BIG=PROMPTS_SMALL+[  # broaden trajectory distribution for held-out generalization
    "a golden retriever puppy in a field","a vintage red sports car on a coastal road","a steaming cup of coffee on a wooden table",
    "a snowy alpine village at dusk","a close-up of a honeybee on a sunflower","a futuristic city skyline, concept art",
    "a still life of fruit in a ceramic bowl","an astronaut floating in space, earth behind","a cozy library with tall bookshelves",
    "a tiger walking through tall grass","a plate of sushi, overhead shot","a stormy ocean with crashing waves",
    "a hot air balloon over a green valley","a robot in a workshop, cinematic lighting","a field of lavender at sunset, photography",
    "a bowl of fresh strawberries and cream"]
import json as _json
_ps=os.environ.get("PROMPTSET","small")
if _ps=="hf": PROMPTS=_json.load(open("calib_prompts.json"))[:int(os.environ.get("NPROMPTS","48"))]   # PartiPrompts diverse calib
elif _ps=="big": PROMPTS=PROMPTS_BIG
else: PROMPTS=PROMPTS_SMALL
print(f"[prompts] set={_ps} n={len(PROMPTS)}",flush=True)
def hook(m,a,k,o):
    out=o[0] if isinstance(o,(tuple,list)) else o.sample
    hs=k.get("hidden_states",a[0] if a else None); ehs=k.get("encoder_hidden_states",a[1] if len(a)>1 else None); ts=k.get("timestep",a[2] if len(a)>2 else None)
    traj[(pidx[0],kc[0])]={"hs":hs.detach(),"ehs":ehs.detach(),"ts":ts.detach(),"vT":out.detach(),
        "extra":{kk:v.detach() if torch.is_tensor(v) else v for kk,v in k.items() if kk not in("hidden_states","encoder_hidden_states","timestep")}}; kc[0]+=1
h=tf.register_forward_hook(hook,with_kwargs=True)
with torch.no_grad():
    for j,p in enumerate(PROMPTS):
        pidx[0]=j; kc[0]=0; pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(j))
h.remove()
chunks=[(pi,k) for (pi,k) in traj if k+M8<K]
import random as _rnd   # shuffled subsample (NOT strided: stride can align with 12-chunks/prompt -> only k=0 bucket)
EVALC=[chunks[i] for i in sorted(_rnd.Random(0).sample(range(len(chunks)), min(24,len(chunks))))]
states=list(traj.keys())
print(f"[bank] {len(states)} states, {len(chunks)} m=8 chunks",flush=True)

# ---- wrap trunk, set trainable params ----
QMODS=[];
for n,m in tf.named_modules():
    if is_trunk_lin(n,m): patch(m,False); QMODS.append(m)
    elif is_trunk_conv(n,m): patch(m,True); QMODS.append(m)
island_p=[];
for n,p in tf.named_parameters():
    if "._u" in n: continue
    p.requires_grad_(False)
for n,m in tf.named_modules():
    if is_island(n):
        for p in m.parameters(recurse=False): p.requires_grad_(True); island_p.append(p)
# scale_shift_table is a Parameter on the block, not a module
for n,p in tf.named_parameters():
    if "scale_shift_table" in n: p.requires_grad_(True); island_p.append(p)
master=[m._u for m in QMODS]
for p in master: p.requires_grad_(True)
nq=sum(p.numel() for p in master); ni=sum(p.numel() for p in island_p)
print(f"[wrap] {len(QMODS)} trunk modules, {nq/1e6:.0f}M quantized params, {ni/1e6:.1f}M island params, GPU {torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)

def tf_call(hs,ehs,ts,ex):
    o=tf(hidden_states=hs,encoder_hidden_states=ehs,timestep=ts,**ex); return (o[0] if isinstance(o,(tuple,list)) else o.sample)
def roll8(pi,k,grad_last):  # differentiable last grad_last steps; returns endpoint + intermediates dict
    x=traj[(pi,k)]["hs"][:1].float(); inter={}
    for j in range(1,M8+1):
        r=k+j-1; d=traj[(pi,r)]; lat=torch.cat([x,x],0).to(d["hs"].dtype)
        use_grad = j > (M8-grad_last)
        ctx=torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            o=tf_call(lat,d["ehs"],d["ts"],d["extra"]).float(); vu,vc=o.chunk(2,0); v=vu+GUID*(vc-vu); x=x+(SIG[r+1]-SIG[r])*v
        if not use_grad: x=x.detach()
        if j in (2,4,6): inter[j]=x
    return x, inter
@torch.no_grad()
def eval_c8(cs=chunks):
    bk={'E':[0.,0.],'M':[0.,0.],'L':[0.,0.]}
    for (pi,k) in cs:
        xS,_=roll8(pi,k,0); xT=traj[(pi,k+M8)]["hs"][:1].float(); e=(xS-xT).pow(2).sum().item(); t=xT.pow(2).sum().item()
        b='E' if k<4 else('M' if k<8 else 'L'); bk[b][0]+=e; bk[b][1]+=t
    return {b:(v[0]/v[1])**0.5 if v[1]>0 else 0 for b,v in bk.items()}
def fmt(d): return f"E {d['E']:.3f} M {d['M']:.3f} L {d['L']:.3f}"
def vel_loss(st):  # raw per-branch velocity match
    d=traj[st]; v=tf_call(d["hs"],d["ehs"],d["ts"],d["extra"]); return F.mse_loss(v.float(), d["vT"].float())

# ---- Phase 0 sanity ----
ALPHA[0]=0.0; print(f"[P0 alpha=0 (master=fp)] vel {vel_loss(states[0]).item():.2e} chunk8 {fmt(eval_c8())}  (should ~match FP16)",flush=True)
ALPHA[0]=1.0; print(f"[P0 alpha=1 (hard {MODE})] vel {vel_loss(states[0]).item():.2e} chunk8 {fmt(eval_c8())}  (PTQ baseline)",flush=True)
print(f"[P0] conv_depth wrapped? {'conv_depth' in [n for n,m in tf.named_modules() if hasattr(m,'_u')]} (must be False)",flush=True)
if STEPS==0: print("[done] sanity only",flush=True); sys.exit(0)

# ---- optimizer ----
OptC = bnb.optim.AdamW8bit if BNB else torch.optim.AdamW
opt=OptC([{"params":master,"lr":2e-5},{"params":island_p,"lr":2e-6}], betas=(0.9,0.95), weight_decay=0.0)
if GRADCKPT:
    try: tf.enable_gradient_checkpointing()
    except Exception as e: print("[warn] no grad ckpt",e,flush=True)
best=[9e9,None]; t0=time.time()
for step in range(1,STEPS+1):
    # schedules
    ALPHA[0]=min(1.0, step/float(RAMP))             # phase1 soft landing
    rho = 0.0 if step<P2START else (0.01 if step<8000 else (0.03 if step<15000 else (0.05 if step<24000 else 0.10)))
    phase2 = step>=P2START
    opt.zero_grad(set_to_none=True)
    st=random.choice(states); Lv=vel_loss(st); L=0.7*Lv
    logm=0.0
    if phase2 and step%M8EVERY==0:   # m=8 rollout; M8EVERY=1 dense (A100), 4 throughput (24GB)
        pi,k=random.choice(chunks); xS,inter=roll8(pi,k,KBPTD); xT=traj[(pi,k+M8)]["hs"][:1].float()
        Lm=F.mse_loss(xS,xT)
        for j,xj in inter.items(): Lm=Lm+0.25*F.mse_loss(xj, traj[(pi,k+j)]["hs"][:1].float())
        L=0.25*Lv+1.0*Lm; logm=Lm.item()
    L.backward()
    gn=torch.nn.utils.clip_grad_norm_(master+island_p, 0.5)
    if torch.isfinite(gn): opt.step()
    proximal_pull(rho)
    if step%1000==0 or step==1:
        c8=eval_c8(EVALC); mx=max(c8.values())
        if ALPHA[0]>=1.0 and mx<best[0]:   # only track best on the REAL fully-quantized model
            best[0]=mx; best[1]={ "u":[m._u.detach().cpu().clone() for m in QMODS], "isl":[p.detach().cpu().clone() for p in island_p], "step":step}
        bs=best[1]['step'] if best[1] else -1
        print(f"[s{step}] a{ALPHA[0]:.2f} rho{rho} Lv{Lv.item():.2e} Lm{logm:.2e} gn{gn:.1f} chunk8 {fmt(c8)} | best {best[0]:.3f}@{bs} ({time.time()-t0:.0f}s)",flush=True)
        if step%2000==0 and best[1]: torch.save(best[1], f"ckpt_bonsai_{TAG}.pt")   # heavy 6GB write: every 2k
if best[1]: torch.save(best[1], f"ckpt_bonsai_{TAG}.pt"); print(f"[done] best chunk8 {best[0]:.3f} @ step {best[1]['step']}, saved ckpt_bonsai_{TAG}.pt",flush=True)
else: print(f"[done] no alpha>=1 eval reached (steps<{RAMP}+eval_interval); nothing saved",flush=True)
