"""Stage 9C-probe — texture distillation (300-500 steps). Warm-start a coherent ternary checkpoint;
keep ALL-ternary g112 (no group-size change, no FP16 swap). Add two texture-aware losses to the
existing rollout objective:
  (1) HF-latent: Charbonnier high-pass(Laplacian) of student vs teacher rollout latents (endpoint+inter).
  (2) late-FFN feature: teacher FP16 FFN(student input) vs student ternary FFN(student input) on the
      late blocks (15-19) -> trains the ternary FFN to imitate the FP16 FFN on the student's own state.
Attention frozen first FREEZE_ATTN steps. AdamW-prox. Render every RENDER_EVERY on held-out hard set.
ENV: RESUME=ckpt_ternary256_preview_v0.pt POOL=sana_calib_256.jsonl STEPS=400 LAM_HF=0.05 LAM_FFN=0.10
     FREEZE_ATTN=150 RAMP_LAM=100 RENDER_EVERY=150 TAG=p9c
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time, random, json, copy
os.environ["HF_HUB_OFFLINE"]="1"; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from diffusers import SanaPipeline
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize import bonsai_muon as BM
dev="cuda"; MODEL="models/Sana_1600M_512px"; K=20; G=112; GUID=4.5; M8=8
RESUME=os.environ["RESUME"]; POOL=os.environ.get("POOL","sana_calib_256.jsonl"); TAG=os.environ.get("TAG","p9c")
STEPS=int(os.environ.get("STEPS","400")); LAM_HF=float(os.environ.get("LAM_HF","0.05")); LAM_FFN=float(os.environ.get("LAM_FFN","0.10"))
FREEZE_ATTN=int(os.environ.get("FREEZE_ATTN","150")); RAMP=int(os.environ.get("RAMP_LAM","100"))
RENDER_EVERY=int(os.environ.get("RENDER_EVERY","150")); KBPTD=int(os.environ.get("KBPTD","4")); NHARD=4; NLATE=5
HARDFILE=os.environ.get("HARDFILE","shared_hard.json")
print(f"[cfg] RESUME={RESUME} POOL={POOL} STEPS={STEPS} LAM_HF={LAM_HF} LAM_FFN={LAM_FFN} FREEZE_ATTN={FREEZE_ATTN} TAG={TAG}",flush=True)

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))
def is_attn_mod(m): return False
def _blk(n):
    try: return int(n.split("transformer_blocks.")[1].split(".")[0])
    except: return -1
LATE=list(range(20-NLATE,20))

pool=[json.loads(l) for l in open(POOL)]
calibP=[r["prompt"] for r in pool if r["split"]=="calib"]; valP=[r["prompt"] for r in pool if r["split"]=="val"]
hardP=json.load(open(HARDFILE))[:NHARD]
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer
pipe.scheduler.set_timesteps(K,device=dev); SIG=pipe.scheduler.sigmas.float().to(dev)

# ---- teacher bank (ehs-dedup) ----
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
        for pi,p in enumerate(prompts):
            ti0["ti"]=ti; cur[0]=0; sink.append(ti)
            pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,output_type="latent",generator=torch.Generator(dev).manual_seed(ti*7))
            ti+=1
h.remove()
calib_chunks=[(ti,k) for ti in calib_tis for k in range(K-M8)]; val_chunks=[(ti,k) for ti in val_tis for k in range(K-M8)]
calib_states=[(ti,k) for ti in calib_tis for k in range(K)]
EVALC=[val_chunks[i] for i in sorted(random.Random(0).sample(range(len(val_chunks)),min(24,len(val_chunks))))]
print(f"[bank] {len(traj)} states | calib {len(calib_tis)} val {len(val_tis)}",flush=True)
with torch.no_grad():
    refs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
print(f"[fp16] {len(refs)} hard refs",flush=True)

# ---- teacher late-FFN copies (FP16) BEFORE warm-start ----
teacher_ff={i: copy.deepcopy(tf.transformer_blocks[i].ff).eval() for i in LATE}
for i in LATE:
    for p in teacher_ff[i].parameters(): p.requires_grad_(False)
print(f"[teacher-ffn] copied late blocks {LATE}",flush=True)

# ---- warm-start trunk+islands from checkpoint, then quantize ----
ck=torch.load(RESUME,map_location="cpu")
def dequant(code,ls,in_f):
    o,ng,g=code.shape; s=torch.exp(ls.float()).unsqueeze(-1); return (code.float()*s).reshape(o,ng*g)[:,:in_f]
qmods=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
with torch.no_grad():
    for (n,m),code,ls,s in zip(qmods,ck["codes"],ck["log_scales"],ck["shapes"]):
        m.weight.data.copy_(dequant(code,ls,s[1]).reshape(m.weight.shape).to(m.weight.dtype).to(dev))
    isl_p0=[p for nm,p in tf.named_parameters() if is_island(nm)]
    for p,c in zip(isl_p0,ck["islands"]): p.data.copy_(c.to(p.dtype).to(dev))
print(f"[warm-start] from {RESUME} step {ck['step']}",flush=True)
BM.replace_modules_with_bonsai_quant(tf, predicate, mode="ternary", group_size=G, init="mse_ternary", forward_blend_alpha=1.0)
for p in tf.parameters(): p.requires_grad_(False)
attn_masters=[]; ffn_masters=[]; scales=[]
for n,m in tf.named_modules():
    if isinstance(m,BM._GroupwiseSTEBase):
        m.weight_master.requires_grad_(True); m.log_scales.requires_grad_(True); scales.append(m.log_scales)
        (ffn_masters if "ff" in n else attn_masters).append(m.weight_master)
island_p=[]
for n,p in tf.named_parameters():
    if is_island(n): p.requires_grad_(True); island_p.append(p)
ste_mods=[m for m in tf.modules() if isinstance(m,BM._GroupwiseSTEBase)]
import bitsandbytes as bnb
opt=bnb.optim.AdamW8bit([{"params":ffn_masters,"lr":2e-5},{"params":attn_masters,"lr":5e-6},
                         {"params":scales,"lr":5e-5},{"params":island_p,"lr":1e-6}], betas=(0.9,0.95))
ATTN_GROUP=1  # index of attn_masters group for freeze
print(f"[quant] {len(ste_mods)} STE | ffn {len(ffn_masters)} attn {len(attn_masters)} | GPU {torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)

# student late-FFN hooks (capture input + output during forward)
cap={}
def mk(i):
    def f(mod,inp,out): cap[i]=(inp[0], out)
    return f
for i in LATE: tf.transformer_blocks[i].ff.register_forward_hook(mk(i))

LAP=torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=torch.float32,device=dev).view(1,1,3,3)
def HP(x):
    C=x.shape[1]; return F.conv2d(x, LAP.repeat(C,1,1,1).to(x.dtype), padding=1, groups=C)
def charb(e): return torch.sqrt(e*e+1e-6)
def hf_loss(xS,xT):
    hs=HP(xS); ht=HP(xT); return charb(hs-ht).mean()/(ht.abs().mean()+1e-6)

def tf_call(hs,ehs,ts,ex):
    o=tf(hidden_states=hs,encoder_hidden_states=ehs,timestep=ts,**ex); return (o[0] if isinstance(o,(tuple,list)) else o.sample)
def roll8(ti,k,grad_last):
    ehs,ex=ehs_pool[ti]; x=traj[(ti,k)]["hs"][:1].float(); inter={}
    for j in range(1,M8+1):
        r=k+j-1; d=traj[(ti,r)]; lat=torch.cat([x,x],0).to(d["hs"].dtype)
        ug=j>(M8-grad_last); ctx=torch.enable_grad() if ug else torch.no_grad()
        with ctx:
            o=tf_call(lat,ehs,d["ts"],ex).float(); vu,vc=o.chunk(2,0); v=vu+GUID*(vc-vu); x=x+(SIG[r+1]-SIG[r])*v
        if not ug: x=x.detach()
        if j in (2,4,6): inter[j]=x
    return x, inter
@torch.no_grad()
def eval_c8():
    bk={'E':[0.,0.],'M':[0.,0.],'L':[0.,0.]}
    for (ti,k) in EVALC:
        xS,_=roll8(ti,k,0); xT=traj[(ti,k+M8)]["hs"][:1].float(); e=(xS-xT).pow(2).sum().item(); t=xT.pow(2).sum().item()
        b='E' if k<4 else('M' if k<8 else 'L'); bk[b][0]+=e; bk[b][1]+=t
    return {b:(v[0]/v[1])**0.5 if v[1]>0 else 0 for b,v in bk.items()}
def fmt(d): return f"E {d['E']:.3f} M {d['M']:.3f} L {d['L']:.3f}"
def vel_and_ffn(ti,k):   # velocity forward (fills hooks) -> vel loss + late-FFN feature loss
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; cap.clear()
    v=tf_call(d["hs"],ehs,d["ts"],ex)
    Lv=F.mse_loss(v.float(), d["vT"].float())
    Lf=0.0
    for i in LATE:
        hS,yS=cap[i]
        with torch.no_grad(): yT=teacher_ff[i](hS.detach())
        Lf=Lf+ (charb(yS.float()-yT.float()).mean()/(yT.float().abs().mean()+1e-6))
    return Lv, Lf/len(LATE)
@torch.no_grad()
def render(tag):
    qs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
    W=512; M=Image.new("RGB",(W*len(hardP),W*2),(20,20,20))
    for j in range(len(hardP)): M.paste(refs[j],(j*W,0)); M.paste(qs[j],(j*W,W))
    M.save(f"render_{TAG}_{tag}.png"); print(f"[render] render_{TAG}_{tag}.png",flush=True)
@torch.no_grad()
def save_ckpt():
    torch.save({"codes":[m._codes_from_matrix_detached(m.weight_master).cpu() for m in ste_mods],
                "log_scales":[m.log_scales.detach().cpu().clone() for m in ste_mods],
                "shapes":[tuple(m.weight_master.shape) for m in ste_mods],
                "islands":[p.detach().cpu().clone() for p in island_p],"mode":"ternary","group_size":G,"step":step}, f"ckpt_{TAG}.pt")

print(f"[init] val-chunk8 {fmt(eval_c8())}",flush=True); render("s0")
t0=time.time()
for step in range(1,STEPS+1):
    rmp=min(1.0, step/float(RAMP)); lam_hf=LAM_HF*rmp; lam_ffn=LAM_FFN*rmp
    for p in attn_masters: p.requires_grad_(step>=FREEZE_ATTN)   # freeze attention early
    opt.zero_grad(set_to_none=True)
    ti,k=random.choice(calib_states); Lv,Lf=vel_and_ffn(ti,k)
    cti,ck2=random.choice(calib_chunks); xS,inter=roll8(cti,ck2,KBPTD); xT=traj[(cti,ck2+M8)]["hs"][:1].float()
    Lm=F.mse_loss(xS,xT); Lhf=hf_loss(xS,xT)
    for j,xj in inter.items(): Lhf=Lhf+hf_loss(xj, traj[(cti,ck2+j)]["hs"][:1].float())
    L=1.0*Lm+0.25*Lv+lam_hf*Lhf+lam_ffn*Lf
    L.backward(); gn=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],0.5)
    if torch.isfinite(gn): opt.step()
    for m in ste_mods: m.weight_master.data.lerp_(m._quantized_matrix_detached(), 0.02)  # prox pull
    if step%50==0 or step==1:
        print(f"[s{step}] Lm{Lm.item():.3e} Lv{Lv.item():.3e} Lhf{Lhf.item():.3e} Lf{Lf.item():.3e} gn{gn:.1f} val-chunk8 {fmt(eval_c8())} ({time.time()-t0:.0f}s)",flush=True)
    if step%RENDER_EVERY==0: render(f"s{step}"); save_ckpt()
render("final"); save_ckpt(); print("[done]",flush=True)
