"""Stage 9C-full — decoded-image perceptual distillation. The probe proved latent-space texture losses
are visually misaligned; texture lives AFTER the VAE decoder. So: roll the student from a LATE teacher
latent to x0, decode x0 through the VAE, and supervise the DECODED IMAGE with LPIPS + edge(Sobel) +
high-gradient patch-LPIPS + light pixel-L1, plus the latent m=8 + velocity terms. Train late-block FFN
(15-19) + FFN scales + islands; attention frozen early. All-ternary g112 (no group-size change).
Memory: one rollout-to-x0 (serves m8+decode), teacher images decoded ONCE & cached, grad-ckpt tf+vae, bptt=4.
ENV: RESUME=ckpt_ternary256_preview_v0.pt POOL=sana_calib_256.jsonl STEPS=500 LAM_LPIPS=0.05 LAM_EDGE=0.10
     LAM_PATCH=0.05 LAM_PIX=0.05 FREEZE_ATTN=1000 BPTT=4 RENDER_EVERY=250 TAG=p9cf
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time, random, json
os.environ["HF_HUB_OFFLINE"]="1"; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from diffusers import SanaPipeline
from PIL import Image
import lpips as lpips_lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize import bonsai_muon as BM
from clark_quantize.sana_ckpt import island_values, load_compact_checkpoint, validate_quant_names
dev="cuda"; MODEL="models/Sana_1600M_512px"; K=20; G=112; GUID=4.5
RESUME=os.environ["RESUME"]; POOL=os.environ.get("POOL","sana_calib_256.jsonl"); TAG=os.environ.get("TAG","p9cf")
STEPS=int(os.environ.get("STEPS","500")); BPTT=int(os.environ.get("BPTT","4")); FREEZE_ATTN=int(os.environ.get("FREEZE_ATTN","1000"))
LAM_LPIPS=float(os.environ.get("LAM_LPIPS","0.05")); LAM_EDGE=float(os.environ.get("LAM_EDGE","0.10"))
LAM_PATCH=float(os.environ.get("LAM_PATCH","0.05")); LAM_PIX=float(os.environ.get("LAM_PIX","0.05"))
RAMP=int(os.environ.get("RAMP_LAM","100")); RENDER_EVERY=int(os.environ.get("RENDER_EVERY","250")); NHARD=4; NLATE=5
HARDFILE=os.environ.get("HARDFILE","shared_hard.json"); LATE_K=[12,14,16]   # start steps for rollout-to-x0 (4,6,8 steps)
print(f"[cfg] RESUME={RESUME} POOL={POOL} STEPS={STEPS} LPIPS={LAM_LPIPS} EDGE={LAM_EDGE} PATCH={LAM_PATCH} BPTT={BPTT} TAG={TAG}",flush=True)

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))
def _blk(n):
    try: return int(n.split("transformer_blocks.")[1].split(".")[0])
    except: return -1
LATE=list(range(20-NLATE,20))

pool=[json.loads(l) for l in open(POOL)]
calibP=[r["prompt"] for r in pool if r["split"]=="calib"]; valP=[r["prompt"] for r in pool if r["split"]=="val"]
hardP=json.load(open(HARDFILE))[:NHARD]
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer; vae=pipe.vae
pipe.scheduler.set_timesteps(K,device=dev); SIG=pipe.scheduler.sigmas.float().to(dev)
VSCALE=getattr(vae.config,"scaling_factor",1.0) or 1.0
@torch.no_grad()
def vae_decode_ng(x):   # x latent [1,C,h,w] -> image [-1,1], no grad
    return vae.decode((x/VSCALE).to(vae.dtype), return_dict=False)[0].float().clamp(-1,1)
def vae_decode(x):
    return vae.decode((x/VSCALE).to(vae.dtype), return_dict=False)[0].float().clamp(-1,1)

# ---- teacher bank (ehs-dedup) + teacher final latent x0 ----
ehs_pool={}; traj={}; x0T={}; cur=[0]; ti0={"ti":0}
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
            out=pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,output_type="latent",generator=torch.Generator(dev).manual_seed(ti*7))
            x0T[ti]=out.images.detach()[:1].float()   # teacher final latent
            ti+=1
h.remove()
calib_states=[(ti,k) for ti in calib_tis for k in range(K)]
val_chunks=[(ti,12) for ti in val_tis]
print(f"[bank] {len(traj)} states | calib {len(calib_tis)} val {len(val_tis)}",flush=True)
# teacher decoded images cached (no grad) + FP16 hard refs
IT={ti: vae_decode_ng(x0T[ti]) for ti in calib_tis}
with torch.no_grad():
    refs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
print(f"[fp16] {len(refs)} hard refs | teacher imgs cached {len(IT)}",flush=True)

# ---- warm-start + quantize ----
ck=load_compact_checkpoint(RESUME)
def dequant(code,ls,in_f):
    o,ng,g=code.shape; s=torch.exp(ls.float()).unsqueeze(-1); return (code.float()*s).reshape(o,ng*g)[:,:in_f]
qmods=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
validate_quant_names(ck,[n for n,_ in qmods])
with torch.no_grad():
    for (n,m),code,ls,s in zip(qmods,ck["codes"],ck["log_scales"],ck["shapes"]):
        m.weight.data.copy_(dequant(code,ls,s[1]).reshape(m.weight.shape).to(m.weight.dtype).to(dev))
    island_named=[(nm,p) for nm,p in tf.named_parameters() if is_island(nm)]
    for (_,p),c in zip(island_named,island_values(ck,island_named)): p.data.copy_(c.to(p.dtype).to(dev))
print(f"[warm-start] step {ck['step']}",flush=True)
BM.replace_modules_with_bonsai_quant(tf, predicate, mode="ternary", group_size=G, init="mse_ternary", forward_blend_alpha=1.0)
for p in tf.parameters(): p.requires_grad_(False)
late_masters=[]; attn_masters=[]; scales=[]
for n,m in tf.named_modules():
    if isinstance(m,BM._GroupwiseSTEBase):
        m.log_scales.requires_grad_(True); scales.append(m.log_scales)
        if ".ff." in n and _blk(n)>=20-NLATE: m.weight_master.requires_grad_(True); late_masters.append(m.weight_master)
        elif (".attn1." in n or ".attn2." in n): attn_masters.append(m.weight_master)
island_p=[]
for n,p in tf.named_parameters():
    if is_island(n): p.requires_grad_(True); island_p.append(p)
ste_mods=[m for m in tf.modules() if isinstance(m,BM._GroupwiseSTEBase)]
try: tf.enable_gradient_checkpointing()
except Exception as e: print("[warn] no tf grad-ckpt",e,flush=True)
try: vae.enable_gradient_checkpointing()
except Exception as e: print("[warn] no vae grad-ckpt",e,flush=True)
import bitsandbytes as bnb
opt=bnb.optim.AdamW8bit([{"params":late_masters,"lr":2e-5},{"params":scales,"lr":5e-5},{"params":island_p,"lr":1e-6}],betas=(0.9,0.95))
LP=lpips_lib.LPIPS(net='vgg').to(dev); LP.eval()
for p in LP.parameters(): p.requires_grad_(False)
print(f"[quant] {len(ste_mods)} STE | late-FFN masters {len(late_masters)} | GPU {torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)

SOB=torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]],[[-1,-2,-1],[0,0,0],[1,2,1]]],dtype=torch.float32,device=dev).unsqueeze(1)
def lum(I): return (0.299*I[:,0:1]+0.587*I[:,1:2]+0.114*I[:,2:3])
def edge(I): return F.conv2d(lum(I),SOB.to(I.dtype),padding=1)
def edge_loss(IS,IT): return (edge(IS)-edge(IT)).abs().mean()
def patch_loss(IS,IT,ksz=160,n=2):
    with torch.no_grad():
        e=edge(IT).abs().sum(1,keepdim=True); H,W=e.shape[-2:]
        pe=F.avg_pool2d(e,ksz,stride=ksz//2); idx=pe.flatten().topk(n).indices
        nw=pe.shape[-1]; coords=[((i//nw)*(ksz//2),(i%nw)*(ksz//2)) for i in idx.tolist()]
    L=0.0
    for (y,x) in coords:
        y=min(y,IT.shape[-2]-ksz); x=min(x,IT.shape[-1]-ksz)
        L=L+LP(IS[:,:,y:y+ksz,x:x+ksz], IT[:,:,y:y+ksz,x:x+ksz]).mean()
    return L/max(1,len(coords))

def tf_call(hs,ehs,ts,ex):
    o=tf(hidden_states=hs,encoder_hidden_states=ehs,timestep=ts,**ex); return (o[0] if isinstance(o,(tuple,list)) else o.sample)
def roll_to_x0(ti,k,grad_last):
    ehs,ex=ehs_pool[ti]; x=traj[(ti,k)]["hs"][:1].float(); n=K-k
    for j in range(1,n+1):
        r=k+j-1; d=traj[(ti,r)]; lat=torch.cat([x,x],0).to(d["hs"].dtype)
        ug=j>(n-grad_last); ctx=torch.enable_grad() if ug else torch.no_grad()
        with ctx:
            o=tf_call(lat,ehs,d["ts"],ex).float(); vu,vc=o.chunk(2,0); v=vu+GUID*(vc-vu); x=x+(SIG[r+1]-SIG[r])*v
        if not ug: x=x.detach()
    return x
def vel_loss(ti,k):
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; v=tf_call(d["hs"],ehs,d["ts"],ex); return F.mse_loss(v.float(),d["vT"].float())
@torch.no_grad()
def eval_perc():   # mean LPIPS on val (student x0 decode vs teacher)
    tot=0.0
    for ti in val_tis[:16]:
        xS=roll_to_x0(ti,12,0); IS=vae_decode_ng(xS); tot+=LP(IS, vae_decode_ng(x0T[ti])).mean().item()
    return tot/min(16,len(val_tis))
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

print(f"[init] val-LPIPS {eval_perc():.4f}",flush=True); render("s0")
t0=time.time()
for step in range(1,STEPS+1):
    rmp=min(1.0,step/float(RAMP))
    for p in attn_masters: p.requires_grad_(step>=FREEZE_ATTN)
    opt.zero_grad(set_to_none=True)
    ti=random.choice(calib_tis); k=random.choice(LATE_K)
    xS=roll_to_x0(ti,k,BPTT); xT=x0T[ti]
    Lm=F.mse_loss(xS,xT)/(xT.pow(2).mean()+1e-6)
    IS=vae_decode(xS); Tt=IT[ti]
    Lp=LP(IS,Tt).mean(); Le=edge_loss(IS,Tt); Lpa=patch_loss(IS,Tt); Lpx=(IS-Tt).abs().mean()
    Lv=vel_loss(ti,k)
    L=Lm+0.25*Lv+rmp*(LAM_LPIPS*Lp+LAM_EDGE*Le+LAM_PATCH*Lpa+LAM_PIX*Lpx)
    L.backward(); gn=torch.nn.utils.clip_grad_norm_(late_masters+scales+island_p,0.5)
    if torch.isfinite(gn): opt.step()
    for m in ste_mods: m.weight_master.data.lerp_(m._quantized_matrix_detached(),0.02)
    if step%50==0 or step==1:
        print(f"[s{step}] Lm{Lm.item():.3e} Lv{Lv.item():.3e} LPIPS{Lp.item():.3e} edge{Le.item():.3e} patch{Lpa.item():.3e} gn{gn:.1f} val-LPIPS{eval_perc():.4f} ({time.time()-t0:.0f}s)",flush=True)
    if step%RENDER_EVERY==0: render(f"s{step}"); save_ckpt()
render("final"); save_ckpt(); print("[done]",flush=True)
