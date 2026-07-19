"""Prism-endpoint inference oracle (NO training). The density LOSS can't move support (numel dilution).
So test the hypothesis directly: CONSTRUCT the Prism endpoint from the FP16 teacher and render.
  FP16 ref
  q1024t (our trained ternary, ~47.7% zeros, natural scale)
  PTQ ternary g112 (~52% active, natural scale)  [standard collapse baseline]
  Prism endpoint: 38% zeros (top-62% magnitude per group) + 0.625x scale + FP16 signs
If the endpoint row is sharper than q1024t -> densify+shrink IS the lever (build per-element training).
If not -> capacity ceiling, take the recipe to FLUX.
ENV: CKPT=ckpt_ternary_preview_v0.pt PROMPTS=shared_hard.json OUT=oracle_endpoint.png TZERO=0.38 SR=0.625
"""
import torch, torch.nn as nn, os, sys, json
os.environ["HF_HUB_OFFLINE"]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize.ternary import ternarize
from clark_quantize.sana_ckpt import island_values, load_compact_checkpoint, validate_quant_names
from diffusers import SanaPipeline
from PIL import Image, ImageDraw
dev="cuda"; DT=torch.bfloat16; MODEL="models/Sana_1600M_512px"; K=20; GUID=4.5; G=112
CKPT=os.environ.get("CKPT","ckpt_ternary_preview_v0.pt"); OUT=os.environ.get("OUT","oracle_endpoint.png")
PROMPTS=json.load(open(os.environ.get("PROMPTS","shared_hard.json")))[:4]
TZERO=float(os.environ.get("TZERO","0.38")); SR=float(os.environ.get("SR","0.625"))
def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))

@torch.no_grad()
def ptq_natural(W):   # MSE-OPTIMAL ternary g112 PTQ baseline (AUDIT FIX: was 0.7xabsmean heuristic that never chose support)
    Wq,_=ternarize(W.float(), group=G, method="optimal")
    return Wq.to(W.dtype)
@torch.no_grad()
def endpoint(W, tzero, sr):   # top-(1-tzero) magnitude per group active, scale = sr * absmean(active)
    o,i=W.shape; Wg=W.float().view(o,i//G,G); a=Wg.abs()
    kth=torch.quantile(a, tzero, dim=-1, keepdim=True)   # tzero-quantile threshold per group
    m=(a>=kth).float()
    s=sr*((a*m).sum(-1,keepdim=True)/m.sum(-1,keepdim=True).clamp_min(1.))
    return (s*torch.where(Wg>=0,1.,-1.)*m).view(o,i).to(W.dtype)

ck=load_compact_checkpoint(CKPT)
def dequant(code,ls,in_f):
    o,ng,g=code.shape; s=torch.exp(ls.float()).unsqueeze(-1); return (code.float()*s).reshape(o,ng*g)[:,:in_f]
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=DT); pipe.to(dev); tf=pipe.transformer
qmods=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
validate_quant_names(ck,[n for n,_ in qmods])
Wfp={n:m.weight.data.clone() for n,m in qmods}
Wq={}
for (n,m),code,ls,s in zip(qmods,ck["codes"],ck["log_scales"],ck["shapes"]):
    Wq[n]=dequant(code,ls,s[1]).reshape(m.weight.shape).to(m.weight.dtype).to(dev)
island_p=[(n,p) for n,p in tf.named_parameters() if is_island(n)]
ISL_tr=[c.to(dev) for c in island_values(ck,island_p)]; ISL_fp=[p.data.clone() for n,p in island_p]
print(f"[endpoint] {len(qmods)} trunk | step {ck['step']} | tzero {TZERO} sr {SR}",flush=True)

@torch.no_grad()
def set_state(mode):
    for (n,m) in qmods:
        if mode=="fp16": m.weight.data.copy_(Wfp[n])
        elif mode=="q1024t": m.weight.data.copy_(Wq[n])
        elif mode=="ptq": m.weight.data.copy_(ptq_natural(Wfp[n].reshape(m.weight.shape[0],-1)).reshape(m.weight.shape))
        else: m.weight.data.copy_(endpoint(Wfp[n].reshape(m.weight.shape[0],-1),TZERO,SR).reshape(m.weight.shape))
    for (n,p),tr,fp in zip(island_p,ISL_tr,ISL_fp): p.data.copy_(fp if mode=="fp16" else tr)
@torch.no_grad()
def render(p,seed): return pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(seed)).images[0]

configs=[("FP16 ref","fp16"),("q1024t (47.7% zero)","q1024t"),("PTQ ternary","ptq"),(f"Prism endpoint ({int(TZERO*100)}% zero, {SR}x)","endpoint")]
T=360; LBL=220; rows=[]
for name,mode in configs:
    set_state(mode); imgs=[render(p,9000+j).resize((T,T)) for j,p in enumerate(PROMPTS)]
    row=Image.new("RGB",(LBL+T*len(PROMPTS),T),(15,15,15)); d=ImageDraw.Draw(row); d.text((8,T//2-8),name,fill=(235,235,235))
    for j,im in enumerate(imgs): row.paste(im,(LBL+j*T,0))
    rows.append(row); print(f"[render] {name}",flush=True)
M=Image.new("RGB",(LBL+T*len(PROMPTS),T*len(rows)),(15,15,15))
for i,r in enumerate(rows): M.paste(r,(0,i*T))
M.save(OUT); print(f"[done] {OUT}",flush=True)
