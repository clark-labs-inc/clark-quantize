"""Stage 9B-0 — inference-only RATE oracle (NO training).
Question: how many extra FFN / late-block bits recover the FP16 texture (harp strings, wall text,
wine-glass edges, horse detail)? Keep attention = trained ternary g112 (oracle says attention is not
the sharpness bottleneck). Vary FFN / late-5 group size: trained-g112 vs teacher-ternary-{g64,g32} vs
teacher-FP16. Render the held-out montage; pick the cheapest rate that recovers texture.
ENV: CKPT=ckpt_ternary_preview_v0.pt PROMPTS=shared_hard.json OUT=oracle_rate.png
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, json
os.environ["HF_HUB_OFFLINE"]="1"
from diffusers import SanaPipeline
from PIL import Image, ImageDraw
dev="cuda"; DT=torch.bfloat16; MODEL="models/Sana_1600M_512px"; K=20; GUID=4.5
CKPT=os.environ.get("CKPT","ckpt_ternary_preview_v0.pt"); OUT=os.environ.get("OUT","oracle_rate.png")
PROMPTS=json.load(open(os.environ.get("PROMPTS","shared_hard.json")))[:4]; NLATE=5

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))
def _blk(n):
    try: return int(n.split("transformer_blocks.")[1].split(".")[0])
    except: return -1
def is_ffn(n): return ".ff." in n
def is_late(n): return _blk(n)>=20-NLATE

import sys as _sys, os as _os
_sys.path.insert(0,_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from clark_quantize.ternary import ternarize
@torch.no_grad()
def qter(W,g):   # MSE-OPTIMAL ternary at group g: closed-form top-k support + optimal scale (AUDIT FIX: was 0.7xabsmean heuristic, which never chose the support -> understated g32/g64)
    o,i=W.shape; pad=(-i)%g; Wp=torch.cat([W,torch.zeros(o,pad,device=W.device,dtype=W.dtype)],1) if pad else W
    Wq,_=ternarize(Wp.float(), group=g, method="optimal")
    return Wq[:,:i].to(W.dtype)

ck=torch.load(CKPT,map_location="cpu")
def dequant(code,ls,in_f):
    o,ng,g=code.shape; s=torch.exp(ls.float()).unsqueeze(-1); return (code.float()*s).reshape(o,ng*g)[:,:in_f]
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=DT); pipe.to(dev); tf=pipe.transformer
qmods=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
Wfp={n:m.weight.data.clone() for n,m in qmods}                     # teacher FP16 (2D view for convs)
shp2={n:(m.weight.shape, m.weight.data.reshape(m.weight.shape[0],-1).shape) for n,m in qmods}
Wtr={}
for (n,m),code,ls,s in zip(qmods,ck["codes"],ck["log_scales"],ck["shapes"]):
    Wtr[n]=dequant(code,ls,s[1]).reshape(m.weight.shape).to(m.weight.dtype).to(dev)   # trained g112 effective weights
island_p=[(n,p) for n,p in tf.named_parameters() if is_island(n)]
isl_tr=[c.to(dev) for c in ck["islands"]]
print(f"[rate-oracle] {len(qmods)} trunk, {len(island_p)} islands | ckpt step {ck['step']}",flush=True)

_cache={}
def teacher_q(n,g):   # teacher weight ternary-quantized at group g (or FP16 if g==0), cached
    if g==0: return Wfp[n]
    key=(n,g)
    if key not in _cache:
        full=Wfp[n]; W2=full.reshape(full.shape[0],-1)
        _cache[key]=qter(W2,g).reshape(full.shape)
    return _cache[key]

@torch.no_grad()
def set_state(weight_fn):   # weight_fn(n)-> "trained" | ("q",g) | ("fp",) ; islands stay trained
    for n,m in qmods:
        w=weight_fn(n)
        if w=="trained": m.weight.data.copy_(Wtr[n])
        elif w[0]=="q":  m.weight.data.copy_(teacher_q(n,w[1]))
        else:            m.weight.data.copy_(Wfp[n])
    for (n,p),tr in zip(island_p,isl_tr): p.data.copy_(tr)

def ffn_rate(g):       return lambda n: ("q",g) if is_ffn(n) else "trained"
def late_trunk(g):     return lambda n: ("q",g) if is_late(n) else "trained"
def late_ffn(g):       return lambda n: ("q",g) if (is_ffn(n) and is_late(n)) else "trained"
def ffn64_lateffn32():
    return lambda n: ("q",32) if (is_ffn(n) and is_late(n)) else (("q",64) if is_ffn(n) else "trained")
configs=[
    ("REF FP16",            lambda n:("fp",)),
    ("A baseline g112",     lambda n:"trained"),
    ("B FFN g64",           ffn_rate(64)),
    ("C FFN g32",           ffn_rate(32)),
    ("D late5 trunk g64",   late_trunk(64)),
    ("E late5 trunk g32",   late_trunk(32)),
    ("F late5 FFN g32",     late_ffn(32)),
    ("G FFNg64+lateFFNg32", ffn64_lateffn32()),
    ("H FFN FP16",          lambda n:("fp",) if is_ffn(n) else "trained"),
    ("I late5 trunk FP16",  lambda n:("fp",) if is_late(n) else "trained"),
]
@torch.no_grad()
def render(p,seed): return pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(seed)).images[0]

T=320; LBL=170; rows=[]
for name,fn in configs:
    set_state(fn)
    imgs=[render(p,9000+j).resize((T,T)) for j,p in enumerate(PROMPTS)]
    row=Image.new("RGB",(LBL+T*len(PROMPTS),T),(15,15,15)); d=ImageDraw.Draw(row); d.text((8,T//2-8),name,fill=(235,235,235))
    for j,im in enumerate(imgs): row.paste(im,(LBL+j*T,0))
    rows.append(row); print(f"[render] {name}",flush=True)
M=Image.new("RGB",(LBL+T*len(PROMPTS),T*len(rows)),(15,15,15))
for i,r in enumerate(rows): M.paste(r,(0,i*T))
M.save(OUT); print(f"[done] {OUT}",flush=True)
