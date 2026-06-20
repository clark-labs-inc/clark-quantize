"""Stage 9A — Oracle FP16-swap localization (inference only, NO training).
Load FP16 Sana teacher + reconstruct the trained ternary q1024t. For each oracle config, restore a
SUBSET of quantized trunk weights to FP16 teacher weights and render the hard prompts. The config
whose render most restores FP16 sharpness localizes the fidelity gap -> where to spend bits/training.
Output: grid montage (rows=configs, cols=prompts) for visual comparison.
ENV: CKPT=ckpt_ternary_preview_v0.pt PROMPTS=shared_hard.json OUT=oracle_fp16_swap.png
"""
import torch, torch.nn as nn, os, json
os.environ["HF_HUB_OFFLINE"]="1"
from diffusers import SanaPipeline
from PIL import Image, ImageDraw
dev="cuda"; DT=torch.bfloat16; MODEL="models/Sana_1600M_512px"; K=20; GUID=4.5
CKPT=os.environ.get("CKPT","ckpt_ternary_preview_v0.pt"); OUT=os.environ.get("OUT","oracle_fp16_swap.png")
PROMPTS=json.load(open(os.environ.get("PROMPTS","shared_hard.json")))[:4]

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))
def blk(n):
    try: return int(n.split("transformer_blocks.")[1].split(".")[0])
    except: return -1
def is_attn(n): return ".attn1." in n or ".attn2." in n
def is_vo(n): return is_attn(n) and ("to_v" in n or "to_out.0" in n)
def is_qk(n): return is_attn(n) and ("to_q" in n or "to_k" in n)
def is_ffn(n): return ".ff." in n
NLAST=5  # last 5 of 20 blocks = late

ck=torch.load(CKPT,map_location="cpu"); G=ck["group_size"]
def dequant(code,ls,in_f):
    o,ng,g=code.shape; s=torch.exp(ls.float()).unsqueeze(-1); return (code.float()*s).reshape(o,ng*g)[:,:in_f]
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=DT); pipe.to(dev); tf=pipe.transformer
qmods=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
Wfp={n:m.weight.data.clone() for n,m in qmods}                       # FP16 teacher trunk
Wq={}
for (n,m),code,ls,shp in zip(qmods,ck["codes"],ck["log_scales"],ck["shapes"]):
    Wq[n]=dequant(code,ls,shp[1]).reshape(m.weight.shape).to(m.weight.dtype).to(dev)
island_p=[(n,p) for n,p in tf.named_parameters() if is_island(n)]
Wfp_isl={n:p.data.clone() for n,p in island_p}
isl_trained=[c.to(dev) for c in ck["islands"]]
print(f"[oracle] {len(qmods)} trunk mods, {len(island_p)} islands | ckpt step {ck['step']}",flush=True)

@torch.no_grad()
def set_state(trunk_fp16, islands_fp16):   # trunk_fp16(name)->True means keep FP16; else quantized
    for n,m in qmods: m.weight.data.copy_(Wfp[n] if trunk_fp16(n) else Wq[n])
    for (n,p),tr in zip(island_p,isl_trained): p.data.copy_(Wfp_isl[n] if islands_fp16 else tr)

configs=[
    ("FP16 reference",      lambda n: True,  True),
    ("all-ternary (ours)",  lambda n: False, False),
    ("attention FP16",      lambda n: is_attn(n), False),
    ("V/O FP16",            lambda n: is_vo(n),   False),
    ("Q/K FP16",            lambda n: is_qk(n),   False),
    ("FFN FP16",            lambda n: is_ffn(n),  False),
    (f"late {NLAST} blocks FP16", lambda n: blk(n)>=20-NLAST, False),
]
@torch.no_grad()
def render(p,seed): return pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(seed)).images[0]

T=320; LBL=150; rows=[]
for name,tf16,isl16 in configs:
    set_state(tf16,isl16)
    imgs=[render(p,9000+j).resize((T,T)) for j,p in enumerate(PROMPTS)]
    row=Image.new("RGB",(LBL+T*len(PROMPTS),T),(15,15,15)); d=ImageDraw.Draw(row)
    d.text((8,T//2-8),name,fill=(235,235,235))
    for j,im in enumerate(imgs): row.paste(im,(LBL+j*T,0))
    rows.append(row); print(f"[render] {name}",flush=True)
M=Image.new("RGB",(LBL+T*len(PROMPTS),T*len(rows)),(15,15,15))
for i,r in enumerate(rows): M.paste(r,(0,i*T))
M.save(OUT); print(f"[done] {OUT}  (top=FP16 ref; compare rows to it)",flush=True)
