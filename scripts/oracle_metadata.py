"""Metadata-compression oracle (inference-only, NO training). The reframe: ternary q1024t is already
Prism-regime quality; the missing work is 8x STORAGE, not a harder loss. Test whether compressing the
metadata (group scales -> log8 8-bit, islands -> FP8) damages the render. Same ternary CODES throughout.
  M0 ternary + FP16 scales + BF16 islands  (current 6.4x)
  M1 + log8 scales       (scale damage)
  M2 + FP8 islands       (island damage)
  M3 + log8 + FP8        (8x storage candidate: log2 3 + 8/112 ~ 1.657 b/wt trunk, FP8 islands -> ~2.0 global)
Render hard prompts; if M3 ~ M0, we have ternary quality at 8x storage.
ENV: CKPT=ckpt_ternary_preview_v0.pt PROMPTS=shared_hard.json OUT=oracle_metadata.png
"""
import torch, torch.nn as nn, os, json
os.environ["HF_HUB_OFFLINE"]="1"
from diffusers import SanaPipeline
from PIL import Image, ImageDraw
dev="cuda"; DT=torch.bfloat16; MODEL="models/Sana_1600M_512px"; K=20; GUID=4.5
CKPT=os.environ.get("CKPT","ckpt_ternary_preview_v0.pt"); OUT=os.environ.get("OUT","oracle_metadata.png")
PROMPTS=json.load(open(os.environ.get("PROMPTS","shared_hard.json")))[:4]
def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))

ck=torch.load(CKPT,map_location="cpu")
def log8(ls):   # 8-bit logarithmic quant of the (already log-domain) scales: 256 levels over range
    lo,hi=ls.min(),ls.max(); q=torch.round((ls-lo)/(hi-lo+1e-9)*255)/255*(hi-lo)+lo; return q
def fp8(w):     # round-trip through float8_e4m3fn
    return w.to(torch.float8_e4m3fn).to(w.dtype)
def dequant(code,ls,in_f):
    o,ng,g=code.shape; s=torch.exp(ls.float()).unsqueeze(-1); return (code.float()*s).reshape(o,ng*g)[:,:in_f]

pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=DT); pipe.to(dev); tf=pipe.transformer
qmods=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
island_p=[(n,p) for n,p in tf.named_parameters() if is_island(n)]
ISL_fp16=[p.data.clone() for n,p in island_p]; ISL_tr=[c.to(dev) for c in ck["islands"]]
print(f"[meta] {len(qmods)} trunk, {len(island_p)} islands | step {ck['step']}",flush=True)

@torch.no_grad()
def set_state(scale_log8, island_fp8):
    for (n,m),code,ls,s in zip(qmods,ck["codes"],ck["log_scales"],ck["shapes"]):
        L=log8(ls.float()) if scale_log8 else ls.float()
        m.weight.data.copy_(dequant(code,L,s[1]).reshape(m.weight.shape).to(m.weight.dtype).to(dev))
    for (n,p),tr in zip(island_p,ISL_tr): p.data.copy_(fp8(tr) if island_fp8 else tr)
@torch.no_grad()
def render(p,seed): return pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(seed)).images[0]

configs=[("M0 fp16 scale / bf16 isl (6.4x)",False,False),("M1 +log8 scale",True,False),
         ("M2 +fp8 isl",False,True),("M3 log8+fp8 (8x)",True,True)]
T=380; LBL=210; rows=[]
for name,sl,ifp in configs:
    set_state(sl,ifp); imgs=[render(p,9000+j).resize((T,T)) for j,p in enumerate(PROMPTS)]
    row=Image.new("RGB",(LBL+T*len(PROMPTS),T),(15,15,15)); d=ImageDraw.Draw(row); d.text((8,T//2-8),name,fill=(235,235,235))
    for j,im in enumerate(imgs): row.paste(im,(LBL+j*T,0))
    rows.append(row); print(f"[render] {name}",flush=True)
M=Image.new("RGB",(LBL+T*len(PROMPTS),T*len(rows)),(15,15,15))
for i,r in enumerate(rows): M.paste(r,(0,i*T))
M.save(OUT); print(f"[done] {OUT}",flush=True)
