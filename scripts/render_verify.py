"""Render verification: load a trained Sana-Bonsai checkpoint, bake hard codes, render
FP16-vs-quantized side by side at fixed seed. The REAL quality gate (chunk8 != render).
ENV: CKPT=ckpt_bonsai_A1f.pt  MODE=ternary|binary  OUT=render_A1f.png
"""
import torch, torch.nn as nn, torch.nn.functional as F, os
os.environ["HF_HUB_OFFLINE"]="1"
from diffusers import SanaPipeline
from PIL import Image
dev="cuda"; MODEL="models/Sana_1600M_512px"; K=20; G=112; GUID=4.5
MODE=os.environ.get("MODE","ternary"); CKPT=os.environ["CKPT"]; OUT=os.environ.get("OUT","render.png")
import json as _json
_pf=os.environ.get("PROMPTS_FILE","")
if _pf: PROMPTS=_json.load(open(_pf))[:int(os.environ.get("NREND","4"))]   # held-out render prompts
else: PROMPTS=["a red panda astronaut, studio photo","an oil painting of a lighthouse at sunset","a photograph of a golden retriever puppy in a field"]
def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))
def quant_groups(u, mode):
    out,inf=u.shape; ug=u.view(out,inf//G,G)
    if mode=="binary": t=torch.where(ug>=0,1.,-1.); s=ug.abs().mean(-1,keepdim=True)
    else:
        tau=0.7*ug.abs().mean(-1,keepdim=True); t=torch.where(ug>tau,1.,torch.where(ug<-tau,-1.,0.))
        s=(ug.abs()*(t!=0).float()).sum(-1,keepdim=True)/((t!=0).float().sum(-1,keepdim=True).clamp_min(1.))
    return (s*t).view(out,inf)

pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer
# FP16 reference renders first (before any wrapping)
refs=[]
for j,p in enumerate(PROMPTS):
    img=pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(j)).images[0]; refs.append(img)
print("[fp16] rendered refs",flush=True)

# wrap trunk, load checkpoint, BAKE hard codes into weights
QMODS=[]
for n,m in tf.named_modules():
    if is_trunk_lin(n,m) or is_trunk_conv(n,m):
        conv=isinstance(m,nn.Conv2d); w=m.weight.data; w2=w.reshape(w.shape[0],-1) if conv else w
        m._u=w2.clone().float(); m._ksh=tuple(w.shape); m._conv=conv; QMODS.append(m)
island_p=[]   # MUST match training collection order exactly: module-params first, then scale_shift_table
for n,m in tf.named_modules():
    if is_island(n):
        for p in m.parameters(recurse=False): island_p.append(p)
for n,p in tf.named_parameters():
    if "scale_shift_table" in n: island_p.append(p)
ck=torch.load(CKPT,map_location=dev)
print(f"[ckpt] {CKPT} step {ck['step']} | {len(ck['u'])} masters {len(ck['isl'])} islands (QMODS {len(QMODS)} islands {len(island_p)})",flush=True)
with torch.no_grad():
    for m,u in zip(QMODS,ck["u"]): m._u=u.to(dev).float()
    for p,c in zip(island_p,ck["isl"]): p.copy_(c.to(dev))
    for m in QMODS:                       # bake hard quantized weight (alpha=1)
        What=quant_groups(m._u,MODE); m.weight.data=What.view(m._ksh).to(m.weight.dtype)
print(f"[baked] hard {MODE} | GPU {torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)

qs=[]
for j,p in enumerate(PROMPTS):
    img=pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(j)).images[0]; qs.append(img)
print("[quant] rendered",flush=True)
# montage: 2 rows (FP16 top, quant bottom) x len(PROMPTS) cols
W=512; M=Image.new("RGB",(W*len(PROMPTS), W*2),(20,20,20))
for j in range(len(PROMPTS)): M.paste(refs[j],(j*W,0)); M.paste(qs[j],(j*W,W))
M.save(OUT); print(f"[done] saved {OUT} (top=FP16, bottom={MODE} {CKPT} s{ck['step']})",flush=True)
