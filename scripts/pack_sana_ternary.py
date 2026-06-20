"""Reconstruct a trained Clark-Air QAT checkpoint (compact int8 codes + scales + islands) into a
diffusers-loadable Sana transformer, save the UNPACKED FP16 safetensors (compatibility/fallback,
mirrors prism-ml/bonsai-image-ternary-4B-unpacked), and render the hard held-out prompts to verify.
Foundation for all packed variants (MLX/GemLite) — they pack the same codes.
ENV: CKPT=ckpt_ternary_preview_v0.pt OUT=out/clark-air-sana-ternary-unpacked RENDER=1
"""
import torch, torch.nn as nn, os, json
os.environ["HF_HUB_OFFLINE"]="1"
from diffusers import SanaPipeline
from PIL import Image
dev=os.environ.get("DEV","cuda"); DT=torch.bfloat16 if dev=="cuda" else torch.float32
MODEL="models/Sana_1600M_512px"; K=int(os.environ.get("STEPS","20")); GUID=4.5
CKPT=os.environ.get("CKPT","ckpt_ternary_preview_v0.pt"); OUT=os.environ.get("OUT","out/clark-air-sana-ternary-unpacked")
RENDER=int(os.environ.get("RENDER","1")); NREND=int(os.environ.get("NREND","4"))
def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))

ck=torch.load(CKPT,map_location="cpu"); G=ck["group_size"]; MODE=ck["mode"]
print(f"[ckpt] {CKPT} step {ck['step']} mode {MODE} g{G} | {len(ck['codes'])} trunk + {len(ck['islands'])} islands",flush=True)
def dequant(code, logscale, in_features):  # code [out,ng,g] int8, logscale [out,ng] -> [out,in_features]
    out,ng,g=code.shape; s=torch.exp(logscale.float()).unsqueeze(-1)  # [out,ng,1]
    q=(code.float()*s).reshape(out,ng*g)[:,:in_features]; return q

pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=DT); pipe.to(dev); tf=pipe.transformer
qmods=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
assert len(qmods)==len(ck["codes"]), f"trunk count mismatch {len(qmods)} vs {len(ck['codes'])}"
nz=ntot=0
with torch.no_grad():
    for (n,m),code,ls,shp in zip(qmods,ck["codes"],ck["log_scales"],ck["shapes"]):
        in_features=shp[1]; q=dequant(code,ls,in_features)               # [out,in]
        if isinstance(m,nn.Conv2d): m.weight.data.copy_(q.reshape(m.weight.shape).to(m.weight.dtype))
        else: m.weight.data.copy_(q.to(m.weight.dtype))
        nz+=(code!=0).sum().item(); ntot+=code.numel()
    island_p=[p for nm,p in tf.named_parameters() if is_island(nm)]
    assert len(island_p)==len(ck["islands"]), f"island count mismatch {len(island_p)} vs {len(ck['islands'])}"
    for p,isl in zip(island_p,ck["islands"]): p.data.copy_(isl.to(p.dtype).to(dev))
print(f"[reconstruct] {ntot/1e6:.0f}M trunk weights | nonzero {100*nz/ntot:.1f}% (zeros {100*(1-nz/ntot):.1f}%) | bits/wt~{(nz/ntot)*1.585+16/G:.3f}",flush=True)

os.makedirs(OUT,exist_ok=True)
tf.save_pretrained(os.path.join(OUT,"transformer"))
print(f"[unpacked] saved diffusers transformer -> {OUT}/transformer",flush=True)
# size report
def dirsize(p): return sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(p) for f in fs)
print(f"[size] unpacked transformer {dirsize(os.path.join(OUT,'transformer'))/1e9:.2f} GB",flush=True)

if RENDER:
    hard=json.load(open("shared_hard.json"))[:NREND]
    refs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hard)]
    W=512; M=Image.new("RGB",(W*len(hard),W),(20,20,20))
    for j in range(len(hard)): M.paste(refs[j],(j*W,0))
    M.save("render_packed_verify.png"); print("[render] render_packed_verify.png (reconstructed model)",flush=True)
print("[done]",flush=True)
