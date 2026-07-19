"""Metadata-compression oracle (inference-only, NO training). The reframe: the all-g112 ternary champion
is the quality foundation; the missing work is 8x STORAGE, not a harder loss. Test whether compressing the
metadata (group scales -> logN bits, islands -> FP8) damages the render. Same ternary CODES throughout.
  M0 ternary + FP16 scales + BF16 islands  (current 6.4x)
  M1 + log8 scales       (scale damage)
  M2 + FP8 islands       (island damage)
  M3 + log8 + FP8        (8x candidate with entropy trits, ~7.95x with 40-trit packs)
  M4 + log6 + FP8        (strict 8x candidate with 40-trit packs)
Render hard prompts; if M3/M4 ~ M0, we have ternary quality at 8x storage.
ENV: CKPT=ckpt_ternary_preview_v0.pt PROMPTS=shared_hard.json OUT=oracle_metadata.png
"""
import math, torch, torch.nn as nn, os, sys, json
os.environ["HF_HUB_OFFLINE"]="1"
from diffusers import SanaPipeline
from PIL import Image, ImageDraw
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize.sana_ckpt import load_compact_checkpoint, validate_quant_names
dev="cuda"; DT=torch.bfloat16; MODEL="models/Sana_1600M_512px"; K=20; GUID=4.5
CKPT=os.environ.get("CKPT","ckpt_ternary_preview_v0.pt"); OUT=os.environ.get("OUT","oracle_metadata.png")
LEDGER=os.environ.get("LEDGER",os.path.splitext(OUT)[0]+"_ledger.json")
SCALE_QLO=float(os.environ.get("SCALE_QLO","0.001")); SCALE_QHI=float(os.environ.get("SCALE_QHI","0.999"))
TRUNK_FRAC=float(os.environ.get("TRUNK_FRAC","0.946")); ISLAND_FRAC=float(os.environ.get("ISLAND_FRAC","0.054"))
PROMPTS=json.load(open(os.environ.get("PROMPTS","shared_hard.json")))[:4]
def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out","conv_depth"))

ck=load_compact_checkpoint(CKPT)
GROUP_SIZE=int(ck.get("group_size",112))
def ledger(scale_bits, island_bits, code_bpw):
    b_trunk=code_bpw+(scale_bits/GROUP_SIZE)
    b_global=TRUNK_FRAC*b_trunk+ISLAND_FRAC*island_bits
    return {"trunk_bpw":b_trunk,"global_bpw":b_global,"compression":16.0/b_global}
def quantize_log_scales(ls, bits):   # uniform quantization in log space, clipped by robust quantiles
    x=ls.float(); levels=(1 << bits)-1
    lo=torch.quantile(x.flatten(),SCALE_QLO); hi=torch.quantile(x.flatten(),SCALE_QHI)
    if (hi-lo).abs()<1e-8:
        return torch.full_like(x,lo)
    y=((x-lo)/(hi-lo)*levels).round().clamp(0,levels)
    return lo+y/levels*(hi-lo)
def fp8(w):     # round-trip through float8_e4m3fn
    return w.to(torch.float8_e4m3fn).to(w.dtype)
def dequant(code,ls,in_f):
    o,ng,g=code.shape; s=torch.exp(ls.float()).unsqueeze(-1); return (code.float()*s).reshape(o,ng*g)[:,:in_f]

pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=DT); pipe.to(dev); tf=pipe.transformer
qmods=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
validate_quant_names(ck,[n for n,_ in qmods])
island_p=[(n,p) for n,p in tf.named_parameters() if is_island(n)]
ck_islands=ck.get("islands",{})
missing_islands=[]
ISL_tr=[]
for n,p in island_p:
    src=ck_islands.get(n) if isinstance(ck_islands,dict) else None
    if src is None:
        src=p.data.detach().cpu().clone()
        missing_islands.append(n)
    ISL_tr.append(src.to(dev))
print(f"[meta] {len(qmods)} trunk, {len(island_p)} islands | stored {len(island_p)-len(missing_islands)} missing/frozen {len(missing_islands)} | step {ck['step']}",flush=True)
if missing_islands:
    print(f"[meta] frozen island fallback first={missing_islands[0]}",flush=True)

@torch.no_grad()
def set_state(scale_bits, island_fp8):
    for (n,m),code,ls,s in zip(qmods,ck["codes"],ck["log_scales"],ck["shapes"]):
        L=quantize_log_scales(ls.float(),scale_bits) if scale_bits else ls.float()
        m.weight.data.copy_(dequant(code,L,s[1]).reshape(m.weight.shape).to(m.weight.dtype).to(dev))
    for (n,p),tr in zip(island_p,ISL_tr): p.data.copy_(fp8(tr) if island_fp8 else tr)
@torch.no_grad()
def render(p,seed): return pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(seed)).images[0]

configs=[
    ("M0 fp16 scale / bf16 isl",16,False),
    ("M1 log8 scale / bf16 isl",8,False),
    ("M2 fp16 scale / fp8 isl",16,True),
    ("M3 log8+fp8",8,True),
    ("M4 log6+fp8",6,True),
]
ledger_rows=[]
for name,sbits,ifp in configs:
    ibits=8 if ifp else 16
    exact=ledger(sbits,ibits,math.log2(3))
    pack40=ledger(sbits,ibits,64/40)
    ledger_rows.append({"name":name,"scale_bits":sbits,"island_bits":ibits,
                        "entropy_trit":exact,"pack40_trit":pack40})
    print(f"[ledger] {name} exact={exact['compression']:.3f}x ({exact['global_bpw']:.3f} bpw) "
          f"pack40={pack40['compression']:.3f}x ({pack40['global_bpw']:.3f} bpw)",flush=True)
T=380; LBL=210; rows=[]
for name,sbits,ifp in configs:
    set_state(None if sbits==16 else sbits,ifp); imgs=[render(p,9000+j).resize((T,T)) for j,p in enumerate(PROMPTS)]
    row=Image.new("RGB",(LBL+T*len(PROMPTS),T),(15,15,15)); d=ImageDraw.Draw(row); d.text((8,T//2-8),name,fill=(235,235,235))
    for j,im in enumerate(imgs): row.paste(im,(LBL+j*T,0))
    rows.append(row); print(f"[render] {name}",flush=True)
M=Image.new("RGB",(LBL+T*len(PROMPTS),T*len(rows)),(15,15,15))
for i,r in enumerate(rows): M.paste(r,(0,i*T))
M.save(OUT)
with open(LEDGER,"w") as f:
    json.dump({"checkpoint":CKPT,"group_size":GROUP_SIZE,"trunk_frac":TRUNK_FRAC,
               "island_frac":ISLAND_FRAC,"scale_qlo":SCALE_QLO,"scale_qhi":SCALE_QHI,
               "rows":ledger_rows},f,indent=2,sort_keys=True)
    f.write("\n")
print(f"[done] {OUT}",flush=True)
print(f"[done] {LEDGER}",flush=True)
