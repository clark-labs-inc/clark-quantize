"""Load the Clark Air packed safetensors (the real compressed artifact), reconstruct the
Sana transformer, and render prompts. This is the loader/decoder for the packed format —
unlike the unpacked `transformer/` (bf16 compatibility copy), this reads the 374 MB packed
container, unpacks the ternary codes + group scales + FP8 islands, and runs the model.

ENV: PACKED=...-packed.safetensors OUT=render_packed_model.png NPROMPT=4 DEV=cuda|cpu
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import SanaPipeline
from PIL import Image
from safetensors import safe_open

PACKED = os.environ.get("PACKED", "clark-air-sana-1.6b-packed.safetensors")
MODEL = os.environ.get("MODEL", "models/Sana_1600M_512px")
OUT = os.environ.get("OUT", "render_packed_model.png")
PROMPTS_FILE = os.environ.get("PROMPTS", "shared_hard.json")
DEV = os.environ.get("DEV", "cuda")
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
K = int(os.environ.get("STEPS", "20"))
GUID = float(os.environ.get("GUID", "4.5"))
NPROMPT = int(os.environ.get("NPROMPT", "4"))
SEED0 = int(os.environ.get("SEED", "9000"))
COLS = int(os.environ.get("COLS", "0")) or NPROMPT


def unpack_trits_40(packed, shape, pad):
    vals = packed.astype(np.uint64).copy()
    out = np.empty(vals.size * 40, dtype=np.int8)
    for i in range(40):
        out[i::40] = (vals % 3).astype(np.int8)
        vals //= 3
    if pad:
        out = out[:-pad]
    return (out.reshape(shape) - 1).astype(np.int8)


def is_trunk_lin(n, m):
    return isinstance(m, nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) \
        and any(s in n for s in ("to_q", "to_k", "to_v", "to_out.0"))


def is_trunk_conv(n, m):
    return isinstance(m, nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)


def predicate(n, m):
    return is_trunk_lin(n, m) or is_trunk_conv(n, m)


def dequant_weight(code, logscale, in_features):
    code = torch.as_tensor(code)
    out, ng, group = code.shape
    scale = torch.exp(torch.as_tensor(logscale).float()).unsqueeze(-1)
    return (code.float() * scale).reshape(out, ng * group)[:, :in_features]


# --- decode the packed container ---
code_by_name, scale_by_name, island_by_name = {}, {}, {}
with safe_open(PACKED, framework="pt") as f:
    md = f.metadata()
    for i, cm in enumerate(json.loads(md["codes"])):
        packed = np.frombuffer(f.get_tensor(f"code.{i}").numpy().tobytes(), dtype=np.uint64)
        code = unpack_trits_40(packed, tuple(cm["code_shape"]), cm["pad"])
        code_by_name[cm["name"]] = (code, tuple(cm["weight_shape"]))
    for i, sm in enumerate(json.loads(md["scales"])):
        q = f.get_tensor(f"scale.{i}").to(torch.float32).numpy()
        levels = (1 << int(sm["bits"])) - 1
        logscale = sm["lo"] + q / levels * (sm["hi"] - sm["lo"])
        scale_by_name[sm["name"]] = logscale.reshape(sm["shape"])
    for im in json.loads(md["islands"]):
        raw = f.get_tensor(f"island.{im['name']}")  # uint8
        island_by_name[im["name"]] = raw.contiguous().view(torch.float8_e4m3fn).reshape(im["shape"])
print(f"[decode] {len(code_by_name)} trunk modules, {len(island_by_name)} islands from {PACKED}", flush=True)

pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=DT)
pipe.to(DEV)
tf = pipe.transformer
applied = 0
with torch.no_grad():
    for n, m in tf.named_modules():
        if predicate(n, m):
            code, wshape = code_by_name[n]
            q = dequant_weight(code, scale_by_name[n], wshape[1])
            if isinstance(m, nn.Conv2d):
                m.weight.data.copy_(q.reshape(m.weight.shape).to(m.weight.dtype).to(DEV))
            else:
                m.weight.data.copy_(q.to(m.weight.dtype).to(DEV))
            applied += 1
    for n, p in tf.named_parameters():
        if n in island_by_name:
            p.data.copy_(island_by_name[n].to(p.dtype).to(DEV))
print(f"[apply] {applied} trunk modules + {len(island_by_name)} islands -> transformer", flush=True)

prompts = json.load(open(PROMPTS_FILE))[:NPROMPT]
imgs = [
    pipe(prompt=pr, num_inference_steps=K, height=512, width=512, guidance_scale=GUID,
         generator=torch.Generator(DEV).manual_seed(SEED0 + j)).images[0]
    for j, pr in enumerate(prompts)
]
rows = (len(imgs) + COLS - 1) // COLS
grid = Image.new("RGB", (512 * COLS, 512 * rows), (20, 20, 20))
for j, im in enumerate(imgs):
    grid.paste(im, (512 * (j % COLS), 512 * (j // COLS)))
grid.save(OUT)
print(f"[render] {OUT}  ({len(imgs)} prompts from the PACKED safetensors)", flush=True)
