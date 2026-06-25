"""Pack the Sana ternary champion into a strict 8x storage candidate.

Format:
  trunk codes: 40 ternary symbols per uint64, {-1,0,+1} -> {0,1,2}
  scales: per-tensor uniform log-space quantization, default 6 bits
  islands: FP8 e4m3 round-trip tensors

The script also reloads the packed payload, reconstructs the transformer, and
renders the held-out hard prompts as a sanity check.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import SanaPipeline
from PIL import Image

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize.sana_ckpt import load_compact_checkpoint, validate_quant_names

MODEL = os.environ.get("MODEL", "models/Sana_1600M_512px")
CKPT = os.environ.get("CKPT", "ckpt_sana_t112_champion.pt")
OUT = os.environ.get("OUT", "packed_sana_t112_champion_m4_log6_fp8.pt")
RENDER_OUT = os.environ.get("RENDER_OUT", "render_packed_sana_t112_champion_m4.png")
MANIFEST_OUT = os.environ.get("MANIFEST_OUT", os.path.splitext(OUT)[0] + ".json")
PROMPTS_FILE = os.environ.get("PROMPTS", "shared_hard.json")
SCALE_BITS = int(os.environ.get("SCALE_BITS", "6"))
SCALE_QLO = float(os.environ.get("SCALE_QLO", "0"))
SCALE_QHI = float(os.environ.get("SCALE_QHI", "1"))
RENDER = int(os.environ.get("RENDER", "1"))
DEV = os.environ.get("DEV", "cuda")
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
K = int(os.environ.get("STEPS", "20"))
GUID = float(os.environ.get("GUID", "4.5"))
TRUNK_FRAC = float(os.environ.get("TRUNK_FRAC", "0.946"))
ISLAND_FRAC = float(os.environ.get("ISLAND_FRAC", "0.054"))

POW3_40 = np.array([3 ** i for i in range(40)], dtype=np.uint64)


def is_trunk_lin(n, m):
    return (
        isinstance(m, nn.Linear)
        and n.startswith("transformer_blocks")
        and (".attn1." in n or ".attn2." in n)
        and any(s in n for s in ("to_q", "to_k", "to_v", "to_out.0"))
    )


def is_trunk_conv(n, m):
    return isinstance(m, nn.Conv2d) and n.startswith("transformer_blocks") and (
        ".ff.conv_inverted" in n or ".ff.conv_point" in n
    )


def predicate(n, m):
    return is_trunk_lin(n, m) or is_trunk_conv(n, m)


def is_island(n):
    keys = (
        "patch_embed",
        "time_embed",
        "caption_projection",
        "caption_norm",
        "scale_shift_table",
        "norm_out",
        "proj_out",
        "conv_depth",
    )
    return any(k in n for k in keys)


def pack_trits_40(codes_np):
    flat = codes_np.reshape(-1).astype(np.int8)
    trits = (flat + 1).astype(np.uint64)
    pad = (-trits.size) % 40
    if pad:
        trits = np.concatenate([trits, np.ones(pad, dtype=np.uint64)])
    chunks = trits.reshape(-1, 40)
    packed = (chunks * POW3_40[None, :]).sum(axis=1, dtype=np.uint64)
    return packed, tuple(codes_np.shape), int(pad)


def unpack_trits_40(packed, shape, pad):
    vals = packed.astype(np.uint64).copy()
    out = np.empty(vals.size * 40, dtype=np.int8)
    for i in range(40):
        out[i::40] = (vals % 3).astype(np.int8)
        vals //= 3
    if pad:
        out = out[:-pad]
    return (out.reshape(shape) - 1).astype(np.int8)


@torch.no_grad()
def quantize_log_scales(log_scales, bits):
    x = log_scales.float()
    levels = (1 << bits) - 1
    lo = torch.quantile(x.flatten(), SCALE_QLO)
    hi = torch.quantile(x.flatten(), SCALE_QHI)
    if (hi - lo).abs() < 1e-8:
        q = torch.zeros_like(x, dtype=torch.uint8)
        deq = torch.full_like(x, lo)
    else:
        y = ((x - lo) / (hi - lo) * levels).round().clamp(0, levels)
        q = y.to(torch.uint8)
        deq = lo + y / levels * (hi - lo)
    return deq, {"bits": bits, "lo": float(lo.cpu()), "hi": float(hi.cpu()), "q": q.cpu()}


def dequant_log_scales(meta):
    q = meta["q"].float()
    levels = (1 << int(meta["bits"])) - 1
    lo = float(meta["lo"])
    hi = float(meta["hi"])
    return lo + q / levels * (hi - lo)


def dequant_weight(code, logscale, in_features):
    out, ng, group = code.shape
    scale = torch.exp(logscale.float()).unsqueeze(-1)
    return (code.float() * scale).reshape(out, ng * group)[:, :in_features]


def render_grid(pipe, prompts, out_path):
    imgs = [
        pipe(
            prompt=p,
            num_inference_steps=K,
            height=512,
            width=512,
            guidance_scale=GUID,
            generator=torch.Generator(DEV).manual_seed(9000 + j),
        ).images[0]
        for j, p in enumerate(prompts)
    ]
    grid = Image.new("RGB", (512 * len(imgs), 512), (20, 20, 20))
    for j, img in enumerate(imgs):
        grid.paste(img, (512 * j, 0))
    grid.save(out_path)


def main():
    ck = load_compact_checkpoint(CKPT)
    group_size = int(ck.get("group_size", 112))
    pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=DT)
    pipe.to(DEV)
    tf = pipe.transformer
    qmods = [(n, m) for n, m in tf.named_modules() if predicate(n, m)]
    validate_quant_names(ck, [n for n, _ in qmods])
    island_p = [(n, p) for n, p in tf.named_parameters() if is_island(n)]
    ck_islands = ck.get("islands", {})

    codes = []
    scales = []
    trunk_params = 0
    code_bits = 0
    scale_bits = 0
    roundtrip_bad = 0
    for (name, _module), code, logscale, shape in zip(qmods, ck["codes"], ck["log_scales"], ck["shapes"]):
        arr = code.cpu().numpy().astype(np.int8)
        packed, code_shape, pad = pack_trits_40(arr)
        if not np.array_equal(unpack_trits_40(packed, code_shape, pad), arr):
            roundtrip_bad += 1
        _deq, smeta = quantize_log_scales(logscale, SCALE_BITS)
        codes.append({"name": name, "packed": packed, "code_shape": code_shape, "pad": pad, "weight_shape": tuple(shape)})
        scales.append({"name": name, **smeta})
        trunk_params += int(np.prod(tuple(shape)))
        code_bits += int(packed.size * 64)
        scale_bits += int(smeta["q"].numel() * SCALE_BITS)

    islands = {}
    island_params = 0
    for name, p in island_p:
        src = ck_islands.get(name) if isinstance(ck_islands, dict) else None
        if src is None:
            src = p.detach().cpu()
        fp8 = src.to(torch.float8_e4m3fn).cpu()
        islands[name] = fp8
        island_params += p.numel()

    payload = {
        "format": "sana_t112_40trit_logscale_fp8_v1",
        "created_at_unix": time.time(),
        "source_ckpt": CKPT,
        "mode": ck.get("mode", "ternary"),
        "step": int(ck.get("step", 0)),
        "group_size": group_size,
        "scale_bits": SCALE_BITS,
        "scale_qlo": SCALE_QLO,
        "scale_qhi": SCALE_QHI,
        "codes": codes,
        "scales": scales,
        "islands": islands,
        "quant_names": [n for n, _ in qmods],
        "island_names": [n for n, _ in island_p],
    }
    torch.save(payload, OUT)

    island_bits = island_params * 8
    actual_bits = code_bits + scale_bits + island_bits
    actual_params = trunk_params + island_params
    actual_bpw = actual_bits / max(actual_params, 1)
    model_bpw = TRUNK_FRAC * (64 / 40 + SCALE_BITS / group_size) + ISLAND_FRAC * 8
    manifest = {
        "format": payload["format"],
        "source_ckpt": CKPT,
        "packed": OUT,
        "render": RENDER_OUT if RENDER else None,
        "scale_bits": SCALE_BITS,
        "scale_qlo": SCALE_QLO,
        "scale_qhi": SCALE_QHI,
        "group_size": group_size,
        "trunk_params": trunk_params,
        "island_params": island_params,
        "code_bits": code_bits,
        "scale_bits_total": scale_bits,
        "island_bits": island_bits,
        "actual_payload_bpw": actual_bpw,
        "model_fraction_bpw": model_bpw,
        "model_fraction_compression": 16.0 / model_bpw,
        "file_size_bytes": os.path.getsize(OUT),
        "roundtrip_bad_modules": roundtrip_bad,
    }
    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    if RENDER:
        loaded = torch.load(OUT, map_location="cpu", weights_only=False)
        scale_by_name = {s["name"]: s for s in loaded["scales"]}
        code_by_name = {c["name"]: c for c in loaded["codes"]}
        with torch.no_grad():
            for name, module in qmods:
                cmeta = code_by_name[name]
                smeta = scale_by_name[name]
                code = torch.from_numpy(unpack_trits_40(cmeta["packed"], tuple(cmeta["code_shape"]), cmeta["pad"]))
                logscale = dequant_log_scales(smeta)
                q = dequant_weight(code, logscale, cmeta["weight_shape"][1])
                module.weight.data.copy_(q.reshape(module.weight.shape).to(module.weight.dtype).to(DEV))
            for name, p in island_p:
                p.data.copy_(loaded["islands"][name].to(DEV).to(p.dtype))
        prompts = json.load(open(PROMPTS_FILE))[:4]
        render_grid(pipe, prompts, RENDER_OUT)
        print(f"[render] {RENDER_OUT}", flush=True)


if __name__ == "__main__":
    main()
