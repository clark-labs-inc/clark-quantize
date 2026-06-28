"""Quality-vs-bitwidth comparison for Clark Air Sana 1.6B.

Renders, on shared held-out prompts with one fixed seed, a labeled grid:
    fp16 base | naive 3-bit PTQ | naive 2-bit PTQ | Clark Air 1.58-bit (trained ternary)
to show where naive low-bit collapses and how close the *trained* ternary sits to fp16.

The naive columns are per-group affine PTQ of the BASE weights (clark_quantize.affine_nbits);
the 1.58-bit column is our released model, reconstructed from the GemLite pack (trained islands
+ ternary codes × scales) dequantized to bf16 — same math as the kernel, no CUDA pack needed.

Uses model-CPU-offload so it fits in a few GB of free VRAM alongside other GPU jobs.

ENV: PACK=clark-air-sana-1.6b-gemlite-2bit/transformer/gemlite_pack.safetensors
     MODEL=models/Sana_1600M_512px  OUT=comparison_lowbit.png
     WIDTHS=3,2  NREND=3  STEPS=20  GUID=4.5  SEED=9000  PROMPTS=shared_hard.json   (CUDA)
"""
import json, os
import numpy as np, torch, torch.nn as nn
os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import SanaPipeline
from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFont
from clark_quantize.affine_nbits import quantize_affine_nbits, pow2_group

PACK = os.environ.get("PACK", "clark-air-sana-1.6b-gemlite-2bit/transformer/gemlite_pack.safetensors")
MODEL = os.environ.get("MODEL", "models/Sana_1600M_512px")
OUT = os.environ.get("OUT", "comparison_lowbit.png")
WIDTHS = [int(x) for x in os.environ.get("WIDTHS", "3,2").split(",")]
NREND = int(os.environ.get("NREND", "3")); SEED = int(os.environ.get("SEED", "9000"))
K = int(os.environ.get("STEPS", "20")); GUID = float(os.environ.get("GUID", "4.5"))
DEV = "cuda"


def unpack2b(packed, out, inn):  # uint8 [out, in/4] -> ternary int8 [out, in]
    b = packed.astype(np.uint8); lev = np.empty((out, inn), np.int8)
    lev[:, 0::4] = b & 3; lev[:, 1::4] = (b >> 2) & 3; lev[:, 2::4] = (b >> 4) & 3; lev[:, 3::4] = (b >> 6) & 3
    return lev.astype(np.int8) - 1


def is_trunk_lin(n, m):
    return isinstance(m, nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) \
        and any(s in n for s in ("to_q", "to_k", "to_v", "to_out.0"))
def is_trunk_conv(n, m):
    return isinstance(m, nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n, m): return is_trunk_lin(n, m) or is_trunk_conv(n, m)


def render(pipe, prompts):
    return [pipe(prompt=p, num_inference_steps=K, height=512, width=512, guidance_scale=GUID,
                 generator=torch.Generator(DEV).manual_seed(SEED + j)).images[0] for j, p in enumerate(prompts)]


def main():
    pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()  # fit in a few GB of free VRAM next to other GPU jobs
    tf = pipe.transformer
    prompts = json.load(open(os.environ.get("PROMPTS", "shared_hard.json")))[:NREND]
    trunk = [(n, m) for n, m in tf.named_modules() if predicate(n, m)]
    orig = {n: m.weight.detach().clone() for n, m in trunk}

    def restore():
        for n, m in trunk: m.weight.data.copy_(orig[n])

    columns = []
    print("[render] fp16 base", flush=True)
    columns.append(("fp16 base", render(pipe, prompts)))

    for nb in WIDTHS:
        restore()
        for n, m in trunk:
            shp = m.weight.shape
            W2 = m.weight.data.reshape(shp[0], -1) if m.weight.dim() == 4 else m.weight.data
            m.weight.data.copy_(quantize_affine_nbits(W2, nb).reshape(shp))
        print(f"[render] naive {nb}-bit PTQ", flush=True)
        columns.append((f"naive {nb}-bit PTQ", render(pipe, prompts)))

    # Clark Air 1.58-bit: the released model from the GemLite pack (trained islands +
    # ternary codes × scales), dequantized to bf16 — identical to the kernel's math.
    restore()
    sd = load_file(PACK)
    base_sd = tf.state_dict()
    for k, v in sd.items():
        if "::" in k: continue
        if k in base_sd and base_sd[k].shape == v.shape: base_sd[k].copy_(v.to(base_sd[k].dtype))
    tf.load_state_dict(base_sd, strict=False)
    for n, m in trunk:
        out_, inn_ = [int(x) for x in sd[n + "::shape"].tolist()]
        code = torch.from_numpy(unpack2b(sd[n + "::codes2b"].numpy(), out_, inn_)).float()
        scale = sd[n + "::scale"].float(); ng = scale.shape[1]; g = inn_ // ng
        deq = (code.reshape(out_, ng, g) * scale.reshape(out_, ng, 1)).reshape(out_, inn_)
        m.weight.data.copy_(deq.reshape(m.weight.shape).to(m.weight.dtype))
    print("[render] Clark Air 1.58-bit", flush=True)
    columns.append(("Clark Air 1.58-bit", render(pipe, prompts)))

    # compose labeled grid: columns = variants, rows = prompts
    cell, header = 384, 46
    ncol, nrow = len(columns), len(prompts)
    canvas = Image.new("RGB", (ncol * cell, header + nrow * cell), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for c, (label, imgs) in enumerate(columns):
        d.text((c * cell + 10, 13), label, fill=(235, 235, 235), font=font)
        for r, im in enumerate(imgs):
            canvas.paste(im.resize((cell, cell)), (c * cell, header + r * cell))
    canvas.save(OUT)
    print(f"[grid] {OUT}  ({ncol} cols x {nrow} rows)", flush=True)


if __name__ == "__main__":
    main()
