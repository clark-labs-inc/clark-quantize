# Clark Air · Sana 1.6B

**A Sana 1.6B text-to-image transformer compressed to ternary (~2 bits/weight) with
quantization-aware training — 8.6× smaller than FP16, near-FP16 quality, data-free.**

![FP16 vs Clark Air, Sana 1.6B](demo.png)

*FP16 (top) vs **Clark Air** (bottom), Sana 1.6B — identical prompts and seed.*

## What this is

Clark Air takes Sana's diffusion transformer and quantizes its attention + GLU-FFN
trunk weights to **ternary `{-1, 0, +1}`** with one learned log-scale per group of 112.
A small set of conditioning/projection layers (the **islands**, ~5.4% of params —
`patch_embed`, `time_embed`, `caption_projection`, `caption_norm`) are trained and kept
high-precision; `scale_shift_table` and `proj_out` are frozen at their base values.
Training is **quantization-aware** (master weights + straight-through ternary forward)
over a data-free optimal-scale PTQ initialization.

## Footprint (measured)

| Artifact | Size | vs FP16 | What it is |
|---|---|---|---|
| FP16 transformer | 3.21 GB | 1× (100%) | reference |
| **Clark Air — packed** | **374 MB** | **8.6× (≈12%)** | 40 ternary symbols per `uint64` + 6-bit log-scales + FP8 islands, flat `safetensors` |
| Clark Air — unpacked | 3.21 GB | compatibility | dequantized bf16 `diffusers` transformer |

- **Measured 1.85 effective bits/weight → 8.6× smaller** (374 MB packed ÷ 3.21 GB FP16).
- Ternary codes round-trip **exactly** (`roundtrip_bad_modules == 0`); the compact
  container is bit-identical to the reference pack (lossless).
- **Tighter than other shipped ternary image transformers** — Prism Bonsai Image 4B
  is 6.4× (~2.5 eff bpw, FP16 islands), 1.58-bit FLUX is 7.7×. Clark Air reaches 8.6×
  by keeping the conditioning/projection islands at **FP8** (not FP16) plus 6-bit
  log-scales — on a *smaller* model where the island tail is proportionally costlier.

> ⚠️ **Honest scope.** (1) The **unpacked** safetensors is a compatibility format, not
> the compressed size — it dequantizes to bf16 so any `diffusers` install loads it with
> no custom kernels, so it is the *same 3.21 GB* as FP16; the real compressed artifact
> is the **374 MB packed** file. (2) This is **transformer-body, on-disk** compression;
> the full pipeline (text encoder + VAE) adds weight, and there is no 2-bit GPU kernel
> yet, so runtime VRAM is unchanged until a GemLite/MLX 2-bit path lands.

## Using the unpacked artifact (drop-in `diffusers`)

```python
import torch
from diffusers import SanaPipeline, SanaTransformer2DModel

transformer = SanaTransformer2DModel.from_pretrained(
    "clark-labs/clark-quantize-sana-1.6b", subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_512px_diffusers", transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe("a half empty bottle of red wine", num_inference_steps=20, guidance_scale=4.5).images[0].save("out.png")
```

## Details

- **Base model:** Sana 1.6B (512px), `Efficient-Large-Model/Sana_1600M_512px`
- **Quantization:** ternary trunk, group size 112, per-group log-scale; FP8 (e4m3) islands
- **Checkpoint:** QAT step 31000 (`mode=ternary`, `g112`)
- **Quality:** evaluated on held-out hard prompts (text-on-wall, wine bottle, harp,
  horse + person) at 512px / 20 steps / guidance 4.5, same seed as FP16.

## License

Apache-2.0 © Clark Labs, Inc.
