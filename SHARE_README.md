# Clark Air · Sana 1.6B

**A Sana 1.6B text-to-image transformer compressed to ternary (~2 bits/weight) with
quantization-aware training — 6.2× smaller than FP16, near-FP16 quality, data-free.**

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
| **Clark Air — packed** | **517 MB** | **6.2× (≈16%)** | 40 ternary symbols per `uint64` + 6-bit log-scales + FP8 islands |
| Clark Air — unpacked | 3.21 GB | compatibility | dequantized bf16 `diffusers` transformer |

- **Measured 1.85 bits/weight** (≈8× by information content; the 6.2× / 517 MB figure
  is the actual packed file on disk ÷ the FP16 transformer).
- Ternary codes round-trip **exactly** (`roundtrip_bad_modules == 0`).

> ⚠️ **The unpacked safetensors is a compatibility format, not the compressed size.**
> It dequantizes the ternary weights back to bf16 so any `diffusers` install can load
> the model with no custom kernels — so it is the *same 3.21 GB* as FP16. The real
> compressed artifact is the **517 MB packed** file.

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
