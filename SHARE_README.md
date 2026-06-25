# Clark Air · Sana 1.6B

**A Sana 1.6B text-to-image transformer compressed to ternary (~1.85 bits/weight) —
8.6× smaller than FP16, near-FP16 quality, data-free.**

![FP16 vs Clark Air, Sana 1.6B](demo.png)

*FP16 (top) vs **Clark Air** (bottom), Sana 1.6B — identical prompts and seed.*

## What this is

Clark Air quantizes Sana's diffusion-transformer weights to **ternary `{-1, 0, +1}`**
with group-wise scales. A small high-precision tail (~5% of parameters — the
conditioning and projection layers) is kept at higher precision. Training is
**quantization-aware** and **data-free**.

## Footprint (measured)

| Artifact | Size | vs FP16 | What it is |
|---|---|---|---|
| FP16 transformer | 3.21 GB | 1× (100%) | reference |
| **Clark Air — packed** | **374 MB** | **8.6× (≈12%)** | packed ternary `safetensors` |
| Clark Air — unpacked | 3.21 GB | compatibility | dequantized bf16 `diffusers` transformer |

- **Measured ~1.85 bits/weight → 8.6× smaller** (374 MB packed ÷ 3.21 GB FP16).

> ℹ️ The **unpacked** safetensors is a compatibility format, not the compressed size —
> it dequantizes to bf16 so any `diffusers` install loads it with no custom code, so it
> is the *same 3.21 GB* as FP16. The real compressed artifact is the **374 MB packed**
> `safetensors`.

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
- **Quantization:** ternary weights with group-wise scales; small high-precision tail
- **Quality:** evaluated on held-out prompts at 512px / 20 steps, same seed as FP16.

## License

Apache-2.0 © Clark Labs, Inc.
