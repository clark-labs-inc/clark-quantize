# Clark Air — Sana 1.6B · GemLite INT2 (ternary)

A self-contained text-to-image pipeline: the Sana 1.6B transformer trunk runs **ternary
{-1, 0, +1}** weights on real **GemLite INT2** CUDA kernels, bundled with the text encoder,
VAE, tokenizer and scheduler so it loads in one call.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "clark-labs/clark-air-sana-1.6b-gemlite-2bit", trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")

img = pipe("a corgi astronaut floating in space, photorealistic, 8k",
           num_inference_steps=20, guidance_scale=4.5).images[0]
img.save("out.png")
```

`trust_remote_code=True` is required: the transformer is a custom component
(`ClarkAirSanaGemliteTransformer`) that builds the GemLite kernels on load.

## Requirements

- **NVIDIA CUDA GPU** — GemLite INT2 kernels are CUDA-only.
- `gemlite`, `diffusers >= 0.38`, `transformers`, `torch`, `safetensors`.

## What's inside

| Component | Format | Size |
|---|---|---|
| `transformer/` | GemLite INT2, ternary trunk + bf16 islands | **519 MB** |
| `text_encoder/` | Gemma-2, bf16 | 4.9 GB |
| `vae/` | DC-AE (AutoencoderDC), bf16 | 596 MB |
| `tokenizer/` + `scheduler/` | — | 33 MB |

The transformer's attention + GLU-FFN trunk is ternary, packed two bits per weight and
re-quantized at each layer's largest power-of-two GemLite group (g64 attention / g32 FFN);
the small "island" layers (embeddings, norms, modulation, depthwise conv) stay bf16. The
transformer is **519 MB** vs ~9 GB at fp16. The text encoder and VAE are the **original
Sana 1.6B** components, weight-for-weight unchanged (bf16) — only the transformer is quantized.

## Standalone packed file (alternative)

The repo also carries `clark_air_sana_gemlite_2bit.safetensors` + `load_clark_air_gemlite.py` —
a single-file packed transformer with a minimal loader, for when you already have a Sana
pipeline and just want to swap in the gemlite trunk. The `from_pretrained` route above is the
self-contained option.
