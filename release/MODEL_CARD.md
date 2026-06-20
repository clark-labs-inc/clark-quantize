---
license: apache-2.0
base_model: Efficient-Large-Model/Sana_1600M_512px_diffusers
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - text-to-image
  - diffusion
  - quantization
  - ternary
  - sana
  - clark-compress
  - edge
---

# Clark Compress · Sana 1.6B · Ternary

**A tiny, fast text-to-image model. ~6.4× smaller than FP16 — and it still understands your prompt.**

Clark Compress shrinks the Sana 1.6B diffusion transformer to **ternary precision** while keeping images
**coherent and prompt-faithful** — including on prompts the model has never seen. It runs the standard
`diffusers` pipeline today, with packed builds for Apple Silicon and NVIDIA shipping alongside.

- ⚡ **~6.4× smaller** transformer
- 🧠 **Composition preserved** — no semantic collapse, even on held-out prompts
- 🍃 **Data-free** — built without any image dataset
- 📦 **Drop-in** — loads in stock `diffusers`
- 🪪 **Apache 2.0**

## Quick start

```python
import torch
from diffusers import SanaPipeline, SanaTransformer2DModel

transformer = SanaTransformer2DModel.from_pretrained(
    "clark-labs/clark-compress-sana-1.6b", subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_512px_diffusers", transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

img = pipe("a half-empty bottle of red wine", num_inference_steps=20,
           height=512, width=512, guidance_scale=4.5).images[0]
img.save("out.png")
```

## What it's good at

Coherent, recognizable images across everyday and creative prompts — products, scenes, animals,
instruments, motion. This is the **preview** ternary build; we're polishing fine detail and exact text,
and shipping the packed (Apple MLX / NVIDIA) variants for the full size-and-speed win.

## Variants

| repo | format | best for |
|---|---|---|
| `clark-compress-sana-1.6b` (this) | safetensors | runs anywhere via `diffusers` |
| `…-mlx` | packed | Apple Silicon |
| `…-gemlite` | packed | NVIDIA CUDA |

## Notes

Preview research artifact built on Sana 1.6B; inherits the base model's capabilities and biases.
Exact text rendering and the finest detail are still improving.

---

*Built with **Clark Compress** — making great image models small enough to run anywhere.*
