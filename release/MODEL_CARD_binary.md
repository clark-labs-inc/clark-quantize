---
license: apache-2.0
base_model: Efficient-Large-Model/Sana_1600M_512px_diffusers
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - text-to-image
  - diffusion
  - quantization
  - binary
  - sana
  - clark-compress
  - edge
---

# Clark Compress · Sana 1.6B · Binary (Preview)

**The aggressive tier — ~8× smaller. 1-bit weights for maximum compression.**

The binary build pushes Sana 1.6B to **1-bit** `{-1, +1}` weights for the smallest possible footprint.
It's strong on everyday and product-style prompts; for complex, unusual compositions we recommend the
[ternary build](https://huggingface.co/clark-labs/clark-compress-sana-1.6b), which trades a little size
for more headroom.

- ⚡ **~8× smaller** transformer
- 🪶 **1-bit** matrix weights, FP16 scales + protected islands
- 🍃 **Data-free** — built without any image dataset
- 📦 **Drop-in** `diffusers`
- 🪪 **Apache 2.0**

## Quick start

```python
import torch
from diffusers import SanaPipeline, SanaTransformer2DModel

transformer = SanaTransformer2DModel.from_pretrained(
    "clark-labs/clark-compress-sana-1.6b-binary-unpacked", subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_512px_diffusers", transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

img = pipe("a half-empty bottle of red wine", num_inference_steps=20,
           height=512, width=512, guidance_scale=4.5).images[0]
img.save("out.png")
```

## Which build should I use?

| build | size | best for |
|---|---|---|
| [ternary](https://huggingface.co/clark-labs/clark-compress-sana-1.6b) | ~6.4× | **recommended** — best quality, handles complex scenes |
| binary (this) | ~8× | smallest footprint, everyday & product prompts |

## Notes

Preview research artifact on Sana 1.6B; inherits the base model's capabilities and biases. At 1-bit,
unusual compositions and exact text are softer than the ternary build. Packed (Apple MLX / NVIDIA)
variants ship alongside.

---

*Built with **Clark Compress** — making great image models small enough to run anywhere.*
