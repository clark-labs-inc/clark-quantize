---
license: apache-2.0
base_model: Efficient-Large-Model/Sana_1600M_512px_diffusers
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - text-to-image
  - diffusion
  - quantization
  - data-free
---

# Clark-Compress Sana-1.6B Mixed

Data-free, training-free post-training quantization of the Sana 1.6B diffusion
transformer.

| Variant | Size | Compression |
|---|---|---|
| FP16 (base transformer) | 3.21 GB | 1× |
| **Clark-Compress Mixed** | **0.72 GB** | **4.4× (77%)** |

Attention/projection linears use self-calibrated GPTQ with shared incoherence
rotation and true ternary packing; Sana's GLU-conv FFN uses data-free 4-bit
quantization because scalar ternary and scalar entropy-constrained methods fail on
this architecture. **No real images, prompts, or fine-tuning data are used for
quantizer calibration** — the calibration statistics come entirely from the model's
own unconditional prior (Gaussian noise + the null prompt).

## Usage

```python
from load_clark_compress import load
pipe = load(".")                       # this repo dir (.safetensors + manifest.json)
pipe.enable_model_cpu_offload()
pipe("a bowl of ramen, steam rising, photoreal").images[0].save("out.png")
```

The artifact contains the quantized transformer only; the standard Sana text
encoder and VAE are pulled automatically from the base repo
(`Efficient-Large-Model/Sana_1600M_512px_diffusers`).

License: Apache-2.0.
