# Clark Air — Sana 1.6B · GemLite INT2 + NF4 text encoder (slim)

The small build of [`clark-air-sana-1.6b-gemlite-2bit`](https://huggingface.co/clark-labs/clark-air-sana-1.6b-gemlite-2bit):
**3.2 GB** instead of 6.0 GB. The Sana 1.6B transformer trunk runs ternary {-1, 0, +1} on real
**GemLite INT2** kernels, and the Gemma-2 text encoder is **bitsandbytes NF4 (4-bit)**. Renders
are visually identical to the bf16 build.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "clark-labs/clark-air-sana-1.6b-gemlite-2bit-slim", trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")

img = pipe("a corgi astronaut floating in space, photorealistic, 8k",
           num_inference_steps=20, guidance_scale=4.5).images[0]
img.save("out.png")
```

`trust_remote_code=True` is required (custom GemLite transformer component). The 4-bit text
encoder loads automatically from the embedded `quantization_config`.

## Requirements

- **NVIDIA CUDA GPU** — both GemLite INT2 and bitsandbytes NF4 are CUDA-only.
- `gemlite`, `bitsandbytes`, `diffusers >= 0.38`, `transformers`, `accelerate`, `torch`, `safetensors`.

## What's inside

| Component | Format | Size |
|---|---|---|
| `transformer/` | GemLite INT2, ternary trunk + bf16 islands | 519 MB |
| `text_encoder/` | Gemma-2, **bitsandbytes NF4** | 2.1 GB |
| `vae/` | DC-AE (AutoencoderDC), bf16 | 596 MB |
| `tokenizer/` + `scheduler/` | — | 33 MB |

The text encoder + VAE are the **original Sana 1.6B** weights — only the transformer is
quantized (ternary) and the text encoder is NF4-compressed. The NF4 text encoder's 2.1 GB is
dominated by the 256k-token embedding, which stays bf16 (quantizing it hurts caption quality).
For the bf16-text-encoder build (no `bitsandbytes` dependency), use
[`clark-air-sana-1.6b-gemlite-2bit`](https://huggingface.co/clark-labs/clark-air-sana-1.6b-gemlite-2bit).
