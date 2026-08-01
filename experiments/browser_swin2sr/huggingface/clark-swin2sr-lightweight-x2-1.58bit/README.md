---
license: apache-2.0
base_model: caidas/swin2SR-lightweight-x2-64
library_name: transformers
pipeline_tag: image-to-image
tags:
  - image-super-resolution
  - swin2sr
  - transformers
  - transformers.js
  - onnx
  - onnxruntime-web
  - webgpu
  - wasm
  - ternary
  - mixed-precision
---

# Clark Swin2SR ×2 Ternary — a 380 KB mixed-ternary browser AI upscaler

This is a quality-oriented ternary Swin2SR model for 2× image super-resolution.
It runs entirely on the user's device: no server inference, no image upload, and
no watermark.

The model is deliberately not described as “exactly 1.58-bit.” It follows the
same practical design used by quality-oriented low-bit image models:

- 198 matrix and convolution tensors are ternary `{−1, 0, +1}`;
- four sensitive residual-group convolutions—129,600 values—use signed int7 with per-output scales;
- the first and final image-I/O convolutions—8,100 weight values—remain FP16;
- 83.55% of all parameters are ternary and 96.43% are low-bit weights;
- the complete lossless native browser package is **380,107 bytes**, or
  **3.025 bits per model parameter** including its manifest and precision
  islands.

The public repository name is retained for compatibility with existing links,
but the storage accounting below is the authoritative description.

## See the difference

Each comparison starts from the same low-resolution Set5 image. “Before” is
ordinary bicubic enlargement. “After” is this model. The cards use a 100%
detail crop and add no post-model sharpening.

![Bicubic versus Clark Ternary on the Set5 butterfly image](examples/butterfly-before-after.png)

![Bicubic versus Clark Ternary on the Set5 baby image](examples/baby-before-after.png)

![Bicubic versus Clark Ternary on the Set5 woman image](examples/woman-before-after.png)

### [▶ Try the live demo →](https://huggingface.co/spaces/clark-labs/clark-swin2sr-lightweight-x2-1.58bit)

The Space uses WebGPU when available and falls back to WebAssembly. For the
same private on-device workflow inside a product-demo editor, try
[BackdropKit](https://backdropkit.com).

## Quality

### Set5 ×2

Full-image RGB PSNR across all five Set5 images:

| Method | Mean PSNR | Versus bicubic |
|---|---:|---:|
| Bicubic interpolation | 31.79 dB | — |
| Previous Clark Ternary Lite | 34.44 dB | +2.66 dB |
| **Clark Ternary, this release** | **35.30 dB** | **+3.52 dB** |

The complete per-image receipt is in
[`examples/set5-x2-results.json`](examples/set5-x2-results.json).

### Product and multilingual UI development set

On the locked 3,040-image development bundle—product imagery, real UI
captures, and synthetic UI text spanning 204 languages—the release scored:

| Metric | Result |
|---|---:|
| RGB PSNR versus ground truth | **34.31 dB** |
| Edge PSNR versus ground truth | **37.87 dB** |
| Gain over bicubic | **+6.21 dB** |

This is +0.21 dB over the previous Clark Ternary release on the same locked bundle,
with every tested source, split, script, and language group non-regressing.

## Honest storage accounting

| Layer or package view | Result |
|---|---:|
| Model parameters | 1,005,348 |
| Ternary weight values | 839,904 (83.55% of parameters) |
| Int7 retained-island weight values | 129,600 |
| Ternary tensors | 198 |
| Int7 tensors | 4 |
| Retained FP16 tensors | 2 image-I/O convolutions |
| Retained FP16 weight values | 8,100 |
| Packed mixed low-bit trunk, codes + scales | 2.644 bits/quantized weight |
| Lossless native package | **380,107 bytes; 3.025 bits/model parameter** |
| Portable ONNX browser download | **808,002 bytes** |

The distinction matters. `log₂(3) ≈ 1.585` describes the information in one
ternary code; it does not include group scales, raw parameters, FP16 islands,
alignment, or a runtime manifest. This card reports each layer separately.

## Runtime verification

The exact release state passed:

- exact dense-state to portable-ONNX numerical conformance;
- Chromium WebGPU portable ONNX conformance;
- Chromium WASM portable ONNX conformance;
- the independent packed runtime on WASM SIMD and scalar builds;
- native Rust loader and kernel conformance.

At a 64×64 input (128×128 output) in the release browser harness, warm ONNX
inference measured 30.4 ms on WebGPU and 377.3 ms on WASM. Performance varies
with browser, hardware, image size, tiling, and thermal state.

## Files

- `model.safetensors` — standard Transformers/PyTorch weights.
- `onnx/model.onnx.gz` — portable browser model; WebGPU with WASM fallback.
- `onnx/model.onnx` — the same portable graph without transport compression.
- `native/swin2sr-ternary-bonsai-x2.bdk.gz` — the lossless 380 KB packed
  Clark browser-runtime package.
- `technical/quantization.json` — exact artifact hashes, storage, and quality
  receipts.

## Quick start

### Python with Transformers

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

model_id = "clark-labs/clark-swin2sr-lightweight-x2-1.58bit"
processor = AutoImageProcessor.from_pretrained(model_id)
model = Swin2SRForImageSuperResolution.from_pretrained(model_id).eval()

image = Image.open("input.png").convert("RGB")
inputs = processor(image, return_tensors="pt")

with torch.inference_mode():
    output = model(**inputs).reconstruction

output = output[:, :, : image.height * 2, : image.width * 2]
```

### Browser with ONNX Runtime Web

```html
<script type="module">
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

const response = await fetch(
  "https://huggingface.co/clark-labs/clark-swin2sr-lightweight-x2-1.58bit/resolve/main/onnx/model.onnx.gz"
);
if (!response.ok) throw new Error("Model download failed");

const modelBytes = await new Response(
  response.body.pipeThrough(new DecompressionStream("gzip"))
).arrayBuffer();

const providers = "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: providers,
});

// `pixels` is planar RGB Float32 data in [0, 1]. Pad dimensions to multiples
// of eight, run `pixel_values`, then crop `reconstruction` to 2× the original.
const input = new ort.Tensor(
  "float32",
  pixels,
  [1, 3, paddedHeight, paddedWidth]
);
const { reconstruction } = await session.run({ pixel_values: input });
</script>
```

## What it is good at

- Zoomed product screenshots and interface details
- Small labels, icons, controls, and clean UI edges
- Product thumbnails, illustrations, and natural images
- Private, offline, or zero-server-cost image pipelines

## Limitations

- It performs fixed 2× super-resolution.
- It is restorative, not generative: it cannot recover information that never
  existed in a severely blurred or compressed source.
- Benchmark conventions differ. Scores here are full-image RGB PSNR, not
  luminance-only scores with cropped borders.
- The compact packed runtime is Clark-specific; use the ONNX file for standard
  cross-browser integration.

## Lineage and credits

- Architecture: [Swin2SR](https://github.com/mv-lab/swin2sr), Conde et al.
- Base weights: [`caidas/swin2SR-lightweight-x2-64`](https://huggingface.co/caidas/swin2SR-lightweight-x2-64)
- Set5 data: [`eugenesiow/Set5`](https://huggingface.co/datasets/eugenesiow/Set5)
- Quantization, training, browser packaging: [Clark Labs](https://huggingface.co/clark-labs)

## Citation

```bibtex
@misc{clark2026swin2srternary,
  title  = {Clark Swin2SR Lightweight x2 Ternary:
            Heterogeneous-Precision Super-Resolution for the Browser},
  author = {Clark Labs},
  year   = {2026},
  url    = {https://huggingface.co/clark-labs/clark-swin2sr-lightweight-x2-1.58bit}
}
```

Apache 2.0—use it anywhere, including commercially.
