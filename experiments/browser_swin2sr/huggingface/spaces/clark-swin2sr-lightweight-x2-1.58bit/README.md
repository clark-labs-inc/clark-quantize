---
title: Clark Ternary Image Upscaler
emoji: 🔎
colorFrom: purple
colorTo: green
sdk: static
app_file: index.html
fullWidth: true
header: mini
pinned: false
license: apache-2.0
models:
  - clark-labs/clark-swin2sr-lightweight-x2-1.58bit
short_description: Private ternary image upscaling in your browser.
---

# Clark Ternary Image Upscaler

This Space runs the Clark Swin2SR ×2 model directly in your browser. Images
never upload to a server.

It uses WebGPU when the browser and device expose a working adapter, then falls
back automatically to WebAssembly. Bicubic appears immediately while the AI
result is drawn tile by tile with live progress and an estimated finish time.
Model setup and tile inference run in a dedicated browser worker, so uploads,
progress, and cancellation remain responsive even on the CPU fallback.

The model was built for product screenshots, interface details, thumbnails,
illustrations, and natural images. Downloads are full-resolution PNGs with no
watermark.

## What the numbers mean

- The portable compressed ONNX model is **808 KB**; the native mixed-low-bit bundle is **380 KB**.
- Every output is **2× wider and 2× taller**.
- Across all five Set5 test images, it scored **3.52 dB higher** than ordinary
  bicubic enlargement.
- There is no shared server queue: inference uses the visitor's own GPU or CPU.

See the full model card, visual comparisons, limitations, and downloadable
files at
[clark-labs/clark-swin2sr-lightweight-x2-1.58bit](https://huggingface.co/clark-labs/clark-swin2sr-lightweight-x2-1.58bit).

Built by [Clark Labs](https://huggingface.co/clark-labs) for
[BackdropKit](https://backdropkit.com).
