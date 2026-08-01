---
license: apache-2.0
library_name: custom
pipeline_tag: image-to-image
tags:
  - super-resolution
  - video
  - webgpu
  - browser
  - product-demo
  - ui
---

# Clark Realtime UI Video Upscaler x2

A 16.6 KiB RGB x2 super-resolution model for product-demo screenshots and video. It is designed for UI edges, small text, buttons, panels, and code—not portraits or generative detail.

The browser artifact has 8,496 parameters and runs as three fused FP16 WebGPU residual blocks. BackdropKit uses it locally for eligible video previews and exports, with ordinary high-quality browser scaling as the automatic fallback when `shader-f16` WebGPU is unavailable.

## Before and after

### Moving Seedance 2.0 action test

<video controls loop playsinline muted width="100%" poster="examples/seedance-action/seedance-2-action-200px-wipe-poster.png">
  <source src="https://huggingface.co/clark-labs/clark-realtime-ui-video-upscaler-x2/resolve/main/examples/seedance-action/seedance-2-action-200px-wipe.mp4" type="video/mp4">
</video>

[Watch the aligned 200-pixel wipe](examples/seedance-action/seedance-2-action-200px-wipe.mp4) · [Inspect the source/bicubic/Clark control](examples/seedance-action/seedance-2-action-200px-triptych.mp4) · [Download the real 200×112 input](examples/seedance-action/seedance-2-action-real-input-200px-crf36.mp4) · [Watch the original Seedance master](examples/seedance-action/seedance-2-action-original-720p.mp4)

This five-second, 24 FPS test starts with a public 1280×720 Seedance 2.0 action
generation: a running puppy, tracking camera, fur, grass, flowers, and a moving
butterfly. A deterministic product-demo HUD is added before downsampling so the
test also contains the small labels, cards, and buttons this model targets. The
hero intentionally uses a heavily compressed, genuinely tiny 200×112 input.
The aligned wipe compares the same moving pixels: the decoded source is
nearest-neighbor magnified 4× for inspection and Clark's honest 400×224 x2
output is magnified 2×. Display magnification is not counted as model output.

Against a separately derived 400×224 held-out target, across all 120 frames:

| Measure | Bicubic x2 | Clark x2 | Gain |
|---|---:|---:|---:|
| RGB PSNR | 23.383 dB | 23.338 dB | -0.045 dB |
| Edge PSNR | 12.296 dB | 12.538 dB | +0.241 dB |
| Temporal-delta PSNR | 24.784 dB | 24.666 dB | -0.119 dB |

The moving wipe makes the low-resolution input-to-output change legible because
the comparison is spatially aligned. The three-way control remains the
scientific comparison: at this deliberately extreme size and compression,
Clark improves edge recovery over bicubic but does not beat bicubic on
whole-frame or temporal-delta fidelity. The source, separate
bicubic and Clark outputs, held-out reference, exact prompt, seed, hashes,
and metric definitions are all included under
`examples/seedance-action/` and `technical/seedance-action-video-evaluation.json`.

The less severe 356×200 to 712×400 test improves RGB PSNR by 0.115 dB and edge
PSNR by 1.222 dB versus bicubic. The 640×360 to 1280×720 test is included as
[a native triptych](examples/seedance-action/seedance-2-action-native-triptych.mp4).
It measures +0.703 dB RGB PSNR, +2.673 dB edge PSNR, and -0.922 dB
temporal-delta PSNR versus bicubic.

### Product-demo stills

![Actual low-resolution input, bicubic x2, and Clark realtime x2](examples/product-demo-before-after.png)

The left column is the actual low-resolution source before either scaler is applied. The middle and right columns both start from that exact source. A held-out high-resolution frame is used only for PSNR measurement; it is not shown or mislabeled as the input.

On three fixed frames from real Clark product-demo videos:

| Measure | Bicubic x2 | Clark x2 | Gain |
|---|---:|---:|---:|
| RGB PSNR | 32.307 dB | 33.570 dB | +1.263 dB |
| Edge PSNR | 30.476 dB | 31.666 dB | +1.191 dB |

Frame timestamps and deterministic crop coordinates are in `technical/example-evaluation.json`. Crops are selected by reference edge density only, never by model score.

## Model

- Input: even-sized RGB image or video frame
- Output: RGB at 2x width and height
- Working grid: 2x RGB pixel-unshuffle
- Body: 12 channels, 3 residual blocks, 2 convolutions per block
- Output: 4x pixel-shuffle from the half-resolution working grid
- Browser weights: FP16, 16,992 bytes
- Browser SHA-256: `c6b4f22d8c38270225b809e97d233eb68d7a48eef17595ac17a60b4540c72dce`
- Safetensors: 34,712 bytes

Training used 206,232 product, UI, and multilingual text images, with 268 held-out images. The RGB refinement ran for 5,000 steps from an exactly equivalent luma initialization.

## Measured browser performance

On an Apple M5 in Chrome 150 WebGPU, 1920x1080 to 3840x2160:

- Neural render: 8.9 ms median, 10.6 ms p95 over 120 measured frames
- Neural render plus H.264 enqueue: 9.4 ms median, 11.0 ms p95
- H.264: 120/120 output chunks, maximum queue 2, no encoder error

These measurements show compute headroom; they are not a promise of 60 displayed or exported frames per second on every browser. Decode cadence, browser scheduling, GPU, thermals, encoder, and export composition remain platform-dependent. The full receipt is in `technical/runtime-receipt.json`.

## Browser use

Host `browser/model.bin`, `browser/realtimeVideoUpscaler.js`, and `browser/runtimeCapabilities.js` together. Then pass the hosted model URL explicitly:

```js
import {
  createRealtimeVideoUpscaler,
  REALTIME_VIDEO_MODEL,
} from "./browser/realtimeVideoUpscaler.js";

const runtime = await createRealtimeVideoUpscaler({
  width: video.videoWidth,
  height: video.videoHeight,
  model: {
    ...REALTIME_VIDEO_MODEL,
    url: "./browser/model.bin",
  },
});

const enhancedCanvas = await runtime?.enhance(video);
// Draw enhancedCanvas into your composition or encoder.
runtime?.destroy();
```

The fast neural path requires WebGPU plus `shader-f16`. Applications should preserve a standard canvas/video scaling path when those capabilities are absent.

## Package contents

- `browser/model.bin`: exact FP16 shader payload
- `browser/model.json`: offsets, sizes, hash, and architecture ledger
- `browser/realtimeVideoUpscaler.js`: fused browser runtime
- `model.safetensors`: framework-friendly weights
- `realtime_student.py`: PyTorch architecture
- `examples/`: real product-demo comparisons
- `technical/`: evaluation, runtime, and package receipts

Built by Clark Labs for [BackdropKit](https://backdropkit.com/).
