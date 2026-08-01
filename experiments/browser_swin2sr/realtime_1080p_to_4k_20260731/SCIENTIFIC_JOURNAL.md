# Realtime 1080p/2K to 4K browser video super-resolution

Date: 2026-07-31

## Frozen target

- Input: 1920x1080 decoded product-demo video.
- Output: 3840x2160, preserving source frame timestamps.
- Preview gates: 60 FPS stretch target (16.67 ms total), 30 FPS hard gate
  (33.33 ms total), no unbounded decode or inference queue.
- Export gate: at least source FPS with deterministic timestamps; slower devices
  may use bounded offline export but must never stall preview playback.
- Browser-local only. WebGPU is an acceleration tier, never a feature gate.
- Report decode, GPU import, upscale/inference, composition, encode, queue wait,
  dropped frames, and peak resident allocations separately.

## Why released Swin2SR is only the teacher

The measured Clark Ternary v2 Swin2SR path is about 2.17 TMAC for one 1080p
input frame and projects to 6.28 seconds per frame on M5 Chrome. It is retained
for still images, settled focus regions, reference targets, and distillation.
No kernel-only change can close the roughly 188x gap to 30 FPS while also
leaving time for decode, composition, and encode.

## Student topology hypothesis

The first student operates primarily on luma:

1. Import the decoded frame as a GPU external texture.
2. Convert/split Y and chroma while pixel-unshuffling luma by 2. A 1920x1080
   luma plane becomes four channels on a 960x540 working grid without throwing
   away input samples.
3. Project 4 channels to a device-selected feature width.
4. Run three reparameterizable residual blocks, two 3x3 convolutions per block.
5. Project 16 channels to 16 output subpixels and pixel-shuffle by 4 to a 4K
   luma plane.
6. Upscale chroma with a fused edge-aware deterministic path and combine it
   with the reconstructed luma. Add the deterministic 4K baseline as a residual.

The initial 16-channel topology was approximately 8.66 GMAC/frame:

- input projection: 0.30 GMAC;
- six 16x16 3x3 convolutions: 7.17 GMAC;
- output projection/pixel-shuffle preparation: 1.19 GMAC.

The channel count is not frozen until real full-frame shader measurements show
the available budget. Training may use extra branches, attention, and larger
teachers only when they can be algebraically reparameterized out of inference.

## Scientific references translated into this lane

- AIM 2024 imposed a 250-GMAC ceiling, but successful 540p-to-4K entries were
  far smaller: BVI-RTVSR used 63K parameters and 9.595 GMAC/frame.
- BVI-RTVSR uses pixel unshuffle, three 24-channel residual blocks, PixelShuffle,
  luma-first reconstruction, bicubic chroma, and multi-teacher distillation.
- SuperBicubic++ showed that richer training-time branches can merge into a
simpler inference convolution, and that 32 to 64 channels roughly doubled
runtime in its implementation.
- SPAN shows a useful alternative for later quality experiments, but its six
  blocks with three convolutions each are above the first full-frame budget.

## Initial quality objective for product demos

Natural-video averages are insufficient. Evaluation must include UI text OCR,
one-pixel line survival, cursor edge quality, compression ringing, color-fringe
error, static-region flicker, scroll/pan consistency, and VMAF/PSNR/SSIM. The
student will be distilled from Clark Swin2SR plus ground truth, with extra loss
on edges, text masks, Laplacian pyramids, and temporal consistency.

## 2026-07-31 M5 WebGPU measurements

All timings below are actual Chrome WebGPU queue-completion measurements on the
M5 MacBook, not MAC-count projections. The source is a reproducible H.264
1920x1080 60 FPS product-demo fixture; the destination canvas and encoder are
3840x2160.

### Deterministic plumbing

- GPU external-texture import, five-tap limited sharpen, 4K composition, and
  queue completion: 2.1 ms median, 2.6 ms p95.
- Hardware H.264 High Profile 4K60 is available through WebCodecs. The isolated
  baseline produced 120/120 chunks with no encoder error.
- The released Swin teacher remains ineligible for full-frame video: its
  measured projection is 6.28 seconds per 1080p frame.

### Precision and channel frontier

Six separate 3x3 convolutions on the 960x540 working grid measured:

| Width | Precision | Convolution MAC | Median | p95 |
| ---: | --- | ---: | ---: | ---: |
| 12 | FP16 | 4.031 G | 8.9 ms | 9.2 ms |
| 14 | FP16 | 5.487 G | 12.0 ms | 12.5 ms |
| 16 | FP16 | 7.166 G | 15.5 ms | 16.0 ms |
| 16 | FP32 | 7.166 G | 21.6 ms | 21.9 ms |

FP16 is 1.39x faster than FP32 at 16 channels. This is a compute and activation
bandwidth problem, not a model-download problem.

### Fused two-convolution residual kernel

Each 8x8 output tile stages a 12x12 input halo and a 10x10 intermediate tile in
workgroup memory, then executes two convolutions and the residual addition in
one dispatch. Three blocks therefore replace six global-memory dispatches.

| Width | Workgroup memory | Core median | Core p95 | Separate-core median |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 5,856 B | 6.5 ms | 7.0 ms | 8.9 ms |
| 14 | 6,832 B | 8.8 ms | 9.4 ms | 12.0 ms |
| 20 | 9,760 B | 20.2 ms | 20.4 ms | not measured |
| 24 | 11,712 B | 31.4 ms | 31.5 ms | not measured |

The 12-channel fusion is 27.0% faster than six separate convolutions. The
24-channel core is rejected for realtime use because it consumes essentially
the whole 30 FPS frame budget before import, reconstruction, or encoding.

### Frozen runtime profiles

**Live 4K60:** 12 channels, three fused residual blocks, 8,016 parameters,
4.155 GMAC including the input/output projections, and 24.88 MB of ping-pong
feature storage. With actual learned weights and concurrent hardware H.264
4K60 encoding, a 300-frame run measured 7.7 ms median, 8.3 ms p95, and 9.0 ms
maximum. It produced 300/300 chunks, encoder queue depth never exceeded one,
and one of 316 presented callbacks arrived while the pipeline was busy. A
separate 600-frame stability run with the same architecture measured 7.7 ms
median, 8.3 ms p95, 8.5 ms maximum, and one busy callback.

**Quality 4K30:** 20 channels, three fused blocks, 11.405 GMAC, and 41.47 MB of
feature storage. The full neural plus 4K H.264 path measured 21.2 ms median and
21.8 ms p95 at a deliberate 30 FPS cadence, with 180/180 chunks, queue depth
one, and four busy callbacks among 398 observed 60 FPS source callbacks.

These are device-specific receipts, not cross-browser promises. Runtime
selection must benchmark a warm frame and keep a bounded latest-frame queue.

## First 12-channel distillation run

The exact runtime topology is implemented in `realtime_student.py`; its tests
pin the 8,016-parameter shape and exact WGSL storage order. The FP16 artifact is
only 16,032 bytes. Ternary packing would save roughly 14 KB while adding code
unpack/dequantization to an activation-dominated kernel, so FP16 is retained for
the live model unless a future ternary kernel proves an actual latency win.

Training uses 206,500 cached images:

- 15,500 RICO screenshots;
- 3,000 Aria UI screenshots;
- 84,000 WebSight product/web screenshots;
- 2,000 public-domain/CC0 natural images;
- 102,000 synthetic text images, 500 for each of 204 language-script groups.

Sampling is product-balanced: 35% RICO, 15% Aria UI, 20% WebSight, 20% FLORES
split equally across every language-script group, and 10% natural images. The
held-out gate uses 16 disjoint images from each image source and one disjoint
sample from every language-script group (268 samples total). Promotion metrics
weight those five source categories according to the same product mixture so
the 204 languages cannot dominate the aggregate.

The selected pilot uses AdamW at 1e-3 with 55% ground-truth Charbonnier, 25%
Swin-teacher Charbonnier, and 20% edge loss. The 1,000-step trajectory was:

| Step | Product-weighted GT PSNR | Gain vs baseline | Teacher PSNR | Gain vs baseline |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 27.1845 | - | 29.2255 | - |
| 250 | 27.9047 | +0.7202 | 30.2186 | +0.9931 |
| 500 | 28.4501 | +1.2656 | 31.0000 | +1.7746 |
| 750 | 28.6887 | +1.5042 | 31.3401 | +2.1146 |
| 1,000 | 28.7881 | +1.6037 | 31.5401 | +2.3146 |

The final learned FP16 payload SHA-256 is
`9865d62d03c220d172916d6c71d484271d4df65c2451c5773992874445e09b25`.

## Honest remaining gates

- The learned model beats its bilinear starting point, but has not yet been
  compared against bicubic, the full Swin teacher, or the released ternary
  still-image model on the sealed product promotion suite.
- Static-region flicker, scroll/pan consistency, VMAF, OCR retention, and
  one-pixel-line survival remain unmeasured on real held-out video.
- The browser harness proves H.264 chunks and bounded queues, not a complete
  audio-preserving MP4/WebM muxed export.
- The optimized kernel requires WebGPU plus `shader-f16`. WebGPU without FP16,
  WebGL, SIMD-WASM, scalar WASM, and CPU downgrade profiles remain to be built
  and measured.
- Border and sampling parity between PyTorch training and the fused WGSL path
  needs a tensor-level conformance fixture before the model can be promoted.

## 2026-08-01 RGB refinement and product promotion

The first luma residual artifact was numerically useful but produced visible
monochrome edge halos on some UI text. The refinement expanded only the input
and output projections to RGB while preserving the same 12-channel, three-block
fused body. It was initialized to reproduce the luma model exactly, then trained
for 5,000 steps at batch size 16 and learning rate 3e-4.

The RGB student has 8,496 parameters and a 16,992-byte FP16 browser payload. Its
SHA-256 is
`c6b4f22d8c38270225b809e97d233eb68d7a48eef17595ac17a60b4540c72dce`.

On the same 268 held-out images, product-weighted ground-truth PSNR improved
from 28.7881 dB at the exact luma-equivalent initialization to 29.2434 dB. On
three fixed real product-demo frames, evaluated from a hidden high-resolution
target, the correct comparison is actual low-resolution source to bicubic x2
and Clark x2:

| Measure | Bicubic x2 | RGB Clark x2 | Gain |
| --- | ---: | ---: | ---: |
| RGB PSNR | 32.3066 dB | 33.5696 dB | +1.2630 dB |
| Edge PSNR | 30.4756 dB | 31.6664 dB | +1.1907 dB |

The displayed source column is the untouched low-resolution frame enlarged
with nearest-neighbor solely for legibility. The high-resolution target is
never shown or mislabeled as the source; it is used only for the metrics. Crop
selection is deterministic reference-edge density and never model score.

The RGB path raises per-frame work from 4.155 to 4.404 GMAC. On the Apple M5,
an actual 1920x1080 to 3840x2160 WebGPU run over 120 measured frames produced:

- 8.9 ms median, 10.6 ms p95, and 12.3 ms maximum without encoding;
- 9.4 ms median, 11.0 ms p95, and 11.5 ms maximum with H.264 enqueue;
- 120/120 H.264 chunks, no encoder error, and maximum queue depth two.

The interactive browser did skip decoded-frame callbacks, so these are compute
and encoder-latency receipts, not a universal claim of sustained 60 displayed
FPS. The exact measurements are in `runtime-receipt-rgb-20260801.json`.

The model is now the default local neural path for eligible BackdropKit video
preview, quick video export, Export Kit video, and GIF generation. Browsers
without WebGPU `shader-f16` retain the ordinary browser scaling path. A fresh,
unused WebGPU adapter is handed from capability probing to inference; this fixes
the prior consumed-adapter failure observed in Chrome.

The public package is `clark-labs/clark-realtime-ui-video-upscaler-x2` on
Hugging Face. Remaining scientific work is tensor-level PyTorch/WGSL parity,
temporal flicker and OCR evaluation, broader device measurements, and a truly
accelerated non-WebGPU neural kernel rather than the current visual fallback.

## 2026-08-01 moving Seedance action evaluation

A five-second public ByteDance Seedance 2.0 prediction was added as a moving
held-out reference. The 1280x720, 24 FPS clip contains a running puppy, a
tracking camera, fine fur and grass, moving flowers, and a butterfly. A
deterministic BackdropKit-style HUD was composited before degradation to add
small labels, progress lines, cards, and a button without changing the motion
source. The exact public prediction, prompt, seed, and source hash are recorded
in `technical/seedance-action-video-evaluation.json` inside the Hugging Face
package.

The reference was Lanczos-downsampled and encoded to a real 640x360 input.
Bicubic and the RGB Clark student received the same decoded input. Across all
120 synchronized frames:

| Measure | Bicubic x2 | RGB Clark x2 | Gain |
| --- | ---: | ---: | ---: |
| RGB PSNR | 32.4819 dB | 33.1853 dB | +0.7034 dB |
| Edge PSNR | 17.9861 dB | 20.6586 dB | +2.6725 dB |
| Temporal-delta PSNR | 38.5197 dB | 37.5978 dB | -0.9220 dB |

The action test strengthens the evidence that the 16.6 KiB model recovers
per-frame structure and moving edges better than bicubic. It also closes one
previously unmeasured gate with an unfavorable result: frame-independent
sharpening increases temporal-delta error. The next training round should add a
paired-frame temporal-delta loss and motion-stratified clips, with a hard gate
that preserves the current spatial and edge gains.

Native, web-proxy, and detail-crop comparison videos are in
`examples/seedance-action/`. The real low-resolution input, separate bicubic and
Clark outputs, held-out reference, and a labeled four-way still are included so
the demo cannot mislabel the reference as the input.

### Compressed 200p visibility stress test

The first 360p triptych was too visually subtle at normal model-card width. A
second, deliberately severe demo now derives a 712x400 held-out target, encodes
a real 356x200 H.264 CRF 32 input, and gives that same decoded input to bicubic
and Clark. A large two-panel moving crop makes the input-to-output change
legible; a separate three-way crop preserves bicubic as the scientific control.

Across 120 frames, Clark improves RGB PSNR by 0.1152 dB and edge PSNR by 1.2218
dB over bicubic. Temporal-delta PSNR remains 0.1467 dB worse. This is a useful
demo correction, not a scientific breakthrough: it makes the existing model's
effect visible while reinforcing that the next model must target compression
restoration and paired-frame temporal fidelity directly.

### True 200-pixel-wide aligned wipe

The 356x200 demo was 200 pixels tall, not 200 pixels wide, and its side-by-side
layout forced the eye to compare different spatial locations. The replacement
hero uses a real 200x112 H.264 CRF 36 input, an honest 400x224 x2 output, and a
moving aligned wipe. For presentation only, the source is nearest-neighbor
magnified 4x and the x2 output 2x; that magnification is not attributed to the
model. A separate source/bicubic/Clark triptych remains the scientific control.

Across 120 frames, Clark is 0.0453 dB behind bicubic on RGB PSNR, 0.2414 dB
ahead on edge PSNR, and 0.1186 dB behind on temporal-delta PSNR. This harsher
fixture is therefore valuable both as a visible input-to-output demo and as a
failure-revealing promotion gate: the current student is not yet a general
compression-restoration model at 200-pixel-wide inputs.

## 2026-08-01 — 14-channel paired-frame arm stops at its sealed gate

A 14-channel, three-block student with 11,424 parameters was initialized to be
exactly equivalent to the released 12-channel model. Its two added channels
began as an isolated side branch and its output columns were zero, so step zero
could not regress. Training used deterministic translated adjacent pairs from
the 206,232-image rights-clean product/UI/multilingual corpus and combined
ground-truth, teacher, edge, temporal-delta, and static-consistency losses.

At step 250 it improved ground-truth PSNR by 0.003091 dB, teacher PSNR by
0.017232 dB, edge PSNR by 0.000375 dB, temporal-delta PSNR by 0.024076 dB, and
static-delta MAE by 0.0000153. At step 500, spatial metrics and temporal-delta
PSNR remained positive, but static-delta MAE regressed by 0.0000030. The sealed
step-500 rule required both temporal metrics to improve, so the trainer emitted
`sealed_gate_failed` and stopped. This arm is rejected as a release replacement;
the public 12-channel model remains the exact BackdropKit video default.
