# Swin2SR real-time browser kernel study

Date: 2026-07-31

## Objective

Find the smallest set of runtime, kernel, graph, and temporal changes that can
move the released Clark mixed-precision ternary Swin2SR model from interactive
crop enhancement toward large-image throughput and product-demo video. Every
claim must separate measured performance from projection, and every promoted
path must retain exact artifact identity, image quality, portable fallbacks,
and bounded memory.

## Initial production finding

BackdropKit still shipped the previous 683,547-byte all-ternary portable ONNX
artifact. The product bundle has now been updated to the 700,584-byte Clark
Ternary v2 artifact with SHA-256
`b6308e886a0ef07a2b433e2aa83f6a2234d2dbcdbf1427160abedb4e3629aa54`.
The saved-project identifier remains `ternary-g2` for compatibility.
The production URL is versioned as `model-v2.bin`; the prior `model.bin` stays
in the bundle for one-release cached-client compatibility.

The local BackdropKit product boundary was exercised after the update: the
served `model-v2.bin` was exactly 700,584 bytes with the expected SHA-256, the
Focus zoom control triggered the worker, WebGPU inference completed, and the
UI reported `Sharp · GPU`. All 57 product tests and the production Vite build
passed.

## Baseline hypotheses

1. Portable ONNX is dominated by graph/dispatch and activation traffic, not the
   701 KB network download. The unoptimized graph contains 22,103 nodes,
   including 384 MatMul, 431 Transpose, and thousands of dynamic-shape nodes.
2. The packed WASM reference is not a production-speed kernel. It decodes
   tensor metadata and weights repeatedly, recomputes input-independent
   relative bias and shifted masks, allocates new vectors at most operator
   boundaries, and uses scalar convolution loops.
3. The existing `ternary.wgsl` is an unconnected one-output-per-invocation
   proof. It performs base-3 integer division and modulus inside the inner loop
   and does not tile input or weight reuse through workgroup memory.
4. Full-frame 1080p-to-4K Swin2SR is unlikely to become real time through
   packing alone. A real-time mode will probably require both fused kernels and
   a distilled/reparameterized spatial model or temporal dirty-region reuse.

## Frozen measurement rules

- Report cold model fetch/decompression/session creation separately from warm
  inference.
- Use median and p95 over at least five warm runs for interactive kernels and at
  least 30 frames for a video claim.
- Measure actual input pixels per second; label 1080p FPS obtained by linear
  pixel scaling as an estimate until a real 1920×1080 frame is executed.
- Measure WebGPU and forced WASM independently. Do not use WebGPU availability
  as a feature gate.
- For video, report decode, upload/import, inference, composite, encode, and
  queue-wait time separately, plus dropped frames and peak resident buffers.
- Quality gates include full-image RGB PSNR, edge PSNR, UI OCR accuracy,
  temporal warping error, flicker on static regions, and seam error across tile
  boundaries.
- A kernel advances only if its output matches the release state within a
  preregistered maximum error and its end-to-end wall-clock improvement remains
  after preprocessing and readback are included.

## M5 Chrome 150 baseline

Machine/browser receipt: Apple M5, Chrome 150.0.0.0, macOS. The adapter exposed
`shader-f16`, subgroups, timestamp queries, 1,024 maximum compute invocations
per workgroup, and 32,768 bytes of workgroup storage.

The released 700,584-byte gzip expands to 4,869,958 bytes in 3.6-4.1 ms. A
dynamic WebGPU session took 3,093.2 ms to create. Fixed 192x192 dimensions
reduced session creation to 1,075.8 ms, but did not improve warm inference:

| Input | WebGPU median | p95 | input MP/s |
| --- | ---: | ---: | ---: |
| 32x32 | 23.1 ms | 24.6 ms | 0.0443 |
| 64x64 | 24.9 ms | 25.8 ms | 0.1645 |
| 96x96 | 32.4 ms | 32.7 ms | 0.2844 |
| 128x128 | 51.5 ms | 52.0 ms | 0.3181 |
| 192x192 | 111.6 ms | 113.4 ms | 0.3303 |

The 32x32 and 64x64 cases cost nearly the same, proving a large fixed graph and
dispatch component. Throughput plateaus near 0.33 input MP/s. Linear scaling
from 192x192 projects 1080p-to-4K to 6.28 seconds/frame or 0.159 FPS. This is a
projection, not a full-frame measurement.

Forced four-thread WASM sustained only 0.0106-0.0111 input MP/s: 97.0 ms at
32x32, 376.9 ms at 64x64, and 830.1 ms at 96x96. Reusing the current ONNX graph
on CPU is therefore a compatibility path, not a real-time path.

Graph capture rejected ordinary CPU-backed tensors with `External buffer must
be provided for input/output index 0`. This confirms that capture must be
implemented together with fixed buckets and GPU I/O binding; toggling the
session option alone is invalid.

## Compute and memory bound

The released architecture has 24 Swin blocks, 60 channels, 120-channel MLPs,
six 10-dimensional heads, and 8x8 windows. A conservative operation count is
about 1,045,620 MAC per input pixel: 875,520 in transformer projections and
window attention, about 163,620 in convolutions and x2 reconstruction, and the
remainder in small projections. That is about 38.5 GMAC for a 192x192 crop and
2.17 TMAC for a 1080p frame. Thirty 1080p input frames per second would require
about 65 TMAC/s before preprocessing, decode, composite, or encode.

A full 1080p 60-channel FP32 feature map is 498 MB; a 180-channel QKV tensor is
1.49 GB; explicit per-window attention scores are about 3.19 GB. Browser memory
therefore requires tiling and fused attention even if arithmetic were free.

## Packed projection kernel experiment

The existing base-3 shader was compared with a load-time transcode to a 2-bit
GPU layout (16 ternary weights per u32), 8x8 output tiling, 16-wide K tiles,
workgroup input/weight reuse, and one scale multiplication per 60-value group.
The download can remain base-3; only the resident GPU copy expands by 33.3%.

On 4,096 rows, the packed tiled kernel was 5.36x faster than the naive base-3
shader for 60x60, 5.56x for 60x120, and 6.00x for 120x60. Maximum output error
was 1.64e-7. At 36,864 rows, 60x60 took 0.225 ms versus 1.300 ms.

The necessary control falsified the stronger claim: an identically tiled dense
FP32 kernel took 0.200 ms at 36,864 rows, 11% faster than the packed 2-bit
kernel. The 2-bit path is a decisive win over runtime base-3 division, but on
this GPU and these tiny matrices it is a memory/transport optimization, not a
faster arithmetic primitive than cached FP32. It should not be advertised as a
GPU speed breakthrough until fusion makes weight bandwidth material.

## Decoder-free control

The Ternary v2 state was materialized into the original decoder-free ONNX graph.
Its deterministic gzip is 1,692,084 bytes, 2.42x the production package. At
192x192 it created a fixed-shape session in 911.3 ms and ran in 107.8 ms median,
versus 1,075.8 ms and 111.6 ms for the 700,584-byte portable graph. Runtime
base-3 decoding is largely folded at session creation: removing it improves
warm inference only 3.4% and is not worth the 142% download increase.

## Execution design selected for the next prototype

1. Keep Ternary v2 as the download/default and transcode packed weights once.
2. Add fixed 64/96/128/192 buckets with GPU-buffer input/output, graph capture,
   and a bounded LRU of sessions. This targets cold/session and repeated-video
   dispatch overhead without changing model quality.
3. Keep decoded video frames and composition on GPU. Import each `VideoFrame`
   as an external texture, fuse crop/normalize/NCHW preparation into a shader,
   and write model output directly into a GPU buffer/texture. Avoid RGBA CPU
   readback except for a requested download.
4. Fuse each 8x8-window Swin block: partition/shift + QKV + L2 normalization;
   online 64-token attention without a global score tensor; projection +
   reverse shift + residual + layer norm; MLP up + GELU + down + residual.
5. Fuse stage convolution/projection and final convolution/pixel shuffle.
6. For product-demo video, compute 16x16 luma hashes and motion vectors, reuse
   unchanged enhanced tiles, warp pure scroll/pan frames, infer only new or
   dirty strips, and invalidate on scene cuts. Quality is measured with static
   region flicker, warped-edge error, OCR, and seam metrics.
7. In parallel, distill the released Swin teacher into a small
   reparameterizable CNN/SPAN-like student specialized for UI screenshots and
   AV1/H.264 screen recordings, then apply the proven mixed-precision ternary
   schedule. Full-frame real time is an architecture problem, not a packing
   problem.

## Promotion gates

- Still-image mode: real 4K output, bounded overlap tiles, no visible seams,
  peak GPU allocation below a declared device tier, and exact export parity.
- Video preview: 30-frame minimum, p50/p95 stage timings, no queue growth, and
  adaptive quality that never blocks playback.
- Video export: deterministic decode/infer/composite/encode timestamps, all
  `VideoFrame` objects closed promptly, and no dropped output timestamps.
- CPU fallback: use the fast deterministic shader/bicubic path for playback;
  run AI only on selected settled regions or during offline export. The current
  WASM Swin graph cannot be called "real time."

## 2026-08-01 — Kernel study outcome

The architecture study produced a separate 12-channel, three-block RGB
student rather than attempting realtime full Swin2SR. Its 16,992-byte FP16
payload measured 8.9 ms median and 10.6 ms p95 for actual 1080p-to-4K WebGPU
inference on Apple M5 Chrome 150. With H.264 enqueue it measured 9.4 ms median
and emitted all 120 chunks. This closes the M5 compute-kernel experiment, not
universal realtime playback.

The exact selected model, action-video quality results, temporal regression,
and rejected 14-channel follow-up are recorded in
`../realtime_1080p_to_4k_20260731/SCIENTIFIC_JOURNAL.md`. No process in this
kernel-study folder is active.
