# Current Swin2SR and realtime-video state

Updated: 2026-08-01

This page is the live boundary. Historical trajectories remain append-only in
`SCIENTIFIC_JOURNAL.md`; exact paths and immutable receipts are indexed in
`ARTIFACT_INDEX.md`.

## Execution state

- No Swin2SR image or video training process is active on this machine.
- G3 is the selected and deployed still-image ternary package.
- The 12-channel RGB student is the selected and deployed realtime-video
  package on eligible WebGPU `shader-f16` devices.
- The fixed-int7 scale arms, RSTB-bias arm, and 14-channel temporal arm are
  stopped and rejected by their sealed gates.
- LLM and Sana research remain archived reference lanes; they are not active
  BackdropKit objectives.

## Still-image selection: G3

The selected dense state is
`308b7fc0de513a4983563d519219c100ff8f13bb0a5143d5cd3e1d4204d2a612`.
It contains 1,005,348 parameters:

- 839,904 values in 198 ternary tensors, or 83.5436% of all parameters;
- 129,600 signed-int7 values in four RSTB residual-group convolutions, stored
  per output channel;
- 8,100 FP16 image-I/O weights in the first and final convolutions.

The packed native transport is 380,107 bytes, or 3.02468 lossless transport
bits per model parameter. The portable ONNX gzip is 808,002 bytes with
SHA-256
`a5e32c6b853f32e6c2fd39615ff27f7780e076fdac504dc3f2be1e822e3e085c`.
This is an honest Bonsai-style mixed-precision ternary model, not a claim that
every parameter or the complete payload is exactly 1.58 bits.

On the fixed 3,040-sample development bundle, G3 scores:

| Metric | G3 | Gain over previous Ternary Lite g2 |
| --- | ---: | ---: |
| Ground-truth PSNR | 34.307506 dB | +0.209351 dB |
| Edge PSNR | 37.873079 dB | +0.150167 dB |
| Teacher PSNR | 41.703679 dB | +0.721767 dB |
| Gain over bicubic | +6.205912 dB | — |

All tested development groups are non-regressing versus the previous ternary
release. G3 still trails the shipped q4 scientific reference on the headline
quality metrics, so it replaces the old ternary product artifact but does not
close the q4 non-inferiority claim.

The exact G3 graph passed Chromium WebGPU and forced WASM conformance on a
64-to-128 fixture. Warm WebGPU inference was 30.4 ms; forced WASM was
377.3 ms. These are correctness receipts, not full-resolution throughput
claims or cross-browser closure.

Public release identities:

- Hugging Face model commit:
  `b357304af178a6e31a5482958f1adf702f99be82`;
- Hugging Face Space worker/cancellation repair:
  `9896ce694616a8294fc566cd68f3c4a873a33d8c`;
- BackdropKit Cloudflare Worker version:
  `a559a467-dc53-4b91-82e5-10a2f628c40c`.

The public model, Space, and BackdropKit deployment all identify the same
808,002-byte portable artifact hash.

## Rejected still-image follow-ups

Three attempts to improve G3 while retaining its deployed codes were rejected:

- fixed int7 scales with mask 7 at learning rate 2e-4;
- fixed int7 scales with mask 0 at learning rate 2e-4;
- fixed int7 scales with mask 7 at learning rate 5e-5;
- a separate 240-value RSTB-bias-only recovery arm.

Every arm regressed at its sealed step-250 gate. The next unstarted hypothesis
is a zero-initialized tiny output residual adapter, which preserves exact G3
behavior at step zero and adds genuinely new representational capacity.

## Realtime 1080p-to-4K video selection

The selected RGB model has 8,496 parameters, a 16,992-byte FP16 payload, and
SHA-256
`c6b4f22d8c38270225b809e97d233eb68d7a48eef17595ac17a60b4540c72dce`.
On Apple M5 Chrome 150 WebGPU, 1920x1080 to 3840x2160 inference measured
8.9 ms median and 10.6 ms p95 over 120 frames. With H.264 enqueue it measured
9.4 ms median and 11.0 ms p95, produced 120/120 chunks, and reached maximum
queue depth two.

On the 120-frame Seedance action fixture, Clark improved RGB PSNR by
0.703365 dB and edge PSNR by 2.672525 dB over bicubic, but regressed
temporal-delta PSNR by 0.921971 dB. On the severe 200-pixel-wide compressed
fixture it trailed bicubic by 0.045264 dB RGB PSNR while improving edge PSNR by
0.241398 dB. Those failures are retained as promotion gates, not hidden by the
stronger examples.

The zero-regression 14-channel temporal arm improved both temporal metrics at
step 250, but its static-delta error regressed at step 500. Its sealed gate
stopped the run. The 12-channel RGB model remains the selected video default;
browsers without WebGPU `shader-f16` use ordinary high-quality browser
scaling.

## Data boundary

- The still-image corpus contains 206,232 rights-clean product/UI and
  multilingual images covering 204 language-script variants equally.
- The pinned Markov metadata index contains 48,477 screen recordings and
  approximately 12,281 total hours; its first balanced ingest list contains
  4,282 videos and 1,001.506 hours.
- Downloaded Markov samples are 640x360 and downloaded Humyn workflows are
  approximately 382/384x848/864. They are motion priors or evaluation inputs,
  not honest high-resolution reconstruction targets.
- OpenVidHD remains quarantined because its metadata does not close
  item-level source rights.
- Raw video blobs and the 22.8 MB generated index stay outside Git. Their
  pinned revisions, hashes, counts, split rules, and extraction receipts are
  committed in the experiment folder.

## Open gates

1. Train the zero-initialized output residual adapter from exact G3 and reject
   it unless the full 3,040-sample subgroup screen passes.
2. Preserve G3's spatial and edge gains while improving both temporal-delta and
   static-region consistency on source-disjoint real video.
3. Add tensor-level PyTorch/WGSL parity and real held-out OCR, flicker,
   one-pixel-line, and scroll/pan measurements.
4. Measure WebGPU-without-FP16, SIMD-WASM, scalar-WASM, Firefox, Safari, iOS,
   Android, x86, and lower-memory device profiles.
5. Do not claim q4 scientific replacement until one canonical artifact closes
   every row in `ACCEPTANCE_LEDGER.md`.
