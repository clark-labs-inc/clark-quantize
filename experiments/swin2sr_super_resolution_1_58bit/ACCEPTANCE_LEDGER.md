# Acceptance ledger

Updated: 2026-08-01

This ledger separates the deployed G3 ternary-tier improvement from the
stronger scientific claim that a single candidate replaces q4 without quality
loss. Evidence from different checkpoints is never combined into a synthetic
winner.

| Requirement | Current state | Required closing evidence |
|---|---|---|
| Start from the non-quantized Swin2SR teacher | Proven for G3 through the pinned `caidas/swin2SR-lightweight-x2-64` revision and selected lineage | Preserve exact teacher revision, initialization, data, optimizer, RNG, and source hashes for any future winner |
| Progressive hard-forward training | Proven for the ternary trunk and the selected recovery lineage | Future winner must retain continuous-master and hard-endpoint receipts from one canonical run |
| Sensitive mixed precision | Proven useful: G3 retains four RSTB convolutions as int7/output and image I/O as FP16 | Frozen precision map plus matched ablations and exact packed accounting |
| Mainly ternary representation | G3 is 83.5436% ternary by parameter count | Keep the ternary fraction, island identities, code widths, scales, metadata, and whole-artifact bits explicit |
| Honest Bonsai-style payload | G3 native transport is 380,107 bytes and 3.02468 lossless bits/model parameter; portable ONNX gzip is 808,002 bytes | Any successor reports native and portable bytes without calling the full mixed artifact exactly 1.58 bits |
| Better than previous Ternary Lite | Closed for G3: +0.209351 dB ground-truth, +0.150167 dB edge, +0.721767 dB teacher; all tested groups non-regressing | Preserve exact 3,040-sample comparison and subgroup receipts |
| Non-inferior to shipped q4 | Open: G3 is approximately -0.5060 dB ground-truth, -0.4316 dB edge, and -5.1694 dB teacher versus q4 headline means | Paired confidence intervals, p10, OCR, source, script, language, and worst-group gates on one sealed winner |
| Materially smaller than q4 | Closed for G3's deployable transport: 808,002 bytes versus 5,676,042 bytes for q4 | Recompute from the exact future winner and frozen size policy |
| Canonical identity across runtimes | Closed for G3's public model, Space, and BackdropKit portable graph; the shared gzip SHA is `a5e32c6…e085c` | Retain hash identity for every additional backend |
| WebGPU execution | Chromium WebGPU conformance passed; warm 64-to-128 fixture was 30.4 ms | Full product-size latency/memory plus non-FP16 WebGPU profiles |
| WASM/CPU fallback | Chromium forced-WASM, native scalar, SIMD-WASM, and scalar-WASM conformance passed | Firefox, Safari, iOS Safari, Android Chrome, x86, and low-memory receipts on the exact artifact |
| BackdropKit integration | G3 is deployed as the replacement for the older ternary artifact; public package and product hashes match | Keep q4 scientific claims separate until q4 non-inferiority closes |
| Realtime video | 12-channel RGB model is deployed on WebGPU FP16; 1080p-to-4K kernel measured 8.9 ms median on M5 | Temporal consistency, OCR, parity, broader hardware, fallback acceleration, and muxed export gates |

## Current product/science boundary

- **Accepted quality reference:** shipped q4.
- **Deployed ternary tier:** G3, replacing the older g2 Ternary Lite bytes.
- **Selected realtime video tier:** 12-channel RGB student on eligible WebGPU
  FP16 devices, with ordinary browser scaling elsewhere.
- **Rejected successors:** fixed-int7 scale corrections, RSTB-bias correction,
  and the step-500 14-channel temporal arm.
- **No active training:** the next still-image and temporal studies are open
  hypotheses, not running results.

Promotion closes only when every required row refers to the same canonical
artifact. Until then, deployment authorization is a product tradeoff, not a
claim of q4-equivalent scientific quality.
