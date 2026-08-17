# Direct radix-5 GEMV v3 result

Status: **rejected by the preregistered performance gate**.

V3 moved the 256-entry digit table from per-call stack construction into
immutable compile-time storage. Everything else—including the no-gather
decode, 1.75-bit weights, real Qwen tensor, control, cache protocol, and 70%
retention gate—was unchanged from v2.

| Physical cores | aligned-2 GB/s | radix-5 GB/s | radix-5 fraction |
|---:|---:|---:|---:|
| 1 | 15.656 | 4.754 | 36.87% |
| 2 | 26.070 | 8.965 | 41.75% |
| 4 | 27.038 | 14.934 | 67.07% |

Correctness remained exact at the code level and within
`1.1920928955078125e-6` for the full-matrix output. Thread-count outputs were
deterministic. Four-core throughput was 1.68% lower than v2 and missed the
fixed gate, so v3 is not promoted.

This falsifies the specific claim that call-time table construction explains
v2's residual gap. The remaining lookup path assembles seven vectors from 26
indirect table loads per group and uses a 224-byte expanded activation image.
The next distinct design should decode all 26 bytes arithmetically into five
contiguous digit planes and use a 160-byte activation image.

Receipt: `receipts/radix5-direct-gemv-v3.json`

Claim boundary: single K2 microkernel only; no model-quality or end-to-end
performance claim.
