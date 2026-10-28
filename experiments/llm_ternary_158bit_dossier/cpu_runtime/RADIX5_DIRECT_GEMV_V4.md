# Direct radix-5 GEMV v4 result

Status: **rejected by the preregistered performance gate**.

V4 replaced indirect table loads with exact SIMD division by three. Two
16-lane unsigned-16 vectors decode the 26 radix-5 bytes into five digit
planes. The token activation is transposed once into a 160-byte plane-major
image per group instead of the earlier 224-byte segment-major image.

| Physical cores | aligned-2 GB/s | radix-5 GB/s | radix-5 fraction |
|---:|---:|---:|---:|
| 1 | 15.651 | 3.561 | 27.63% |
| 2 | 26.614 | 6.643 | 30.31% |
| 4 | 27.120 | 11.553 | 51.73% |

The format still uses 1.75 bits per weight including FP16 scales. Code
round-trip was exact, the full-matrix output difference was
`1.6689300537109375e-6`, and all thread-count outputs were deterministic.
Performance missed the fixed 70% gate, so v4 is not promoted.

This result rejects the arithmetic plane decoder on AVX2: eliminating
data-dependent loads did not repay the five rounds of widen, multiply, shift,
remainder, pack, and dot-product instructions. V2's ordinary-load lookup
kernel remains the best dense 1.75-bit candidate at 68.22%.

Receipt: `receipts/radix5-direct-gemv-v4.json`

Claim boundary: single K2 microkernel only; no model-quality or end-to-end
performance claim.
