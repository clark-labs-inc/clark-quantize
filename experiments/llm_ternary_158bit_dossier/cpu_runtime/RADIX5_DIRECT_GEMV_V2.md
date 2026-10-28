# Direct radix-5 GEMV v2 result

Status: **rejected by the preregistered performance gate**.

V2 changed only the digit decode operation: four ordinary L1 table loads are
assembled into an AVX2 vector instead of using an AVX2 gather. The 1.75-bit
group-128 weight image, activation layout, real Qwen tensor, control kernel,
cache eviction, sample count, comparison order, and promotion thresholds were
unchanged from v1.

| Physical cores | aligned-2 GB/s | radix-5 GB/s | radix-5 fraction |
|---:|---:|---:|---:|
| 1 | 15.266 | 4.807 | 38.23% |
| 2 | 24.973 | 8.882 | 43.19% |
| 4 | 27.423 | 15.406 | 68.22% |

The registered change improved four-core radix-5 throughput by 85.93% over v1
(15.406 versus 8.286 GB/s), while maintaining exact code round-trip,
determinism, and a maximum full-matrix output difference of
`1.1920928955078125e-6`. It nevertheless missed the fixed 70% retention gate
by 1.78 percentage points and is not promoted.

Disassembly confirms GCC emitted a specialized no-gather clone. It also shows
the digit table being constructed in a roughly 3 KiB stack frame on every
kernel call. A static compile-time table is a plausible, separately testable
follow-up; it must not retroactively change this v2 result.

Receipt: `receipts/radix5-direct-gemv-v2.json`

Claim boundary: this remains a single exact K2 microkernel experiment. It is
not evidence of model quality or end-to-end speed.
