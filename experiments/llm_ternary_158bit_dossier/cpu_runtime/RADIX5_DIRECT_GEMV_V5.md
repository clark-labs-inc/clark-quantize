# Direct radix-5 GEMV v5 result

Status: **rejected by the preregistered performance gate**.

V5 returned to v2's no-gather lookup decoder and processed two output rows at
once. The registered goal was to expose independent table-load and integer-dot
work while sharing each activation vector between the two rows.

| Physical cores | aligned-2 GB/s | radix-5 GB/s | radix-5 fraction |
|---:|---:|---:|---:|
| 1 | 15.673 | 3.632 | 28.14% |
| 2 | 25.918 | 6.857 | 32.12% |
| 4 | 27.561 | 11.946 | 52.63% |

The output hash matches v2, the code round-trip was exact, the maximum
full-matrix difference was `1.1920928955078125e-6`, and thread outputs were
deterministic. Pairing nevertheless reduced four-core throughput by 22.46%
relative to v2, consistent with register pressure and a larger hot loop. V5
missed the fixed 70% gate and is not promoted.

This closes the initial AVX2 direct-radix-5 branch. Across gather, no-gather,
static-table, arithmetic-plane, and paired-row variants, v2 is the best at
68.22% of aligned-2 throughput. The supported CPU compute image therefore
remains the promoted 2.125-bit aligned-2 format; 1.75-bit radix-5 remains a
storage candidate, not a promoted execution format.

Receipt: `receipts/radix5-direct-gemv-v5.json`

Claim boundary: single K2 microkernel only; no model-quality or end-to-end
performance claim.
