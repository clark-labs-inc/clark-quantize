# Direct radix-5 GEMV v1 result

Status: **rejected by the preregistered performance gate**.

The v1 prototype computes directly from a group-128 radix-5 image. Each group
stores 26 radix-5 bytes plus one FP16 scale, or 1.75 bits per weight. The
comparison used the promoted 27B artifact's
`model.language_model.layers.0.mlp.gate_proj.weight` tensor (17408 x 5120),
the same group scales and quantized activation for both formats, two warmups,
seven cache-cold samples, alternating execution order, and workers pinned to
distinct physical cores on the CCX33 node.

## Result

| Physical cores | aligned-2 GB/s | radix-5 GB/s | radix-5 fraction |
|---:|---:|---:|---:|
| 1 | 15.439 | 2.359 | 18.55% |
| 2 | 26.188 | 4.548 | 21.09% |
| 4 | 27.198 | 8.286 | 36.99% |

The radix-5 image is 17.65% smaller than the 2.125-bit aligned-2 image. Its
round-trip was exact, maximum full-matrix output error was
`1.1920928955078125e-6`, and outputs were deterministic across thread counts.
Those correctness results pass. Four-core throughput missed the preregistered
70% retention gate, so v1 is not promoted.

The likely bottleneck is the AVX2 gather used for every four radix-5 bytes.
Any arithmetic plane-decoder alternative must be registered as a new
experiment and compared against this immutable negative result.

Receipt: `receipts/radix5-direct-gemv-v1.json`

Claim boundary: this is a single exact K2 microkernel experiment. It says
nothing about model quality, end-to-end token throughput, or a breakthrough.
