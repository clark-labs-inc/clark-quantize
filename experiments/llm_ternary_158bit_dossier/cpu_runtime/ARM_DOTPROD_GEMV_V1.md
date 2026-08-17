# M5 Max aligned-2 DOTPROD GEMV v1

Status: preregistered ARM K2 milestone passed; full K2 exit remains open.

The first ARM64 implementation consumes the same group-128 ternary codes and
FP16 scales as the AVX2 implementation. It decodes the 2-bit transposed compute
image in NEON registers, uses signed int8 DOTPROD for exact integer products,
applies group scales through four FP32 FMA lanes, and performs one fixed
pairwise reduction per output row.

The authoritative input is the real Qwen3.8-27B layer-0 MLP gate projection,
shape `17408 x 5120`, extracted from the promoted canonical artifact. The
fixture is bound to:

- model revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- artifact manifest SHA-256 `c98d878d...18b75b4`;
- codes SHA-256 `d660e496...f53b41ac`;
- FP16 scales SHA-256 `55fdcd7f...13f731d1`.

## Authoritative result

Host: Apple M5 Max, 18 CPU cores, 64 GiB unified memory. Every timed kernel
sample follows a 512 MiB eight-worker NEON cache-eviction read. The workers are
persistent, but macOS exposes no process affinity API, so placement remains a
measured scheduler property rather than a pinned-core claim.

| Workers | GEMV GB/s | NEON read GB/s | Read-roofline fraction | Median GEMV |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21.13 | 24.34 | 86.81% | 1.120 ms |
| 2 | 38.51 | 47.32 | 81.40% | 0.615 ms |
| 4 | 60.39 | 89.86 | 67.20% | 0.392 ms |
| 8 | 59.82 | 169.99 | 35.19% | 0.396 ms |

Additional gates:

- scalar speedup on 1,024 rows: **28.38x**;
- exact pack/unpack: pass;
- exact output across 1/2/4/8 workers: pass;
- maximum absolute error against scalar: `7.152557e-7`;
- compute image including FP16 group scales: exactly `2.125` bits/weight;
- frozen v1 promotion decision: pass.

Receipt: `receipts/qwen38-arm-dotprod-v1.json`, file SHA-256
`4c8d4d04c1fb757f6899118a35c27e61280ffc8a70bce59fa9516e3f7c98ac1f`.

## Interpretation and next optimization

Four-worker throughput exceeds the frozen 50 GB/s threshold but misses the
program-wide 70% K2 roofline target by 2.8 percentage points. Eight workers do
not improve elapsed time: the 23.7 MB projection is dispatched in about 0.4 ms,
and the Python task boundary plus dot-product instruction throughput binds well
before M5 Max memory bandwidth does.

The next ARM experiment should move dispatch below Python and fuse the gate and
up projections so one native worker rendezvous reuses the quantized activation
and produces both SwiGLU inputs. That is the first K4 rung and the relevant
end-to-end direction; tuning a synthetic loop until it crosses 70% would not
prove graph impact.

This result does not establish model quality, full-layer or graph parity,
token throughput, energy efficiency, or a scientific breakthrough.
