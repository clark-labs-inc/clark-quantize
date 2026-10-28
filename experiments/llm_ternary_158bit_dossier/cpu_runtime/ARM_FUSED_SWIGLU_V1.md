# M5 Max fused ternary SwiGLU v1

Status: preregistered ARM K4 native-operator milestone passed; graph gate open.

The candidate executes the real Qwen3.8-27B layer-0 MLP gate and up
projections in one native operation, reuses the same group-128 A8 activation,
and writes only `SiLU(gate) * up`. The control uses the same ARM DOTPROD kernel
twice and the same native FP32 SwiGLU function, so the measured difference is
fusion and dispatch rather than a changed arithmetic contract.

Both `17408 x 5120` fixtures are authenticated children of model revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and canonical artifact manifest
`c98d878d...18b75b4`. The frozen preregistration SHA-256 is
`9fb73d8345651eb7bda96921a023a9404b50646293309c317d13ea8f9a149384`.

## Authoritative result

Host: Apple M5 Max, 18 CPU cores, 64 GiB unified memory. Every sample follows
a 512 MiB NEON cache-eviction read. Worker pools persist across samples;
macOS does not expose a process-affinity API.

| Workers | Separate median | Fused median | Fused weight GB/s | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.244 ms | 2.276 ms | 20.81 | 0.986x |
| 2 | 1.254 ms | 1.217 ms | 38.92 | 1.031x |
| 4 | 0.833 ms | 0.771 ms | 61.45 | 1.081x |
| 8 | 0.780 ms | 0.612 ms | 77.43 | 1.276x |

Additional gates:

- exact gate and up pack/unpack: pass;
- bit-exact output versus the separate native control: pass at every worker
  count;
- bit-exact output across worker counts: pass;
- two-projection compute image including FP16 scales: `2.125` bits/weight;
- frozen v1 promotion decision: pass.

Receipt: `receipts/qwen38-arm-swiglu-v1.json`, file SHA-256
`7a069eedf3cef9ed981f0a8575d71740ff9c0de67bf5d0b5f53560e99b01732c`.

## Claim boundary

The result validates K4 as a native ARM operator and shows that fusion improves
the useful eight-worker regime without materially harming the one-worker
regime. It does not yet prove that the model graph dispatches this operator,
that generated tokens match the packed reference, that language capability is
retained, or that full-model token throughput or energy beats int4. The next
gate is GGML graph integration followed by numerical/token parity and the
mandatory WikiText, MATH-500, and CUTE/MMLU quick capability receipt.
