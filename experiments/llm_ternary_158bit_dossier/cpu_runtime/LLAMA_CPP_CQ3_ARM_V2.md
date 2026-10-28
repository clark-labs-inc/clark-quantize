# llama.cpp CQ3 ARM row-blocked graph dispatch v2

Status: scoped K2/K11 operator gate passed
Date: 2026-08-17
Pinned llama.cpp: `4df29be4f4c3673f428170fda944a5b19f743bb8`

## Result

The Clark `CQ3_0 x CQ8_0` arithmetic is now dispatched from GGML's real
single-token `MUL_MAT` path on ARM64. The candidate reuses each quantized
activation group across up to 16 weight rows, instead of invoking one complete
row dot product at a time.

The preregistered Apple M5 Max comparison used the Qwen3.8-27B gate/up shape
`m=17408, n=1, k=5120`, Release builds from the same pinned upstream revision,
GGML's 18-thread `test-backend-ops` CPU configuration, two warmups, and nine
alternating process-level pairs.

| Measurement | Scalar GGML control | ARM row-blocked candidate |
| --- | ---: | ---: |
| Median latency | 175.92 us/run | 139.32 us/run |
| Median ratio | 1.000x | 1.263x |
| Frozen requirement | - | at least 1.15x and at most 150 us/run |
| Gate | - | passed |

All nine raw measurements are retained in the receipt. The paired speedup
ranged from `0.857x` to `1.686x`, with a paired median of `1.322x`. One pair
favored the control, so the evidence is a robust median improvement, not a
claim that every invocation is faster under laptop scheduling noise.

## Correctness and portability

- `test-cq3` passes the scalar-versus-dispatched dot test and a 17-row GEMV
  test that exercises one full 16-row block plus a one-row tail.
- Four GGML backend reference cases pass, including the exact 27B projection
  shape. GGML's registered normalized-MSE limit remains `5e-4`.
- The base patch plus ARM delta builds on the CCX33 AMD Milan x86 node with
  GCC 15.2.0, preserving the existing AVX2 path. `test-cq3` and all four CQ3
  backend cases pass there.

The x86 test is a portability and correctness result only; it is not a new x86
performance measurement.

## Frozen evidence

- Base patch SHA-256:
  `b5c584a338018ff7e509bc160c195693102fb6e265b6cbe4935d6e54d7cd67c1`
- ARM delta patch SHA-256:
  `f646aef73c5623af4f325f6923d9855cd698b71f4784e6795a8b658162661127`
- Preregistration SHA-256:
  `48b57265f5f2f2f75827a88563cb512d3660a2b82cc3e7ff16d760df285cce69`
- ARM receipt SHA-256:
  `4e487978b28c8aa63f48013ce7a54ff8b3e88adf1dc7597e7bf5ed646de53878`
- x86 portability receipt: `receipts/qwen38-llama-cq3-arm-x86-portability-v1.json`

The benchmark validates the complete source diff, six changed-file hashes,
CMake cache identities, rebuilt binary hashes, test output, host identity, and
every timing sample before writing the receipt atomically.

## Negative paths retained

The first native ARM hook replaced only the per-row scalar dot and improved the
real-shape GGML benchmark by about `1.03x`. It was rejected because it reloaded
the same activation for each row. Exploratory row blocking then produced:

| Rows held live | Exploratory latency |
| ---: | ---: |
| 4 | 147.60 us/run |
| 8 | 126.45 us/run |
| 16 | 123.68 us/run |
| 32 | 179.47 us/run |

The 32-row candidate was rejected because register pressure and spills erased
the activation-reuse gain. Sixteen rows was frozen for the authoritative run.
The exploratory numbers are selection evidence only and are not mixed into the
authoritative nine-pair result.

## Claim boundary and next gate

This passes only the K2/K11 ARM `CQ3_0` single-token `MUL_MAT` operator gate.
The benchmark uses deterministic synthetic weights at the exact Qwen shape,
not a full GGUF model artifact. It does not prove fused gate/up plus SwiGLU
graph dispatch, token parity, model quality, end-to-end throughput, energy, or
a scientific breakthrough.

The next runtime gate is to dispatch the already validated fused gate/up plus
SwiGLU implementation from the Qwen packed graph. Once a model-changing or
model-executing artifact exists, it must run the frozen CUTE-Quick-20 plus
MATH-500 preflight, or a pinned MMLU-style substitute, before any larger claim
or training expansion.

## Reproduction

Apply `llama_cpp_4df29be_cq3.patch`, then
`llama_cpp_4df29be_cq3_arm_v2.patch`, to the pinned llama.cpp revision. Configure
both control and candidate using the flags frozen in
`llama_cpp_cq3_arm_preregistration_v1.json`. The authoritative runner is:

```bash
python3 benchmark_llama_cpp_cq3_arm.py \
  --candidate-source /path/to/candidate/source \
  --candidate-build /path/to/candidate/build \
  --control-source /path/to/control/source \
  --control-build /path/to/control/build \
  --receipt /absolute/path/to/receipt.json
```
