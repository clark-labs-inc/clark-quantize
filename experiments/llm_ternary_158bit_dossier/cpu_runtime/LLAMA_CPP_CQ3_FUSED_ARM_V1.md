# llama.cpp CQ3 fused gate/up/SwiGLU graph v1

Status: scoped ARM graph-kernel gate passed; model capability gate failed
Date: 2026-08-17
Pinned llama.cpp: `4df29be4f4c3673f428170fda944a5b19f743bb8`

## Result

GGML's CPU graph executor now recognizes the exact single-token sibling graph

1. `CQ3_0 gate_weights x F32 activation`,
2. `CQ3_0 up_weights x the same F32 activation`, and
3. un-swapped F32 SwiGLU.

On ARM64, the candidate quantizes the shared activation once, reuses each CQ8
group across eight gate rows and eight up rows, and applies GGML's native
SwiGLU epilogue without materializing two complete projection outputs.

The preregistered Apple M5 Max comparison used one binary with fusion disabled
or enabled through `GGML_CPU_DISABLE_FUSION`, the exact Qwen3.8-27B projection
shape `17408 x 5120`, two warmups, nine alternating process-level pairs, and
corrected whole-graph timing normalization.

| CPU threads | Unfused median | Fused median | Ratio of medians | Paired median | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2341.94 us | 2225.65 us | 1.052x | 1.054x | passed |
| 4 | 769.60 us | 702.56 us | 1.095x | 1.100x | passed |
| 8 | 529.62 us | 506.63 us | 1.045x | 1.045x | passed |

Every authoritative pair favored the fused candidate. The smallest paired
speedup was `1.039x`, `1.065x`, and `1.035x` in the 1-, 4-, and 8-thread lanes,
respectively. This is a useful graph-level gain, not an end-to-end model claim.

## Correctness and portability

- The direct paired ARM kernel is bit-identical to two independent row-blocked
  CQ3 kernels for both gate and up outputs.
- GGML reference checks pass for a 17-row tail case and the exact
  `17408 x 5120` Qwen shape.
- A fresh GCC 15.2 build on the `quant` AMD EPYC Milan node passes `test-cq3`
  and both graph-reference cases. The fusion remains ARM-gated, so this is an
  x86 portability result rather than an x86 fusion result.
- The standalone and paired kernel contracts are explicitly bounded at 16 and
  8 rows, respectively. A pre-commit review caught the assertions reversed;
  the source-bound benchmark and x86 receipts were rerun after correction.
- The performance harness now forces one whole-graph run when only the final
  tensor is duplicated. The earlier 2-3 microsecond exploratory readings were
  invalid normalization artifacts and are not retained as performance data.

## Model-level quick gate

The exact 1,193,348,832-byte Qwen3.5-4B CQ3 artifact with SHA-256
`072ffb19ae20fdabb0a64f4e49097f8a77fbabd49fd5eb47bc705b5012950476`
was executed CPU-only on the M5 Max with the same server binary in fused and
unfused modes.

All semantic record fields were exact across the two modes for 34 prompts:

- 20 frozen CUTE-Quick character items,
- 8 frozen MATH-500 items, and
- 6 frozen code-trace items.

This passes the runtime-parity gate: the fusion does not introduce additional
behavior change. It does **not** rescue the model artifact. Against the exact
BF16 parent, the fused CQ3 artifact scored `0/20` CUTE versus `8/20`, and `0/8`
MATH-500 versus `5/8`. The comparison exited `2` as designed and the receipt
retains `stop_for_catastrophic_forgetting`.

The scientific conclusion is therefore split cleanly:

- promote the scoped ARM graph-kernel integration; and
- reject the current CQ3 PTQ weights for any model, layer-depth, or 27B
  capability claim.

## Selection evidence and negative paths

The first paired kernel used 16 rows from both matrices, creating 32 live NEON
accumulators. It was neutral or slower at 1 and 8 threads and clearly unstable
at 18 threads, consistent with register spills. Four paired rows were exact
but slower than eight rows at the primary 1- and 4-thread settings. Eight rows
was frozen before the authoritative run.

The 18-thread laptop lane was excluded from promotion because mixed-core
scheduling produced high process-level variance. Its exploratory samples are
selection evidence only.

## Frozen evidence

- Base patch SHA-256:
  `b5c584a338018ff7e509bc160c195693102fb6e265b6cbe4935d6e54d7cd67c1`
- ARM row-blocked patch SHA-256:
  `f646aef73c5623af4f325f6923d9855cd698b71f4784e6795a8b658162661127`
- Fused graph delta SHA-256:
  `cbde7d2e4c23a13eb24207ccea6ceb75d63eeb7a37dfba20aa95fcae9968d5b4`
- Preregistration SHA-256:
  `2613e64d21771205787a92a642ead5010e174fb9b5f1a8005c7928a2e53039b4`
- ARM performance receipt SHA-256:
  `1ab203ef47c789759fc31968b4637d1151ab8d077f9d99cfeb9326d4089844a9`
- x86 portability receipt SHA-256:
  `928cbbbd9fa0b70311b5315d50da4531849eca6c313d4d0a8c37f542527aa2de`
- Model parity/capability receipt SHA-256:
  `ba2c51dcbc75614528ab8f73ba0bf85235af519b52170da5366f2423153cb150`

## Claim boundary and next gate

This passes the K2/K11 ARM single-token fused gate/up/SwiGLU graph-kernel gate.
It does not establish a quality-retaining ternary model, end-to-end token
speedup, prefill performance, energy advantage, 27B feasibility, or a
scientific breakthrough.

The next model gate is not another kernel tweak. It is a quality-aware ternary
training candidate whose exact artifact passes CUTE-Quick-20 plus quick
MATH-500 (or the frozen MMLU substitute) before deeper training or 27B scaling.

## Reproduction

Apply, in order, to the pinned llama.cpp revision:

1. `llama_cpp_4df29be_cq3.patch`
2. `llama_cpp_4df29be_cq3_arm_v2.patch`
3. `llama_cpp_4df29be_cq3_fused_arm_v1.patch`

Then run the frozen benchmark:

```bash
python3 benchmark_llama_cpp_cq3_fused_arm.py \
  --source /path/to/source \
  --build /path/to/build \
  --receipt /absolute/path/to/receipt.json
```
