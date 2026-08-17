# Qwen3.5-4B Native CQ3 PTQ Milestone

Status: completed negative result

Run date: 2026-08-17 UTC

Decision: reject this PTQ artifact; retain the runtime and move to QAT export

## Outcome

The native CPU path is real and end to end: a new GGML `CQ3_0` weight type,
group-128 FP16 scales, AVX2 matrix-vector execution, embedding `GET_ROWS`, GGUF
loading, Python GGUF inspection, and Qwen generation all execute on the CCX33
CPU node. The portable patch is based on llama.cpp revision
`4df29be4f4c3673f428170fda944a5b19f743bb8`.

The model result is not a breakthrough. Direct PTQ catastrophically destroys
the 4B model, and the scope-matched upstream `TQ2_0` control is both smaller and
about 4% faster at decode. The runtime is a usable deployment target for QAT;
the PTQ weights are rejected.

## Artifact accounting

| Artifact | File bytes | SHA-256 |
| --- | ---: | --- |
| BF16 | 8,665,620,192 | `594f6572cebaa26b8cb8a72a675e81f9a27133bbc52fa5d202310b6196a3b974` |
| CQ3 scope-matched PTQ | 1,193,348,832 | `072ffb19ae20fdabb0a64f4e49097f8a77fbabd49fd5eb47bc705b5012950476` |
| TQ2 scope-matched PTQ | 1,159,689,952 | `597dc394be3112c7cd9d025df0b0427da7b0c7a24b65f303349aab2804ef769b` |

CQ3 stores 256 weights in 68 bytes: 64 bytes of two-bit codes and two FP16
group-128 scales. That is 2.125 raw bits per quantized weight. The full GGUF is
2.207 bits per logical model parameter including protected tensors and
container bytes; llama.cpp reports 2.19 BPW for its model payload. This does
not meet the approximately 1.58-bit whole-model goal.

Conversion covered 208 CQ3 tensors out of 441 tensors. Forty-eight
`ssm_alpha`/`ssm_beta` tensors and one `nextn` projection were explicitly kept
in BF16. Conversion was atomic, took 62.20 seconds, peaked at 12,569,352 KiB
RSS, and exited zero.

## Four-physical-core performance

`llama-bench`, pinned to CPUs `0,2,4,6`, five repetitions, prompt 128 and
generation 32:

| Model | Prefill tok/s | Decode tok/s | Bench payload bytes |
| --- | ---: | ---: | ---: |
| BF16 | 16.629 | 4.199 | 8,654,651,392 |
| CQ3 | 17.838 | 26.153 | 1,182,380,032 |
| TQ1 | 16.694 | 18.041 | 1,303,662,592 |
| scope-matched TQ2 | 17.331 | 27.201 | 1,148,721,152 |

CQ3 decodes 6.23 times faster than BF16. It does not beat the appropriate
ternary control: TQ2 decodes 4.01% faster and its full file is 2.90% smaller.

## Mandatory quality tripwire

| Gate | BF16 | CQ3 PTQ | Result |
| --- | ---: | ---: | --- |
| WikiText-2, first four 512-token chunks | PPL 8.496 | PPL 29,536.685 | catastrophic |
| pinned MATH-500 quick slice | 5/8 | 0/8 | fail |
| sealed code traces | 4/6 | 0/6 | fail |
| CUTE-Quick-20 character slice | 8/20 | 0/20 | fail |

The CUTE lane uses official data and prompts from CUTE commit
`4238eaa301abb698d8850f42297564362d4e9032`, but only 20 fixed rows. It is a
development tripwire, not a full or leaderboard-comparable CUTE score.

The atomic comparator exits 2 with `stop_for_catastrophic_forgetting`. The CQ3
artifact often generates blank strings, punctuation, or repeated digits, so
the result is not an answer-normalization artifact.

## Scientific read

The experiment separates two questions:

1. Can a native group-128 ternary representation execute usefully on this CPU?
   Yes: the graph runs and reaches 26.15 decode tok/s.
2. Can abrupt groupwise PTQ produce a useful ternary Qwen model? No: every
   generation-level capability lane and perplexity reject it.

The next discriminating experiment is therefore not another PTQ threshold
search. It is to export or reproduce the previously successful 4B hard-ternary
QAT state under the exact CQ3 scale, rounding, and accumulation contract. That
candidate must rerun the quick MATH-500 and CUTE tripwire before any long
evaluation, deeper quantized scope, or 27B work.
