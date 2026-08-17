# Qwen3.5-4B Layer-30 MMLU Replication

Status: completed positive one-layer capability replication

Run date: 2026-08-17 UTC

Decision: authorize one separately preregistered experiment that adds exactly
one more MLP layer. Broad depth expansion, whole-model ternary conversion, and
27B promotion remain unauthorized.

## Outcome

The exact v4 layer-30 CPU-QAT artifact completed the frozen 64-row, five-shot,
subject-balanced MMLU slice at 43/64 (67.1875%). Direct PTQ also scored 43/64,
and BF16 scored 41/64 (64.0625%). The candidate therefore passed both frozen
rules: it did not lose more than three items versus BF16, and it was at least
as accurate as direct PTQ.

The candidate and direct-PTQ artifacts made identical predictions on all 64
rows. This MMLU slice therefore supports capability preservation for the
one-layer candidate, but it does not show a behavioral advantage from QAT over
direct PTQ. The +2 candidate count versus BF16 is not claimed as an
improvement: the paired slice is small, and the Wilson intervals overlap.

| Arm | Correct | Accuracy | Wilson 95% CI | Report content SHA-256 |
| --- | ---: | ---: | --- | --- |
| BF16 | 41/64 | 64.0625% | 51.82% to 74.71% | `86f45339928231083cfcebc905a3b32fbe1095aca09cc564ecd555fe58223d87` |
| Direct PTQ, layer 30 | 43/64 | 67.1875% | 55.00% to 77.43% | `5ee82eaded8f797fb67581557c36805c9819c6d9a140665dcd3238f528337bf6` |
| CPU-QAT v4, layer 30 | 43/64 | 67.1875% | 55.00% to 77.43% | `7d4ccb140849b3c0063ae51cf8ddf21c80c0c0458dc96d988e310835e9b5f18e` |

The paired candidate-versus-BF16 comparison contains four candidate-only
correct rows, two BF16-only correct rows, 57 identical predictions, and 58
rows with identical correctness. Candidate versus direct PTQ has 64 identical
predictions and 64 identical correctness outcomes.

## Frozen protocol

The manifest contains 64 test rows from 57 MMLU subjects. It uses five fixed
development examples from each row's subject and deterministic near-equal
subject stratification with seed 20260816. Dataset revision
`c30699e8356da336a370243923dbaf21066bb9fe` and both test/support fingerprints
are pinned. The manifest file SHA-256 is
`e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba`.

llama.cpp applied the Qwen chat template with thinking disabled. The scorer
allowed exactly one of the four single-token pieces ` A`, ` B`, ` C`, or ` D`,
requested post-sampling probabilities after only a temperature-1 sampler, and
selected the largest of the four normalized probabilities. Every arm used the
same evaluator SHA-256
`a604ca9a528231fa17de78ab2a7ffed7c1b9a0b9a9ca74459c2d894f1f3612d3`,
four workers, four server slots, and checkpoints every four rows.

## Infrastructure-invalid first attempt

The original preregistration allocated 2,048 tokens per server slot. BF16
completed and checkpointed 20 rows, then a 2,789-token request failed with an
HTTP 400 context-size error. That attempt is infrastructure-invalid and is not
reported as 20/64 or as a model failure.

Before either quantized arm began, an explicit amendment increased only the
total server context from 8,192 to 32,768 tokens, leaving four slots and thus
8,192 tokens per slot. The first 20 contract-checked BF16 records were retained
and resumed. The manifest, rows, answers, prompts, models, scorer, seeds,
checkpoint cadence, and decision thresholds did not change. The partial BF16
count of 13/20 was observable before the amendment; that disclosure is frozen
in `qwen35_4b_cpu_layer30_mmlu_context_amendment_v1.json`. The invalid partial
and failure log are preserved here with exact hashes.

## Relationship to mandatory quick gates

This MMLU replication is additional evidence, not a replacement for the
mandatory capability harness. The same exact v4 artifact had already completed
CUTE-Quick-20 at 7/20, pinned MATH-500 quick at 5/8, sealed code traces at 4/6,
and WikiText evaluation at 8.6729 perplexity. Candidate promotion continues to
require those behavioral checks; loss, KL, or reconstruction alone cannot
authorize greater depth.

## Scope and next gate

This is a paired one-layer Qwen3.5-4B replication, not a leaderboard result,
not a whole-model ternary model, not a Qwen3.8-27B result, and not a
breakthrough. The machine decision is
`authorize_preregistered_two_layer_experiment`, with
`broad_depth_expansion_authorized=false`.

The next experiment may quantize the existing layer-30 MLP plus exactly one
additional preregistered MLP layer. It must again run the mandatory CUTE,
MATH-500, code, and WikiText gates, followed by this frozen MMLU replication or
an equally strict preregistered capability gate before any further expansion.

