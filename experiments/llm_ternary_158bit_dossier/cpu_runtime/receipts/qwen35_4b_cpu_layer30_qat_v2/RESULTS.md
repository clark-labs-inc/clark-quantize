# Qwen3.5-4B Corrected Artifact-Exact Layer-30 CPU QAT

Status: completed negative result

Run date: 2026-08-17 UTC

Decision: reject the candidate for depth expansion; keep the corrected
training/export path and the exact-artifact receipts

## Outcome

The corrected CPU-only experiment successfully trained, exported, packed, and
evaluated one Qwen3.5-4B MLP triplet under one numerical contract. The hard
forward, BF16 patch, and CQ3 GGUF contain the same group-128 ternary values.
Only `blk.30.ffn_gate.weight`, `blk.30.ffn_up.weight`, and
`blk.30.ffn_down.weight` are CQ3; all 438 non-target tensors are byte-identical
across the BF16 parent, direct-PTQ control, and QAT candidate.

This is not a quality win. QAT reduced cached local reconstruction loss, but it
did not beat direct PTQ under the frozen generation gate. It lost one CUTE
item versus direct PTQ and three versus BF16. Its small WikiText improvement
over PTQ is far below the uncertainty of the four-chunk preflight. The frozen
training-value rule therefore rejects depth expansion.

## Exact artifacts

| Artifact | File bytes | SHA-256 |
| --- | ---: | --- |
| BF16 parent GGUF | 8,665,620,192 | `594f6572cebaa26b8cb8a72a675e81f9a27133bbc52fa5d202310b6196a3b974` |
| direct-PTQ layer-30 CQ3 control | 8,542,863,136 | `3e1ba3a29a38de8ff67f7b508563a74b3ea84cdabfa385f0a0b3a749d75e7a9b` |
| corrected-QAT layer-30 CQ3 candidate | 8,542,863,168 | `b1fa2c8323880fe386f6ec3923973ca4cb3b00e8d9593fea8eba574f758e052e` |
| QAT BF16 bridge GGUF | 8,665,620,288 | `4ce54f1462bed7ab1d2623c36ab3a7d016eae6f6f27686d04eb34d60347f5863` |
| local and survival patch, each | 141,558,144 | `168f9debb63a4e0822e3c6659912ec83a78904158965b97fb4e159a0da5f9cfe` |

The survival patch is byte-identical to the local-QAT patch. Its two low-KL
optimization steps changed FP32 masters but changed no exported BF16 ternary
value, so the survival phase receives no artifact-level credit.

## Training signals

The eight cached-reconstruction steps reduced normalized local loss from
`0.37304896` to `0.25528288` (31.57%). The two completion-survival steps
reported KL `0.02136561` then `0.00552006`, with teacher top-1 alignment
`0.95833` on both rows. These teacher-forced signals did not predict the
generation result and cannot authorize promotion.

## Mandatory exact-artifact quality gate

| Frozen signal | BF16 | direct PTQ | corrected QAT | QAT decision |
| --- | ---: | ---: | ---: | --- |
| CUTE-Quick-20 | 8/20 | 6/20 | 5/20 | reject |
| pinned MATH-500 quick | 5/8 | 5/8 | 5/8 | tie |
| sealed code traces | 4/6 | 4/6 | 4/6 | tie |
| WikiText-2 four-chunk PPL | 8.4960 +/- 0.72338 | 8.6975 +/- 0.73256 | 8.6848 +/- 0.73685 | indistinguishable from PTQ |

QAT versus direct PTQ produced exactly equal raw output on 17/20 CUTE items
and 14/14 combined MATH/code items. QAT had zero exclusive wins and PTQ had
one exclusive CUTE win. QAT perplexity is only 0.146% below PTQ, while it is
2.222% above BF16.

The catastrophic-forgetting wrapper says `continue_to_full_evaluation`
because 5/20 is exactly the ceiling-rounded 60%-retention floor relative to
BF16's 8/20. That coarse floor is only an emergency stop rule. It is not a
promotion rule, and it does not override the preregistered requirement that
QAT improve at least one frozen signal without worsening another.

## Correction and evidence boundary

The earlier v1 GGUF generation scores remain valid for their exact files, but
v1 cannot support an artifact-exact-training claim. Its hard forward and
export bridge rounded values differently. V2 starts fresh from BF16, trains
against BF16-rounded CQ3 values, and proves exact selective export. V2 is
nearly behavior-identical to v1: 19/20 raw CUTE outputs and all 14 MATH/code
outputs match, so fixing the numerical contract did not recover quality.

## Scientific read and next experiment

One-layer hard QAT on the current 288-position reconstruction cache is not
better than direct PTQ. The next run must not add layers. It should first
measure train-versus-held-out layer reconstruction for BF16, PTQ, and QAT,
then enlarge the non-evaluation calibration distribution and select an exact
discrete artifact on held-out loss. Every selected artifact must rerun
CUTE-Quick-20 plus pinned MATH-500 before any depth expansion.

Large patches, checkpoints, and GGUF files remain on `quant`; Git retains only
the compact reports and checksums needed to reproduce and audit the decision.
