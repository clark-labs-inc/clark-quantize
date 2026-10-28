# Qwen3.5-4B Held-Out-Selected Layer-30 CPU QAT

Status: completed positive diagnostic result

Run date: 2026-08-17 UTC

Decision: broaden the non-evaluation calibration and held-out experiment;
do not expand quantized depth

## Outcome

Held-out exact-artifact selection fixed the failure mode exposed by v2. V3
trained on 144 activation positions, selected among exact hard artifacts on 72
different positions, and evaluated reconstruction on 72 independent positions
that were used for neither optimization nor selection. The selected step-32
patch improved independent-test reconstruction by 27.67% relative to direct
PTQ.

The exact CQ3 GGUF also improved the mandatory generation signals without
regressing MATH or code: CUTE rose from direct PTQ's 6/20 to 7/20, MATH stayed
5/8, and code stayed 4/6. BF16 scores 8/20 on the same CUTE slice, putting v3
exactly five absolute percentage points below BF16. Four-chunk WikiText PPL
moved from direct PTQ's 8.6975 to 8.6523; this is directionally positive but
too small relative to uncertainty to claim as a likelihood win.

This is a successful one-layer training-method rung, not a whole-model result
and not a breakthrough. The preregistration explicitly forbids depth expansion
from this small diagnostic.

## Exact artifacts and scope

| Artifact | File bytes | SHA-256 |
| --- | ---: | --- |
| BF16 parent GGUF | 8,665,620,192 | `594f6572cebaa26b8cb8a72a675e81f9a27133bbc52fa5d202310b6196a3b974` |
| direct-PTQ layer-30 CQ3 control | 8,542,863,136 | `3e1ba3a29a38de8ff67f7b508563a74b3ea84cdabfa385f0a0b3a749d75e7a9b` |
| v2 full-cache QAT CQ3 | 8,542,863,168 | `b1fa2c8323880fe386f6ec3923973ca4cb3b00e8d9593fea8eba574f758e052e` |
| v3 held-out-selected QAT CQ3 | 8,542,863,168 | `e3790341b0946801a881419e37beb02bf784653b336668f4366d28f7717de8e2` |
| v3 selected BF16 patch | 141,558,144 | `9f1c71d44c3eec637a6eb929ecf197282a51f00c17d7bb2b5485b77f0ada7013` |

Only `blk.30.ffn_gate.weight`, `blk.30.ffn_up.weight`, and
`blk.30.ffn_down.weight` are CQ3. The scope receipt proves all 438 non-target
tensors are byte-identical across BF16, direct PTQ, and v3.

## Frozen reconstruction split

| Split | Positions | direct PTQ loss | v3 loss | v3 relative improvement |
| --- | ---: | ---: | ---: | ---: |
| training | 144 | 0.363190 | 0.097250 | 73.22% |
| validation selection | 72 | 0.386815 | 0.344431 | 10.96% |
| independent test | 72 | 0.346257 | 0.250436 | 27.67% |

The validation sequence improved monotonically from step-zero direct PTQ
`0.386815` to `0.344431` at step 32, so the final preregistered checkpoint was
selected. The independent test contains the second math, instruction, and
reasoning rows and is not used by AdamW or artifact selection.

## Mandatory exact-GGUF quality gate

| Frozen signal | BF16 | direct PTQ | v2 QAT | v3 QAT | V3 result |
| --- | ---: | ---: | ---: | ---: | --- |
| CUTE-Quick-20 | 8/20 | 6/20 | 5/20 | 7/20 | pass boundary |
| pinned MATH-500 quick | 5/8 | 5/8 | 5/8 | 5/8 | tie |
| sealed code traces | 4/6 | 4/6 | 4/6 | 4/6 | tie |
| WikiText-2 four-chunk PPL | 8.4960 | 8.6975 | 8.6848 | 8.6523 | no regression |

Against direct PTQ, v3 has one exclusive CUTE win, no exclusive loss, 17/20
exact CUTE outputs, and 14/14 exact MATH/code outputs. Against BF16, v3 has no
exclusive win and one exclusive CUTE loss. V3 PPL is 0.520% below direct PTQ,
0.374% below v2, and 1.840% above BF16.

## Infrastructure evidence boundary

The initial systemd launch used a nonexistent service user and exited 217
before training. The first real wrapper run completed training but then exited
nonzero because the evaluator imported an older checkout from inherited
`PYTHONPATH`. No reconstruction score was reported from either invalid wrapper
attempt. The evaluator was pinned to the clean checkout, the original training
source commit was preserved, and the resumed wrapper completed successfully.
See `infrastructure-attempts.json`.

## Next experiment

The next rung should broaden calibration substantially while keeping only
layer 30 quantized. It needs a larger train/validation/test prompt bank with
sealed per-task test rows, then the same exact-artifact selection and mandatory
CUTE plus MATH gate. A replicated or larger capability slice must confirm the
five-point CUTE boundary before adding any layer. Whole-model or 27B claims
remain out of scope.

To limit disk growth, reconstructible BF16 export bridges and v2/v3
materialized HF trees were deleted after their hashes and scope receipts were
recorded. The baseline, PTQ control, all CQ3 candidates, patches, and one
checkpoint per run remain on `quant`.
