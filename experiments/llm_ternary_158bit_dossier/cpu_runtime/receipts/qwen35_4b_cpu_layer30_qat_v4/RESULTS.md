# Qwen3.5-4B Broader-Calibrated Layer-30 CPU QAT V4

Status: completed positive one-layer replication

Run date: 2026-08-17 UTC

Decision: the candidate passed every preregistered v4 gate; do not expand
quantized depth until a distinct, frozen capability replication such as a
subject-balanced MMLU slice also passes.

## Outcome

V4 replicated the v3 one-layer recovery on a four-times-larger, task-balanced
calibration bank. The exact hard-ternary patch improved normalized
reconstruction by 27.57% relative to direct PTQ on 231 independent activation
positions used by neither AdamW nor artifact selection. It scored 7/20 on
CUTE-Quick, 5/8 on pinned MATH-500, and 4/6 on sealed code traces. WikiText
PPL was 8.6729, below the preregistered direct-PTQ ceiling of 8.6975.

The machine decision is
`replicated_one_layer_gate_pass_no_depth_authorization`. This is evidence that
the held-out exact-artifact selection method survives broader calibration. It
is not whole-model ternary training, a Qwen 27B result, or a breakthrough.

## Frozen cache and split

The source contains 48 unique prompts: eight each for knowledge, math, code,
instruction following, multilingual behavior, and reasoning. Exact prompt
overlap with the frozen CUTE and MATH/code manifests is empty.

| Split | Rows | Actual positions | Used for |
| --- | ---: | ---: | --- |
| Train | 24 | 532 | AdamW only |
| Validation selection | 12 | 266 | exact hard-artifact selection only |
| Independent test | 12 | 231 | final reconstruction receipt only |

Teacher collection produced 1,029 completion targets. Greedy alignment was
1,019/1,029 and causal top-128 coverage was 1,029/1,029. The collector wrote a
contract-checked partial checkpoint every two rows and removed it only after
the final cache and JSON receipt were atomically finalized.

## Reconstruction

The final exact artifact at step 64 was selected by minimum validation loss.

| Split | Direct PTQ | V4 | Relative improvement |
| --- | ---: | ---: | ---: |
| Train | 0.368619 | 0.117831 | 68.03% |
| Validation selection | 0.359098 | 0.269447 | 24.97% |
| Independent test | 0.369388 | 0.267556 | 27.57% |

## Mandatory exact-GGUF quality gate

| Frozen signal | BF16 | Direct PTQ | V3 | V4 | V4 gate |
| --- | ---: | ---: | ---: | ---: | --- |
| CUTE-Quick-20 | 8/20 | 6/20 | 7/20 | 7/20 | pass boundary |
| MATH-500 quick | 5/8 | 5/8 | 5/8 | 5/8 | pass/tie |
| Sealed code traces | 4/6 | 4/6 | 4/6 | 4/6 | pass/tie |
| WikiText-2, four 512-token chunks | 8.4960 | 8.6975 | 8.6523 | 8.6729 | pass |

V4 and v3 have identical correctness on all 34 CUTE/MATH/code items. Their
exact outputs match on 19/20 CUTE items and 14/14 MATH/code items. Against
direct PTQ, v4 has one exclusive CUTE win and no exclusive loss. Against BF16,
it has no exclusive win and one exclusive CUTE loss. V4 PPL is 0.0246 below
direct PTQ but 0.0206 above v3; these deltas are small relative to the reported
quick-PPL uncertainty and are not claimed as likelihood improvements.

## Exact artifacts and scope

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| V4 exact CQ3 GGUF | 8,542,863,168 | `d6cbc5bf0e9804e49d26bd40ddb9972db40237fdb4f534545419eea56da02ebb` |
| V4 selected BF16 patch | 141,558,144 | `793cfc84332a6749a07cae2cb72ef16e4bc2b6e50577755b362cb2074f25182e` |
| Single retained restart checkpoint | 990,919,547 | `650e83e932e312b5a087aee0a6255bbbd2387983effde5c52e431dbbfaa47853` |

Only `blk.30.ffn_gate.weight`, `blk.30.ffn_up.weight`, and
`blk.30.ffn_down.weight` are CQ3. The scope receipt proves all 438 non-target
tensors are byte-identical across BF16, direct PTQ, and v4. The three candidate
payloads differ from direct PTQ, proving this is not a relabeled control.

## CPU and restart evidence

The cache, training, export, quick generation, and WikiText services all exited
zero under explicit systemd memory and CPU limits. Cache preparation took
28m35s wall time at a 7.4G peak; training plus sealed reconstruction took
6m05s at 3.3G; export and full tensor hashing took 2m59s at 22.6G; quick
generation took 3m21s; WikiText took 1m11s at 3G. Neither cache collection nor
training swapped.

After receipts were copied, the reconstructible HF materialization and BF16
bridge were removed. The operation reclaimed 13,995,016,192 filesystem bytes.
The CQ3 candidate, selected patch, one checkpoint, teacher/activation caches,
and all receipts remain on `quant`.

## Next gate

Freeze a subject-balanced MMLU slice and run the same BF16/direct-PTQ/v4 exact
artifact comparison. No additional layer should be quantized unless that
independent capability replication passes. The 27B phase remains gated behind
both replicated model quality and a whole-model CPU memory/runtime plan.
