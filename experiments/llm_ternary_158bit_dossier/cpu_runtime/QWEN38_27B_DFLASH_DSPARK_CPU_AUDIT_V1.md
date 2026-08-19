# Qwen3.8-27B DFlash/DSpark CPU audit v1

Date: 2026-08-18 to 2026-08-19

Host: four physical AMD EPYC Milan cores, 32 GiB RAM, no swap

Model scope: Qwen/Qwen3.8-27B only

## Outcome

The current evidence does not support shipping an off-the-shelf “DFlash3” CPU path. The exact Qwen3.8 DSpark checkpoint is the closest current successor to DFlash2, but its Q8_0 CPU configuration was slower than target-only inference. Independent draft repacking also falsified the idea that DFlash2 was draft-layout limited.

The useful positive result is lower in the stack: byte-identical exact Q4_K source compiled with pinned Clang 21 reduced the hot four-row tile from an 800-byte GCC stack frame and 38 vector stack moves to 168 bytes and 14 moves. A preregistered holdout reproduced a 1.153x native-kernel speedup. The actual Qwen3.8-27B four-row graph improved only 1.025x, below its frozen 1.05 gate, so that experiment did not proceed to serving or capability evaluation.

The subsequent full-graph program closed this compiler lane. Active-wait Clang reached 1.0600x in the long-process graph test, but the frozen ABBA server transfer measured only 2.294 conventional tok/s versus 2.288 for the parent, or 1.0023x. Candidate MTP3 measured 4.092 tok/s versus 3.610, but the candidate and parent greedy sequences matched on only 1/4 prompts. That changed sequence raised draft acceptance and confounds the MTP comparison; it is not a clean kernel/runtime speedup. The candidate retained the always-run quick gates—CUTE 9/20, MATH-500 5/8, code trace 5/6, and WikiText perplexity ratio 0.9927—but v35 was correctly rejected before MMLU on its frozen performance gates.

## Successive causal tests

| Experiment | Exact result | Decision |
|---|---:|---|
| Inco DFlash2 PR 27342, target and draft repacked | 2.435 accepted tok/s; 1.287x target; exact greedy parity 2/3; 29.91 GiB peak RSS | Reject the drop-in configuration: below the 1.40x diagnostic floor, one divergent prompt, and above the memory gate. |
| Inco DFlash2 PR 27342, repack disabled | 2.244 accepted tok/s; 1.188x target; exact greedy parity 3/3; 20.97 GiB peak RSS | Retain the exact, memory-lean direction only; speed remains below the qualified MTP control. |
| DFlash2 target raw, draft repacked | 2.193 accepted tok/s; 1.250x its same-runtime target; 0.977x prior fully-raw DFlash2 | Draft repacking is not the recovery mechanism. The earlier fully-repacked gain came from target batch verification. |
| DSpark Q8_0, target raw | 1.579 accepted tok/s; 0.836x target; 2.625 committed tokens/target call | Reject this CPU configuration. Do not generalize to BF16/GPU DSpark. |
| v26 streamed pair2, GCC 15 | 1152-byte frame; 50 vector stack moves | Reject before timing. |
| v27 pair-local scaled, GCC 15 | 800-byte frame; 38 vector stack moves | Reject before timing. |
| v28 row-serial inline, GCC 15 | 1024-byte frame; 61 vector stack moves | Reject before timing. |
| v29 noinline row boundary, GCC 15 | 608 combined frame; one vector stack move; bit-exact; 0.799x native speed | Calls and repeated hot-Q4 loads cost more than the spills. |
| v30 v27 source, Clang 21 | 168-byte frame; 14 vector stack moves; 1.153x native speed | Speed-positive, but terminal because one shape's MAD was 3.28% versus 3.00%. |
| v31 independent holdout | 1.153x geometric native speed; every shape 1.140–1.160x; max MAD 2.824% | Native gate passed. |
| v31 real Qwen3.8-27B graph | 1.025x paired median; 0.558% candidate process MAD; 2.4 GHz flat | Reject at graph gate; skip MTP serving and quality. |
| v32-v33 full-graph profiles | The activation-calibrated Q4 anchor remained Q4_K dominated; the long process exhibited a phase change | Select Q4_K as the only material group, but require runtime-policy isolation before a speed claim. |
| v34 Clang/libomp long-process isolation | Active-wait Clang 1.0600x by the frozen median-of-run-medians statistic | Advance to serving validation only. |
| v35 ABBA conventional serving | Candidate 2.294 tok/s; parent 2.288; 1.0023x; candidate below the 2.35 floor | Reject the Clang serving composition. |
| v35 ABBA MTP3 diagnostic | Candidate 4.092 tok/s; parent 3.610; candidate-parent greedy parity only 1/4 prompts | Sequence-confounded acceptance gain; do not promote as a clean speed result. |
| v35 always-run quick quality | CUTE 9/20; MATH-500 5/8; code 5/6; WikiText ratio 0.9927 | Quality screen passed, but does not rescue the failed performance gate; MMLU correctly skipped. |

All numerical-kernel candidates retained identical Q4/Q8 representations. v29-v31 produced the same FP32 verification-file SHA-256 as the exact parent across 1,024 seeded and five adversarial cases. Backend reference tests passed 3/3.

## What “DFlash3-like” means here

[DSpark](https://arxiv.org/abs/2607.05147) is the relevant current algorithmic successor: it adds Markov and confidence heads to improve speculative scheduling. The official [vLLM DSpark documentation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/) describes the serving path, and the pinned [Qwen3.8-27B DSpark checkpoint](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) is architecture-matched. Those GPU-serving results are not CPU evidence; our exact Q8_0 feasibility run measured the CPU configuration directly and rejected it.

The failed DSpark run does not reject trained confidence scheduling. It shows that on this small Milan node the current draft compute and memory traffic are not repaid by 25.2% accepted proposals. A CPU successor therefore needs both:

1. a substantially cheaper drafter or confidence head; and
2. a target verifier whose entire multi-kernel graph, not one tile, is fast enough to monetize accepted blocks.

## Next high-leverage program

Do not continue with another isolated C++ Q4_K rearrangement or compiler-policy sweep. The v31 native win collapsed from 15.3% to 2.5% at the model graph, and the v34 6.0% graph result collapsed to 0.23% in conventional server decode. v35 also shows why target-vs-draft parity within one build is insufficient: cross-build greedy parity must be measured whenever acceptance can change.

The active high-leverage experiment is representation geometry. The frozen v36 pilot fits a dual-ridge, calibration-span residual to the already trained CQ3/AFF3 candidate on layers 0, 32, and 63; selects one global rank and ridge on an internal 48/16 split; exports INT8 factors; and opens a new holdout only after payload hashes are frozen. A pass admits an exact fused AVX2 skinny residual kernel and layerwise expansion. A failure will be used to measure the residual spectrum and select a different shared activation-subspace or sensitivity mode without holdout tuning.

A future DFlash-like CPU lane must combine an exact memory-lean width-2-to-8 verifier with admission or fallback. It should be revisited only after the representation/kernel program provides an independently reproducible target speedup, so draft acceptance cannot hide a flat target path.

The 100 tok/s objective remains a scale-out/stretch target. This four-core node is an inexpensive kernel and correctness laboratory, not evidence that a 27B model can reach 100 tok/s on four Milan cores.

## Receipt boundary

Compact terminal receipts and preregistrations are version controlled. Raw models, per-sample logs, disassembly, service journals, and large run state remain on `quant` under `/workspace/quant-state/`. No model checkpoint was modified.
