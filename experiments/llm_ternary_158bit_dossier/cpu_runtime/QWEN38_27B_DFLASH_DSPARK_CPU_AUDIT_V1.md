# Qwen3.8-27B DFlash/DSpark CPU audit v1

Date: 2026-08-18 to 2026-08-19

Host: four physical AMD EPYC Milan cores, 32 GiB RAM, no swap

Model scope: Qwen/Qwen3.8-27B only

## Outcome

The current evidence does not support shipping an off-the-shelf “DFlash3” CPU path. The exact Qwen3.8 DSpark checkpoint is the closest current successor to DFlash2, but its Q8_0 CPU configuration was slower than target-only inference. Independent draft repacking also falsified the idea that DFlash2 was draft-layout limited.

The useful breakthrough is lower in the stack: byte-identical exact Q4_K source compiled with pinned Clang 21 reduced the hot four-row tile from an 800-byte GCC stack frame and 38 vector stack moves to 168 bytes and 14 moves. A preregistered holdout reproduced a 1.153x native-kernel speedup. The actual Qwen3.8-27B four-row graph improved only 1.025x, below its frozen 1.05 gate, so serving and capability evaluation were not run.

## Successive causal tests

| Experiment | Exact result | Decision |
|---|---:|---|
| DFlash2 target raw, draft repacked | 2.193 accepted tok/s; 1.250x its same-runtime target; 0.977x prior fully-raw DFlash2 | Draft repacking is not the recovery mechanism. The earlier fully-repacked gain came from target batch verification. |
| DSpark Q8_0, target raw | 1.579 accepted tok/s; 0.836x target; 2.625 committed tokens/target call | Reject this CPU configuration. Do not generalize to BF16/GPU DSpark. |
| v26 streamed pair2, GCC 15 | 1152-byte frame; 50 vector stack moves | Reject before timing. |
| v27 pair-local scaled, GCC 15 | 800-byte frame; 38 vector stack moves | Reject before timing. |
| v28 row-serial inline, GCC 15 | 1024-byte frame; 61 vector stack moves | Reject before timing. |
| v29 noinline row boundary, GCC 15 | 608 combined frame; one vector stack move; bit-exact; 0.799x native speed | Calls and repeated hot-Q4 loads cost more than the spills. |
| v30 v27 source, Clang 21 | 168-byte frame; 14 vector stack moves; 1.153x native speed | Speed-positive, but terminal because one shape's MAD was 3.28% versus 3.00%. |
| v31 independent holdout | 1.153x geometric native speed; every shape 1.140–1.160x; max MAD 2.824% | Native gate passed. |
| v31 real Qwen3.8-27B graph | 1.025x paired median; 0.558% candidate process MAD; 2.4 GHz flat | Reject at graph gate; skip MTP serving and quality. |

All numerical-kernel candidates retained identical Q4/Q8 representations. v29-v31 produced the same FP32 verification-file SHA-256 as the exact parent across 1,024 seeded and five adversarial cases. Backend reference tests passed 3/3.

## What “DFlash3-like” means here

[DSpark](https://arxiv.org/abs/2607.05147) is the relevant current algorithmic successor: it adds Markov and confidence heads to improve speculative scheduling. The official [vLLM DSpark documentation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/) describes the serving path, and the pinned [Qwen3.8-27B DSpark checkpoint](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) is architecture-matched. Those GPU-serving results are not CPU evidence; our exact Q8_0 feasibility run measured the CPU configuration directly and rejected it.

The failed DSpark run does not reject trained confidence scheduling. It shows that on this small Milan node the current draft compute and memory traffic are not repaid by 25.2% accepted proposals. A CPU successor therefore needs both:

1. a substantially cheaper drafter or confidence head; and
2. a target verifier whose entire multi-kernel graph, not one tile, is fast enough to monetize accepted blocks.

## Next high-leverage program

Do not continue with another isolated C++ Q4_K rearrangement. The v31 native win collapsing from 15.3% to 2.5% at the model graph proves that the bottleneck has moved or that compiler-wide changes offset part of the tile gain.

The next experiment should profile the full Clang graph and compose kernels under one frozen gate:

- quantify Q4_K, Q8xQ8 tinyBLAS, Q6_K, OpenMP, attention, and repack shares after the Clang change;
- optimize the top two measured shares together, preserving exact parent outputs;
- only then retest MTP3 serving, where the prior four-core best was 3.612 accepted tok/s with full quick capability retention;
- revisit a trained DSpark/confidence scheduler only after its draft cost has an Amdahl path to a system gain.

The 100 tok/s objective remains a scale-out/stretch target. This four-core node is an inexpensive kernel and correctness laboratory, not evidence that a 27B model can reach 100 tok/s on four Milan cores.

## Receipt boundary

Compact terminal receipts and preregistrations are version controlled. Raw models, per-sample logs, disassembly, service journals, and large run state remain on `quant` under `/workspace/quant-state/`. No model checkpoint was modified.
