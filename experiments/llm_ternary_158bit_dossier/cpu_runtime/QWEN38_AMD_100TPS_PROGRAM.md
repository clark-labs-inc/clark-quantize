# Qwen3.8-27B at 100 accepted tokens/s on AMD CPUs

Status: primary research objective; no result claimed

The target is at least **100 accepted, end-to-end output tokens per second** for
one Qwen3.8-27B sequence on an AMD CPU. The numerator contains only tokens the
target model commits. Draft tokens, prompt tokens, batched requests, and
microkernel work do not count. Aggregate server throughput is measured, but it
cannot substitute for the single-stream result.

## Why this is not an ordinary quantization project

The measured 18.97 GB Q4_K_M artifact would require at least 1.897 TB/s of
weight reads at 100 conventional decode steps/s. Even the rejected 11.39 GB
TQ2_0 artifact requires 1.139 TB/s before recurrent state, activations,
sampling, and imperfect bandwidth utilization are counted. The present
four-core CCX33 is therefore a correctness and ablation host, not a machine on
which the headline result is physically possible.

The program must attack both multiplicative terms:

```text
accepted tokens / second
    = accepted tokens / target call
      ----------------------------------------------
      target verification + proposal + commit time
```

The primary architecture is a quality-qualified compressed target plus a
target-conditioned proposer. Several proposed positions are verified in one
target call, turning repeated bandwidth-bound GEMV into a small multi-token
GEMM. The first proposer is Qwen3.8-27B's official one-layer MTP sidecar. If its
sequential cost or acceptance is insufficient, the next scientific question
is a one-pass block/tree proposer trained from target hidden states—not a
blind sweep over more weight codecs.

## Concrete first experiment

The pinned parent config declares one MTP hidden layer. The official GGUF
repository contains a 1.68 GB Q4_0 MTP sidecar, and the pinned llama.cpp source
already implements `draft-mtp` and batched target verification. This makes a
bounded, high-information experiment possible:

1. prove exact greedy token parity against target-only generation;
2. measure proposal latency, target verification latency at widths 1/2/4/8,
   accepted tokens per target call, rollback cost, and total accepted tok/s;
3. repeat across code, math/reasoning, technical prose, and general prompts;
4. fit a measured roofline for the present node and an AMD EPYC 9R45-class
   dual-socket node;
5. reject the configuration before scale-up if its optimistic final-node bound
   remains below 100 accepted tok/s.

Upstream MTP support is not accepted as correctness evidence. Open upstream
reports describe greedy divergence on quantized targets, so exact token-ID
parity is the first gate. For stochastic sampling, the deployed algorithm must
use target-probability rejection sampling and pass a separate audited trace
contract before any distribution-preservation claim.

The first implementation result is now terminal. The official MTP sidecar at
draft width three reaches 3.0597 accepted tok/s on four physical Milan cores,
accepts 250 of 384 drafts, and commits 2.923 tokens per target call. A new AVX2
Q4_K kernel that shares weight decode across three real target rows improves a
shape-matched MTP2 run from 2.3341 to 2.6802 accepted tok/s (1.148x) with exact
control/candidate token parity. It does not improve MTP3 because three drafts
produce four target rows and use the existing GEMM4 path. MTP3 therefore
remains the local serving winner, while the three-row kernel is promoted as a
real end-to-end optimization. See `QWEN38_Q4_K_FUSED_RAGGED_AVX2_V2.md`.

## Target-model lane

Q3_K_M is the immediate practical baseline. Post-trained aligned two-bit or
mixed 2/3/4/6-bit tensors remain valuable only if they preserve the Q4 quality
gate and materially reduce target verification time. Shared residual
eigenmodes are one subordinate target-compression experiment; their own block
error gate cannot promote them and they are not the primary route to 100
tok/s.

The multi-token target kernel is more important than another single-token
GEMV microbenchmark. It must keep weights NUMA-local, share each weight load
across proposed positions, fuse dequantization and epilogues, and expose the
exact algebra used by post-training. AVX-512/VNNI is the final fast path, with
AVX2 retained as the portable correctness path.

## Final validation

The frozen machine-readable contract is
`qwen38_27b_amd_100tps_goal_v1.json`. A final result must include:

- at least 100 single-stream accepted output tok/s under the frozen corpus;
- target-only and speculative semantic parity receipts;
- CUTE 8/20, MATH-500 4/8, code 4/6, MMLU 50/64, and WikiText perplexity no
  worse than 1.05x the Q4_K_M control;
- exact model, MTP/draft, runtime, command, CPU, NUMA, memory-bandwidth, RSS,
  and energy provenance;
- separate prefill, aggregate throughput, TTFT, proposal, verification,
  acceptance, and rollback measurements.

The first final-node candidate is a 192-physical-core, dual-socket AMD EPYC
9R45-class host. It is rented only after small-node measurements show a
credible 100 tok/s roofline. Neither a projected roofline nor an MTP acceptance
number is itself the result.
