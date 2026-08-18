# Primary CPU Tokens-per-Second Program

Status: primary optimization objective; correctness-gated

Objective: achieve at least 100 accepted single-stream output tokens/s for
`Qwen/Qwen3.8-27B` on an AMD CPU without hiding proposal, verification,
sampling, storage, memory, startup, or quality costs. The exact contract is in
`qwen38_27b_amd_100tps_goal_v1.json`. There is no fixed 1.58-bit target. Qwen
4B results are historical references only and receive no new performance work.

## Non-negotiable rule

A faster result is promoted only when its correctness evidence is valid. Pure
kernel experiments must pass their scalar-oracle, thread-determinism, and graph
parity contracts. Once a change reaches a runnable graph or model artifact, it
must pass CUTE-Quick-20, pinned MATH-500 quick, sealed code traces, and
WikiText. Model-changing or large graph changes also run the frozen MMLU slice.
Infrastructure-invalid runs are neither zeroes nor completed scores.

Only target-accepted output tokens count toward the 100 tok/s objective.
Rejected or unverified draft tokens, prompt tokens, microkernel operations,
and tokens from other concurrent requests do not count. Aggregate server
throughput and prefill throughput are secondary metrics and are always
reported separately.

Accuracy and performance arms run alone on the measurement node. QAT, cache
collection, evaluation, and throughput processes never share the four-core
`quant` node during authoritative measurements.

## Permanent scorecard

Every end-to-end receipt binds the exact model, tokenizer, runtime binary,
source commit, command, thread placement, and host. It records:

| Surface | Required lanes |
| --- | --- |
| Load | cold load, warm load, page faults, transcode time |
| Prompt | time to first token and prefill tokens/s at 128, 512, and 2,048 input tokens |
| Decode | median token-generation tokens/s for 128 generated tokens at 1, 2, and 4 physical cores |
| Memory | model bytes, peak RSS, KV/recurrent state, scratch peak |
| Quality | CUTE, MATH-500, code traces, WikiText; MMLU for model/large-graph changes |
| Efficiency | joules/token or package energy when counters are available |

SMT lanes may be exploratory, but the primary `quant` comparison uses the four
physical cores. Report prefill and decode separately; a combined average may
not conceal a regression in either product surface.

## Promotion ladder

1. **Microkernel:** exact pack/decode, registered numerical tolerance, scalar
   oracle, deterministic 1/2/4-thread output, and paired timing.
2. **Graph operator:** exact or tolerance-bounded output against the unfused
   graph on tails and exact Qwen shapes, plus paired process-level timing.
3. **End-to-end runtime:** identical artifact, sampler, prompt, context, and
   thread contract; warmup and alternating paired repetitions; complete quality
   preflight.
4. **Model/format Pareto claim:** include all canonical and compute-image bytes,
   startup transcode, peak RSS, throughput, and quality against BF16 plus a
   strong CPU int4 control.

For ordinary runtime changes, require at least a 5% paired median improvement
on the targeted end-to-end lane, no authoritative pair below 1.0x, and no
correctness difference. Smaller wins may be retained as engineering evidence
but are not promoted. Model/format changes are judged on the full
quality-adjusted memory/speed/energy Pareto frontier rather than speed alone.

## Current priority order

1. Establish reproducible Q4_K_M and Q3_K_M target-only quality, memory,
   prefill, single-token decode, and multi-token verification baselines.
2. Validate the official Qwen3.8 one-layer MTP sidecar on AMD: exact greedy
   parity first, then proposal cost and accepted tokens per target call across
   different workloads.
3. Build a measured end-to-end roofline. A lane stops before expensive scale-up
   if its optimistic bound cannot reach 100 accepted tok/s on the final host.
4. Build an activation-sensitivity ledger and post-train only target tensors
   whose lower precision materially reduces multi-token target verification.
5. Optimize NUMA-local low-bit GEMM for verification widths 2/4/8, fusing
   dequantization, projections, nonlinearities, residuals, and state commit.
6. If recursive MTP is too slow or inaccurate, train a target-conditioned
   one-pass block/tree proposer rather than opening another codec sweep.
7. Run final correctness, capability, and at least 100 accepted tok/s validation
   on a measured high-bandwidth AMD node.

Model precision is a measured allocation variable. Lower precision is useful
only when the complete model stays on the quality-throughput-memory Pareto
frontier.
