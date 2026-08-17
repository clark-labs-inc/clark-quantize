# Continuous CPU Tokens-per-Second Side Quest

Status: permanently active alongside model-quality research

Objective: maximize useful `Qwen/Qwen3.8-27B` CPU prompt-prefill and
token-generation throughput without changing accepted model behavior or hiding
storage, memory, startup, or quality costs. Qwen 4B results are historical
references only and receive no new performance work.

## Non-negotiable rule

A faster result is promoted only when its correctness evidence is valid. Pure
kernel experiments must pass their scalar-oracle, thread-determinism, and graph
parity contracts. Once a change reaches a runnable graph or model artifact, it
must pass CUTE-Quick-20, pinned MATH-500 quick, sealed code traces, and
WikiText. Model-changing or large graph changes also run the frozen MMLU slice.
Infrastructure-invalid runs are neither zeroes nor completed scores.

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

1. Establish reproducible end-to-end BF16, direct-CQ3, and quality-retaining
   QAT decode/prefill baselines on x86 AVX2.
2. Profile the real single-token graph and remove the largest wall-time source,
   beginning with CQ3 MLP gate/up/down traffic and DeltaNet state updates.
3. Integrate and measure the promoted aligned-2 AVX2 GEMV compute image in the
   real graph.
4. Add fused gate/up plus SwiGLU and down-projection paths with graph parity.
5. Optimize prompt-prefill independently with unpack-amortizing GEMM tiles.
6. Repeat on ARM DOTPROD, then add AVX-512/VNNI paths when suitable hardware is
   available.

This side quest never authorizes deeper quantization. It supplies the runtime
half of the breakthrough criterion while the quality lane supplies the model
half.
