# Qwen3.8-27B fused ragged Q4_K AVX2 kernel v2

Status: promoted for the three-row verification path on AMD Milan; not the
current serving-configuration winner

## Result

The candidate decodes each repacked Q4_K nibble and scale block once, then
updates independent Q8_K accumulators for each real activation row. It removes
both failure modes of the preceding attempts:

- cache-order-only v1 retained repeated Q4 decode and improved the width-three
  diagnostic by only 1.031x; and
- padding three real rows to GEMM4 computed a dummy fourth row and measured
  0.998x versus control on the real Qwen graph.

The true fused candidate produced these paired prompt-only graph results on
four physical AMD EPYC Milan cores. Each number is the pooled median of 14
full-Qwen3.8-27B Q4_K_M samples in control, candidate, candidate, control
order.

| target rows | control | fused | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 555.113 ms | 559.007 ms | 0.993x |
| 2 | 656.793 ms | 585.755 ms | 1.121x |
| 3 | 947.509 ms | 819.809 ms | 1.156x |

The width-one change is a 0.70% regression and remains inside the frozen 2%
ordinary-decode limit. The two- and three-row improvements come from sharing
Q4 loads, low/high-nibble extraction, and scale/minimum unpack. Q8 activation
loads, integer dot products, row scaling, and stores remain row-specific.

## End-to-end speculative result

An MTP draft width is not the target row count. Target verification includes
the current sampled token, so:

```text
target verification rows = 1 + MTP draft width
```

This distinction changed the experiment outcome. MTP3 verifies four rows and
routes through llama.cpp's existing GEMM4 path. The fused ragged kernel was
therefore correctly bypassed: candidate MTP3 was 3.0619 accepted tok/s versus
3.0597 control, or 1.0007x. Control and candidate target and MTP3 token IDs
were exact matches across all four 96-token prompts. The frozen MTP3 speed
gate was rejected without treating it as a kernel correctness failure.

MTP2 verifies three rows and exercises the fused kernel. On the same frozen
four-prompt, 96-token-per-prompt corpus:

| configuration | accepted tok/s | paired ratio |
| --- | ---: | ---: |
| control MTP2 | 2.3341 | 1.000x |
| fused MTP2 | 2.6802 | **1.148x** |

Target-only candidate throughput was 0.9971x control. Target-only and MTP2
token IDs matched exactly across runtimes, and MTP2 matched target-only on all
four prompts in both runtimes. Draft proposal and acceptance were identical:
222 of 309 draft tokens accepted, with 2.405 committed tokens per target call.
This isolates the 14.83% end-to-end improvement to the target kernel.

The kernel passes its paired promotion gate. It is not the immediate serving
winner because fused MTP2 reaches only 0.876x the control MTP3 throughput.
MTP3 remains the best measured configuration at 3.0597 accepted tok/s on this
host.

The mandatory CUTE-Quick-20 tripwire also passed at the exact Q4 control score,
9/20. Candidate and control matched on every item ID, prediction, gold answer,
normalized correctness, raw output, and finish reason. This is a frozen
non-leaderboard catastrophic-forgetting check, not a broad quality claim. An
earlier wrapper with an empty health-check port executed zero evaluation
requests and is classified as infrastructure-invalid rather than scored.

## Benchmark validity correction

`test-backend-ops` did not exercise model-load repacking for this experiment.
A one-shot dispatch marker was absent in that synthetic test and present in a
real-model three-token graph. Synthetic timings from that test are excluded;
all speed decisions above use the actual 18.97 GB Qwen3.8-27B GGUF.

## What this establishes

This is evidence that multi-row weight-decode sharing survives the whole model
and speculative server overhead. It is not evidence for 100 tok/s, a new model
quality level, or final-node scaling. At current acceptance, MTP3 still has a
lower-bound 8.189 GB of target-plus-draft weight traffic per accepted token,
which implies 819 GB/s at 100 accepted tok/s before non-weight work.

The next high-value kernel work is:

1. profile the existing four-row GEMM path and its epilogues on an AVX-512/VNNI
   AMD target rather than altering it blindly on the AVX2-only Milan host;
2. reduce or fuse the sequential official-MTP proposal cost, since MTP3 is the
   measured serving winner and does not use the ragged path;
3. port the proven multi-accumulator decode-sharing pattern to the exact final
   host ISA and NUMA sharding; and
4. combine it with a quality-qualified smaller target image, because kernels
   alone cannot remove the present weight-bandwidth lower bound.

Authoritative artifacts are the two source patches, the MTP2 and MTP3
preregistrations, and terminal receipts under `cpu_runtime/receipts`.
