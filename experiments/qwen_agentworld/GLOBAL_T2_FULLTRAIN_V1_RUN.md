# qwen35_global_t2_fulltrain_v1

Started: 2026-06-30 16:20 UTC on `a100`.

## Hypothesis

Qwen behavior recovery requires global co-adaptation of all 2D tensors under a
hard ternary forward pass. This run is intended to test the Bonsai/Prism-style
mechanism: the whole 2D trunk trains into a codebook-native basis, rather than
being repaired by local MLP/block patches.

## Policy

- Base: `/workspace/models/Qwen3.5-9B`
- Quantized: all 2D tensors, including embeddings, lm head, attention, linear
  attention, and MLP tensors
- Format: ternary g128
- Continuous: 1D norms
- No binary, mixed-rate policy, BF16 rescue islands, LoRA/adapters, GJ/ABQ
  patching, or external judge/reward model in this run
- Physical storage target after snap/pack: Q2_0-style
  `2 + 16/128 = 2.125 bpw`, about `7.53x` against BF16

## Data Preflight

Launcher preflight passed:

- Cache: `/workspace/agentworld_runs/qwen35_multimodal_cache_20260630`
- Rows: 14,896
- Bad shards: 0
- Task counts: `aya=4864`, `openr1=4352`, `mm_openr1=1975`,
  `cauldron=3705`
- Fixed10 JSONL preflight: 10 rows from
  `/workspace/AgentWorldBench_round8_fixed10_20260629`
- Chat template: present
- `<think>` and `</think>` are tokenizer IDs; `<predicted_observation>` tags
  are not single special tokens
- Projected hidden labels are not present in the current cache, so this launch
  sets hidden weights to `0.0` instead of using the requested hidden geometry
  term

## Optimizer / Curriculum

Trainer changes active in this run:

- `mirror_sgd`
- group-normalized updates
- groupwise AdaGrad accumulator, one scalar per quant group
- loss-adaptive LR, clamped to `0.25x..1.0x`
- task-family loss normalization, clamped to `0.25x..2.0x`
- cache glob interleaving across available shard families
- Cauldron cap per loaded shard:
  - Phase 1: `cauldron:0.10`
  - Phase 2: `cauldron:0.20`

Phase 1:

- 10,000 steps
- `alpha: 0 -> 1`
- max train tokens: 1200
- LR: `5e-6`
- prox: `0`
- format boost: `2.0`
- hidden weight: `0.0` due missing hidden labels

Phase 2:

- 50,000 steps
- `alpha: 1`
- prox: `0.0005`
- same optimizer controls
- hidden weight remains `0.0` unless a hidden-labeled cache is regenerated

Long generation gates are intentionally disabled at launch. They should only
run after short generation is no longer collapsed.

## Initial Health Check

At the first health check:

- Launcher PID: `240989`
- Trainer PID: `241059`
- GPU: A100 80GB, about `75.9GB / 80GB`, `100%` utilization
- Phase 1 started at `2026-06-30T16:20:54+00:00`
- First logged steps were finite; early KL/top1 examples:
  - step 1: `kl=0.0165`, `top1=0.979`, `mass=0.985`
  - step 3: `kl=0.2380`, `top1=0.954`, `mass=0.992`,
    adaptive LR scale reduced to `0.264`

## Relaunch Notes

The first launch reached phase 1 step 268, then crashed with CUDA OOM inside
the mirror optimizer while computing update RMS via a materialized
`update.square().mean()` tensor. Logs were preserved at:

`/workspace/agentworld_runs/qwen35_global_t2_fulltrain_v1_oom_step268_20260630T1628Z`

Fixes applied:

- Compute update RMS through `torch.linalg.vector_norm(update) / sqrt(numel)`
  instead of materializing a squared tensor.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the launcher.

A short relaunch then failed immediately because the optimizer patch used
`math.sqrt` without importing `math`. Logs were preserved at:

`/workspace/agentworld_runs/qwen35_global_t2_fulltrain_v1_math_import_20260630T1644Z`

Fix:

- Added the missing `import math`.

The current replacement launch started at `2026-06-30T16:44:11+00:00`:

- Launcher PID: `241384`
- Trainer PID: `241454`
- GPU at the post-relaunch health check: about `39GB / 80GB`, high utilization
- Last sampled step at that check: phase 1 step 87, still before the old step
  268 OOM boundary

## Known Limitations

This is the first real global run, but it is not yet the full ideal plan:

- Hidden residual geometry is disabled because the existing teacher cache does
  not include projected hidden labels.
- The dataset index currently shows text, reasoning, multimodal image/reasoning,
  and Cauldron rows, but no separate `mm_video` task count.
- The training sampler is shard-interleaved with task caps and family
  normalization, not a fully weighted in-memory 40/30/20/10 sampler.

These are acceptable for v1 because the key test is global all-2D hard-forward
QAT with a state-light optimizer. If v1 shows useful code/sign/scale movement
but weak behavior recovery, the next revision should regenerate a hidden-labeled
cache and add a stricter task-weighted sampler before restarting.
