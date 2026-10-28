# Qwen3.8-27B Global Hard-Ternary Pilot Protocol

- Goal ID: `qwen38-27b-trained-ternary-quality`
- Study ID: `qwen38-27b-global-qat-forgetting-v1`
- Experiment ID: `qwen38-27b-a100-hard-global-pilot-v1`
- Run ID: `qwen38-27b-a100-hard-global-pilot-v1-20260816T063712Z`
- Model: `/workspace/models/Qwen3.8-27B`
- Model revision: `1d4bf0e8f5fccd42e6efbc05b30b680ae4aeeef9`
- Accelerator: one non-spot NVIDIA A100 80 GB PCIe (`quant`)
- Remote run root: `/workspace/quant-runs/qwen38-global-pilot-v1`
- Remote receipt root: `/workspace/quant-state/qwen38-global-pilot-v1`

## Hypothesis

A hard-forward, group-128 ternary QAT update over the complete text transformer,
using cached BF16 teacher targets and a state-light factored optimizer, can recover
measurable quality from the catastrophic global PTQ baseline while fitting one
A100. The pilot is successful only if it improves the teacher-distribution
quality measurements and does not introduce a greater than five percentage-point
absolute accuracy loss against BF16 on any frozen capability task.

## Frozen capability gate

- MATH-500 test, revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`,
  32 deterministic subject-stratified rows, zero shot, deterministic answer-only
  generation capped at 96 new tokens.
- GSM8K test, revision `740312add88f781978c0658806c59bc2815b9866`,
  32 deterministic rows, four shot, deterministic generation.
- MMLU test/dev, revision `c30699e8356da336a370243923dbaf21066bb9fe`,
  64 deterministic subject-stratified rows, five shot, exact option likelihood.
- Seed: `20260816`.
- The sealed row manifest and scorer contract must be identical for BF16, current
  base-3 PTQ, and trained hard-ternary arms.
- HumanEval is deferred from this first pilot because executing generated programs
  adds a separate code-sandbox validity boundary; MATH-500 supplies a direct
  reasoning-retention signal without that confound.

The gate is a forgetting measurement, not a claim of benchmark-grade precision.
Small frozen samples provide paired deltas; confidence intervals and every raw
prediction remain in each report. A larger follow-up gate is required before
promotion.

## Training boundary

- Quantize/train the text transformer's rank-2 Linear matrices with hard ternary
  forward weights and continuous BF16 masters.
- Protect visual, multimodal projection, and MTP modules from replacement.
- Protect token embeddings and the untied LM head in this first memory-bounded
  pilot; their high-precision cost is included in the realized bit-rate report.
- Train selected text norms in FP32.
- Optimizer: factored Adafactor with post-accumulate updates, no persistent full
  gradient set, explicit per-layer checkpointing.
- Teacher is the exact BF16 parent. Training examples must not contain any frozen
  evaluation row.
- First launch is a one-step no-save memory smoke. A longer bounded pilot may run
  only after the smoke records peak memory, finite loss, and no OOM.

## Stop rules

Stop without promotion on any of the following:

- CUDA OOM, non-finite loss/gradient/update, or process failure.
- Free `/workspace` disk below 300 GB.
- GPU is no longer exclusively available to this experiment.
- Any mutation of `/workspace/models/Qwen3.8-27B`.
- More than two retained training checkpoints on the A100.
- Mean teacher KL exceeds 10 after the first completed update.
- Capability accuracy falls more than 5 absolute percentage points below BF16
  on MATH-500, GSM8K, or MMLU.

## Artifact and ownership rules

Raw datasets, teacher caches, checkpoints, packed weights, and per-sample outputs
stay on the A100. Only source code, protocol, compact receipts, and summary
reports may return to this checkout. Unrelated client infrastructure is
explicitly out of scope.
The pre-existing `/workspace/quant-runs/qwen38-qat/checkpoints/checkpoint-10` is
preserved, so this experiment may retain at most one additional checkpoint.
