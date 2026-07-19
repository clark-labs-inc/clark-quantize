# Qwen3.5 Global T2 Fulltrain v2

## Purpose

`qwen35_global_t2_fulltrain_v2` is the follow-up to v1. The v1 run proved that
global all-2D QAT can run on one A100, but it reached hard ternary with behavior
already collapsed: codebook distance was near zero while KL/top1/mass were bad.

The v2 hypothesis is that the hard ternary branch must be trained during the
alpha ramp and residual-stream geometry must be anchored.

## Changes From v1

- Adds hard-shadow loss during alpha ramp:
  - `hard_shadow_weight_start=0.05`
  - `hard_shadow_weight_end=0.25`
  - `hard_shadow_every=2`
- Requires projected teacher hidden labels:
  - layers `0,1,8,12,20,27,30,31`
  - projection dim `128`
  - layer-normalized hidden states times fixed Rademacher projection
- Keeps v1 state-light optimizer:
  - mirror SGD
  - per-group AdaGrad
  - loss-adaptive LR
  - task-family normalization
- Keeps the same target:
  - all 2D tensors ternary g128
  - 1D norms continuous/trainable
  - no adapters, binary, or BF16 islands

## Prepared Files

- `train_all2d_ternary_qat.py`
  - supports memory-safe hard-shadow backward passes
  - supports compact `teacher_hidden_proj` anchors
- `label_teacher_cache_hidden_proj.py`
  - labels existing top-k teacher cache shards with projected hidden anchors
- `build_hiddenproj_cache_v2.sh`
  - builds `/workspace/agentworld_runs/qwen35_multimodal_cache_hiddenproj_20260701`
- `launch_global_t2_fulltrain_v2.sh`
  - refuses to launch unless projected hidden labels are present

## Execution Order

Do not run v2 while v1 is occupying the A100 unless the user explicitly wants to
preempt v1.

1. Build the hidden-projected cache:

   ```bash
   cd /workspace/clark-quantize
   nohup experiments/qwen_agentworld/build_hiddenproj_cache_v2.sh \
     > /workspace/agentworld_runs/qwen35_multimodal_cache_hiddenproj_20260701/build.log 2>&1 &
   ```

2. Launch v2 after cache build succeeds:

   ```bash
   cd /workspace/clark-quantize
   nohup experiments/qwen_agentworld/launch_global_t2_fulltrain_v2.sh \
     > /workspace/agentworld_runs/qwen35_global_t2_fulltrain_v2/nohup.log 2>&1 &
   ```

## Caveat

Phase 3 on-policy survival is not automated in the v2 launcher yet. The launcher
covers the two highest-priority fixes: hard-shadow ramp training and projected
hidden anchoring.
