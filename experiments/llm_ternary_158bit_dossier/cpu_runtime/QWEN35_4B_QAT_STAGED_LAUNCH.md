# Qwen3.5-4B Staged Global Ternary QAT Launch

Status: ready for an admitted 80 GB GPU; not launched on the 32 GB CPU node.

The full learning-rate and anchor schedule remains a 5B-token experiment. The
first process stops just before 100M tokens, writes an exact atomic checkpoint,
and emits `capability_gates/step-<N>-required.json`. Changing the operational
stop does not change the scientific schedule or the checkpoint run identity.

The launch is frozen in
`qwen35_4b_qat_stage1_preregistration_v1.json`. It reproduces the configuration
that previously reached mean teacher KL 0.425 and 75.1% teacher top-1 agreement,
but those older metrics did not include a generation or capability evaluation.

## Launch

On a clean checkout with sealed data and at least 65 GiB free on an 80 GB-class
GPU:

```bash
sudo systemd-run --unit=qwen35-qat-stage1 \
  --property=WorkingDirectory=/workspace/clark-quantize \
  --property=Restart=no \
  --collect \
  --setenv=DATA_ROOT=/workspace/kd_arms \
  --setenv=OUT_ROOT=/workspace/qat/qwen35-4b-global-v1 \
  --setenv=STOP_AFTER_TOKENS=100000000 \
  /workspace/clark-quantize/scripts/spot/run_qwen35_4b_qat_stage.sh
```

SIGTERM and SIGINT are converted into a checkpoint after the current optimizer
step. An interruption checkpoint has no stage gate marker and may resume
directly. A planned stage checkpoint has a marker and cannot resume without a
passing receipt for that exact checkpoint state hash.

## Quick capability receipt

Freeze a 100-row subject-balanced MMLU manifest once. Run both commands against
the exact model revision and identical manifest; the candidate command uses the
atomic checkpoint path named by the gate request.

```bash
python experiments/large_scale_ternary/prepare_gate_b_manifest.py \
  --tasks mmlu \
  --mmlu-revision <exact-mmlu-commit> \
  --mmlu-limit 100 \
  --out /workspace/evals/qwen35-4b-mmlu-quick.json

CUDA_VISIBLE_DEVICES="" python scripts/eval_ternary_ckpt.py \
  --model Qwen/Qwen3.5-4B \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --arm bf16 \
  --manifest /workspace/evals/qwen35-4b-mmlu-quick.json \
  --device cpu --out /workspace/evals/qwen35-4b-bf16-mmlu.json

CUDA_VISIBLE_DEVICES="" python scripts/eval_ternary_ckpt.py \
  --model Qwen/Qwen3.5-4B \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --arm qat_ternary \
  --ckpt /workspace/qat/qwen35-4b-global-v1/checkpoints/checkpoint-<N> \
  --manifest /workspace/evals/qwen35-4b-mmlu-quick.json \
  --device cpu --out /workspace/evals/qwen35-4b-step-<N>-mmlu.json

python scripts/authorize_qat_continuation.py \
  --gate-request /workspace/qat/qwen35-4b-global-v1/capability_gates/step-<N>-required.json \
  --baseline /workspace/evals/qwen35-4b-bf16-mmlu.json \
  --candidate /workspace/evals/qwen35-4b-step-<N>-mmlu.json \
  --out /workspace/evals/qwen35-4b-step-<N>-authorization.json
```

The authorization passes only when the candidate retains at least 60% of the
BF16 control's correct MMLU items. Zero-correct controls, denominator drift,
manifest drift, model-revision drift, report tampering, and checkpoint-state
hash drift all fail closed. This is a catastrophic-forgetting tripwire, not a
promotion decision.

When a deployable character-generating artifact exists, the default promotion
tripwire remains WikiText perplexity plus the pinned MATH-500 quick slice and
CUTE-Quick-20. A pinned MMLU slice is the documented substitute when character
generation is unavailable.

## Continue only after passing

```bash
sudo systemd-run --unit=qwen35-qat-stage2 \
  --property=WorkingDirectory=/workspace/clark-quantize \
  --property=Restart=no \
  --collect \
  --setenv=DATA_ROOT=/workspace/kd_arms \
  --setenv=OUT_ROOT=/workspace/qat/qwen35-4b-global-v1 \
  --setenv=STOP_AFTER_TOKENS=500000000 \
  --setenv=RESUME_CHECKPOINT=/workspace/qat/qwen35-4b-global-v1/checkpoints/checkpoint-<N> \
  --setenv=CAPABILITY_GATE_RECEIPT=/workspace/evals/qwen35-4b-step-<N>-authorization.json \
  /workspace/clark-quantize/scripts/spot/run_qwen35_4b_qat_stage.sh
```

If the quick gate fails, retain the checkpoint and receipt as a negative result
and do not extend the run.
