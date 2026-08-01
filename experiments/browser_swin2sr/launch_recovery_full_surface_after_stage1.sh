#!/bin/zsh
set -euo pipefail

repo_root=/Users/stan/Documents/git/clark-quantize
experiment="$repo_root/experiments/browser_swin2sr"
recovery="$experiment/recovery_stage1_from_factorial24_20260729"
full_surface="$experiment/full_surface_from_recovery_20260730"
development_q4=/Users/stan/.cache/clark-quantize/swin2sr-q4-full/metrics.json
development_bundle=/Users/stan/.cache/clark-quantize/swin2sr-heldout-bundle-v1/bundle.json
final_q4=/Users/stan/.cache/clark-quantize/swin2sr-q4-glyphsafe-v4/metrics.json
final_bundle=/Users/stan/.cache/clark-quantize/swin2sr-heldout-bundle-glyphsafe-v4/bundle.json
training_ui=/Users/stan/.cache/clark-quantize/swin2sr-hf-100k-v1
training_text=/Users/stan/.cache/clark-quantize/swin2sr-flores200-text-102k-glyphsafe-v2
training_text_summary="$training_text/summary.json"
training_text_receipt="$training_text/validation-receipt.json"
heldout_ui=/Users/stan/.cache/clark-quantize/swin2sr-rico-heldout-v1/manifest.jsonl
heldout_text=/Users/stan/.cache/clark-quantize/swin2sr-flores200-devtest-2040-v1/manifest.jsonl
validation=/Users/stan/Documents/git/clark-desktop/promos
base_onnx=/Users/stan/.cache/huggingface/hub/models--Xenova--swin2SR-lightweight-x2-64/snapshots/54ccc3f9ed678912a9c47dc14f6c478855380a2b/onnx/model.onnx
stage1_summary="$recovery/stage1-summary.json"
stage2_summary="$recovery/stage2-summary.json"
stage3_summary="$recovery/stage3-summary.json"
stage2_source_gate="$recovery/stage2-source-ready.json"

cd "$repo_root"
while [[
  ! -f "$stage1_summary"
  || ! -f "$training_text_summary"
  || ! -f "$final_q4"
]]; do
  sleep 30
done

python3 "$experiment/validate_training_corpus.py" \
  --root "$training_text" \
  --expected-images 102000 \
  --expected-languages 204 \
  --expected-per-language 500 \
  --exclude-bundle "$development_bundle" \
  --exclude-bundle "$final_bundle" \
  --out "$training_text_receipt"

while [[ ! -f "$stage2_source_gate" ]]; do
  sleep 30
done

python3 "$experiment/seal_stage2_source.py" \
  --verify "$stage2_source_gate"

base=$(jq -er '.base_checkpoint' "$recovery/stage1-study-manifest.json")

if [[ ! -f "$stage2_summary" ]]; then
  python3 "$experiment/run_recovery_factorial.py" \
    --base-checkpoint "$base" \
    --out "$recovery" \
    --q4-reference "$development_q4" \
    --data "$training_ui" \
    --data "$training_text" \
    --data-receipt "$training_text_receipt" \
    --validation-data "$validation" \
    --eval-manifest "$heldout_ui" \
    --eval-manifest "$heldout_text" \
    --eval-max-samples 800 \
    --stage stage2 \
    --steps 2000 \
    --batch-size 8 \
    --crop-size 128 \
    --learning-rate 1e-6 \
    --scalar-lr-multiplier 100 \
    --localized-stage-steps 100 \
    --validate-every 250 \
    --checkpoint-every 500 \
    --concurrency 2 \
    --device mps \
    --seed 20260729 \
    --promote-from "$stage1_summary" \
    --promote-count 8 \
    --resume-stage stage1 \
    --reset-data-stream
fi

if [[ ! -f "$stage3_summary" ]]; then
  python3 "$experiment/run_recovery_factorial.py" \
    --base-checkpoint "$base" \
    --out "$recovery" \
    --q4-reference "$development_q4" \
    --data "$training_ui" \
    --data "$training_text" \
    --data-receipt "$training_text_receipt" \
    --validation-data "$validation" \
    --eval-manifest "$heldout_ui" \
    --eval-manifest "$heldout_text" \
    --eval-max-samples 3040 \
    --stage stage3 \
    --steps 5000 \
    --batch-size 8 \
    --crop-size 128 \
    --learning-rate 1e-6 \
    --scalar-lr-multiplier 100 \
    --localized-stage-steps 100 \
    --validate-every 500 \
    --checkpoint-every 1000 \
    --concurrency 2 \
    --device mps \
    --seed 20260729 \
    --promote-from "$stage2_summary" \
    --promote-count 3 \
    --resume-stage stage2
fi

winner=$(jq -er '.ranking[0].name' "$stage3_summary")
winner_result="$recovery/stage3/$winner/result.json"
winner_checkpoint="$recovery/stage3/$winner/checkpoint-final.pt"
loss_recipe=$(jq -er '.arm.loss_recipe' "$winner_result")

[[ -f "$winner_checkpoint" ]] || {
  print -u2 "recovery winner checkpoint is missing: $winner_checkpoint"
  exit 2
}

if [[ ! -f "$full_surface/summary.json" ]]; then
  python3 "$experiment/run_full_surface_comparison.py" \
    --recovered-checkpoint "$winner_checkpoint" \
    --out "$full_surface" \
    --q4-reference "$development_q4" \
    --bundle "$development_bundle" \
    --base-onnx "$base_onnx" \
    --data "$training_ui" \
    --data "$training_text" \
    --data-receipt "$training_text_receipt" \
    --validation-data "$validation" \
    --steps 4000 \
    --schedule-steps 3000 \
    --batch-size 8 \
    --crop-size 128 \
    --learning-rate 1e-6 \
    --loss-recipe "$loss_recipe" \
    --validate-every 250 \
    --checkpoint-every 500 \
    --concurrency 2 \
    --device mps \
    --seed 20260729
fi

final_evaluation="$full_surface/final"
python3 "$experiment/evaluate_final_candidate.py" \
  --comparison-summary "$full_surface/summary.json" \
  --final-bundle "$final_bundle" \
  --final-q4 "$final_q4" \
  --out "$final_evaluation"
