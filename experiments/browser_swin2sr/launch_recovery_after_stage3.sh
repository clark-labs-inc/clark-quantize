#!/bin/zsh
set -euo pipefail

repo_root=/Users/stan/Documents/git/clark-quantize
experiment="$repo_root/experiments/browser_swin2sr"
race="$experiment/factorial_24_stage1_v2_20260729"
recovery="$experiment/recovery_stage1_from_factorial24_20260729"
q4=/Users/stan/.cache/clark-quantize/swin2sr-q4-full/metrics.json
training_ui=/Users/stan/.cache/clark-quantize/swin2sr-hf-100k-v1
training_text=/Users/stan/.cache/clark-quantize/swin2sr-flores200-text-102k-v1
heldout_ui=/Users/stan/.cache/clark-quantize/swin2sr-rico-heldout-v1/manifest.jsonl
heldout_text=/Users/stan/.cache/clark-quantize/swin2sr-flores200-devtest-2040-v1/manifest.jsonl
validation=/Users/stan/Documents/git/clark-desktop/promos
stage3_summary="$race/stage3-summary.json"
source_ranking="$race/stage3-source-aware.json"

cd "$repo_root"
while [[ ! -f "$stage3_summary" ]]; do
  sleep 30
done

python3 "$experiment/rank_factorial_source_aware.py" \
  "$race/stage3" \
  --q4-reference "$q4" \
  --out "$source_ranking"

winner=$(jq -er '.ranking[0].name' "$source_ranking")
base="$race/stage3/$winner/checkpoint-5000.pt"
[[ -f "$base" ]] || {
  print -u2 "source-aware winner checkpoint is missing: $base"
  exit 2
}

python3 "$experiment/run_recovery_factorial.py" \
  --base-checkpoint "$base" \
  --out "$recovery" \
  --q4-reference "$q4" \
  --data "$training_ui" \
  --data "$training_text" \
  --validation-data "$validation" \
  --eval-manifest "$heldout_ui" \
  --eval-manifest "$heldout_text" \
  --eval-max-samples 300 \
  --stage stage1 \
  --steps 500 \
  --batch-size 8 \
  --crop-size 128 \
  --learning-rate 1e-6 \
  --scalar-lr-multiplier 100 \
  --localized-stage-steps 100 \
  --validate-every 100 \
  --checkpoint-every 500 \
  --concurrency 2 \
  --device mps \
  --seed 20260729
