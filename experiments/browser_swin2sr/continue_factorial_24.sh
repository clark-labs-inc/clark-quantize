#!/bin/zsh
set -euo pipefail

repo_root=/Users/stan/Documents/git/clark-quantize
study_root="$repo_root/experiments/browser_swin2sr/factorial_24_stage1_v2_20260729"
runner="$repo_root/experiments/browser_swin2sr/run_factorial_race.py"
training_ui=/Users/stan/.cache/clark-quantize/swin2sr-hf-100k-v1
training_text=/Users/stan/.cache/clark-quantize/swin2sr-flores200-text-102k-v1
heldout_ui=/Users/stan/.cache/clark-quantize/swin2sr-rico-heldout-v1/manifest.jsonl
heldout_text=/Users/stan/.cache/clark-quantize/swin2sr-flores200-devtest-2040-v1/manifest.jsonl
validation_data=/Users/stan/Documents/git/clark-desktop/promos
stage1_summary="$study_root/stage1-summary.json"

cd "$repo_root"

while [[ ! -f "$stage1_summary" ]]; do
  sleep 30
done

# Add a disjoint 200-sample UI/language score to every completed stage-one arm.
python3 "$runner" \
  --out "$study_root" \
  --stage stage1 \
  --data "$training_ui" \
  --data "$training_text" \
  --validation-data "$validation_data" \
  --steps 500 \
  --schedule-steps 500 \
  --batch-size 8 \
  --crop-size 128 \
  --learning-rate 2e-5 \
  --concurrency 2 \
  --device mps \
  --seed 20260729 \
  --validate-every 100 \
  --checkpoint-every 500 \
  --eval-manifest "$heldout_ui" \
  --eval-manifest "$heldout_text" \
  --eval-max-samples 200

# Continue the top eight from their exact step-500 model and optimizer state.
python3 "$runner" \
  --out "$study_root" \
  --stage stage2 \
  --promote-from "$stage1_summary" \
  --promote-count 8 \
  --resume-stage stage1 \
  --resume-step 500 \
  --data "$training_ui" \
  --data "$training_text" \
  --validation-data "$validation_data" \
  --steps 2000 \
  --schedule-steps 500 \
  --batch-size 8 \
  --crop-size 128 \
  --learning-rate 2e-5 \
  --concurrency 2 \
  --device mps \
  --seed 20260729 \
  --validate-every 250 \
  --checkpoint-every 500 \
  --eval-manifest "$heldout_ui" \
  --eval-manifest "$heldout_text" \
  --eval-max-samples 300

# Replicate the top three through 5,000 total steps and a larger held-out slice.
python3 "$runner" \
  --out "$study_root" \
  --stage stage3 \
  --promote-from "$study_root/stage2-summary.json" \
  --promote-count 3 \
  --resume-stage stage2 \
  --resume-step 2000 \
  --data "$training_ui" \
  --data "$training_text" \
  --validation-data "$validation_data" \
  --steps 5000 \
  --schedule-steps 500 \
  --batch-size 8 \
  --crop-size 128 \
  --learning-rate 2e-5 \
  --concurrency 2 \
  --device mps \
  --seed 20260729 \
  --validate-every 500 \
  --checkpoint-every 1000 \
  --eval-manifest "$heldout_ui" \
  --eval-manifest "$heldout_text" \
  --eval-max-samples 800
