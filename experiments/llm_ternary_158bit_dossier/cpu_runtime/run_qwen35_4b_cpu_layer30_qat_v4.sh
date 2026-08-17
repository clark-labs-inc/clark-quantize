#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
model_dir=${MODEL_DIR:-/workspace/models/Qwen3.5-4B}
inventory=${INVENTORY:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v1/model-inventory.json}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v4}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
agent_dir="$repo/experiments/qwen_agentworld"
prereg="$runtime_dir/qwen35_4b_cpu_layer30_qat_preregistration_v4.json"
source_jsonl="$runtime_dir/qwen35_4b_cpu_layer30_train_v4.jsonl"
split_manifest="$runtime_dir/qwen35_4b_cpu_layer30_split_v4.json"
activation_cache="$run_dir/layer30-activations.pt"
cache_summary="$run_dir/cache-summary.json"
selected_patch="$run_dir/layer30-heldout-selected.safetensors"
training_report="$run_dir/layer30-heldout-qat.json"
checkpoint_dir="$run_dir/checkpoints"
train_rows=0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,32,33,34,35,40,41,42,43
validation_rows=4,5,12,13,20,21,28,29,36,37,44,45
test_rows=6,7,14,15,22,23,30,31,38,39,46,47

require_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

mkdir -p "$checkpoint_dir"
cd "$repo"
if [[ -e "$run_dir/TRAINING_COMPLETE" ]]; then
  echo "training completion marker already exists: $run_dir/TRAINING_COMPLETE"
  exit 0
fi

require_sha256 "$prereg" 5c17a5e430a5237806e1b7433c7f804935f436739f21802566b433948efa97dc
require_sha256 "$source_jsonl" e2a355b62c9e456aa22f0fa609705cf17eb66948d80a6470b49bc1132db40cb8
require_sha256 "$split_manifest" a253a051e7945e099d8e5057e51ff0935434f228b901e1c12b5ee6162ae823f2
require_sha256 "$activation_cache" 343cb5e4ce786d4f7b65ee58c6519c83c420fe8c9bb623db0bca0f5ed1a25c10
require_sha256 "$cache_summary" b56889f4a9d3e7d46282972af71ef00ac150e517632bf440e23c4d5ff8534475
require_sha256 "$inventory" 793d872a2b75fb51e640afec9472fba2c605e03b610ffae9c5d171d33851f344

"$python_bin" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["valid"] and p["rows"] == 48 and p["topk_alignment_matches"] == p["topk_alignment_total"] == 1029' \
  "$cache_summary"

if [[ -s "$run_dir/training-source.commit" ]]; then
  receipt_prefix=resume-
else
  receipt_prefix=
fi
git rev-parse HEAD > "$run_dir/${receipt_prefix}training-source.commit"
git status --short > "$run_dir/${receipt_prefix}training-source.status"
if [[ -s "$run_dir/training-source.status" || -s "$run_dir/${receipt_prefix}training-source.status" ]]; then
  echo "held-out QAT requires a clean source checkout" >&2
  exit 2
fi
sha256sum \
  "$prereg" \
  "$source_jsonl" \
  "$split_manifest" \
  "$model_dir/model.safetensors.index.json" \
  "$activation_cache" \
  "$cache_summary" \
  "$inventory" \
  "$agent_dir/train_mlp_triplet_qat.py" \
  "$runtime_dir/eval_mlp_triplet_reconstruction.py" \
  > "$run_dir/${receipt_prefix}training-inputs.sha256"

if [[ ! -s "$selected_patch" || ! -s "$training_report" ]]; then
  resume_args=()
  if [[ -s "$checkpoint_dir/layer-30-latest.pt" ]]; then
    resume_args+=(--resume)
  fi
  /usr/bin/time -v env PYTHONPATH="$repo" "$python_bin" "$agent_dir/train_mlp_triplet_qat.py" \
    --model-dir "$model_dir" \
    --inventory "$inventory" \
    --activation-cache "$activation_cache" \
    --out "$selected_patch" \
    --report "$training_report" \
    --layers 30 \
    --mode ternary \
    --group-size 128 \
    --artifact-exact-ternary \
    --steps-per-layer 64 \
    --batch-size 32 \
    --lr 2e-5 \
    --code-weight 0.001 \
    --grad-clip 1 \
    --log-every 1 \
    --seed 20260817 \
    --device cpu \
    --torch-threads 4 \
    --train-row-indices "$train_rows" \
    --validation-row-indices "$validation_rows" \
    --validation-every 4 \
    --checkpoint-dir "$checkpoint_dir" \
    --checkpoint-every 4 \
    "${resume_args[@]}" \
    > "$run_dir/train.log" 2>&1
fi

PYTHONPATH="$repo" "$python_bin" "$runtime_dir/eval_mlp_triplet_reconstruction.py" \
  --model-dir "$model_dir" \
  --inventory "$inventory" \
  --activation-cache "$activation_cache" \
  --candidate-patch "$selected_patch" \
  --candidate-label heldout-selected-v4 \
  --layer 30 \
  --train-row-indices "$train_rows" \
  --validation-row-indices "$validation_rows" \
  --test-row-indices "$test_rows" \
  --group-size 128 \
  --batch-size 32 \
  --torch-threads 4 \
  --out "$run_dir/reconstruction-eval.json" \
  > "$run_dir/reconstruction-eval.log" 2>&1

sha256sum \
  "$selected_patch" \
  "$training_report" \
  "$checkpoint_dir/layer-30-latest.pt" \
  "$run_dir/reconstruction-eval.json" \
  > "$run_dir/training-artifacts.sha256"
printf 'complete\n' > "$run_dir/TRAINING_COMPLETE"
