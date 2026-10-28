#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
model_dir=${MODEL_DIR:-/workspace/models/Qwen3.5-4B}
inventory=${INVENTORY:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v1/model-inventory.json}
base_run=${BASE_RUN:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v4}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen35-4b-cpu-layers29-30-qat-v1}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
agent_dir="$repo/experiments/qwen_agentworld"
prereg="$runtime_dir/qwen35_4b_cpu_layers29_30_qat_preregistration_v1.json"
cache_freeze="$runtime_dir/qwen35_4b_cpu_layers29_30_qat_cache_freeze_v1.json"
source_jsonl="$runtime_dir/qwen35_4b_cpu_layer30_train_v4.jsonl"
split_manifest="$runtime_dir/qwen35_4b_cpu_layer30_split_v4.json"
activation_cache="$run_dir/layer29-activations.pt"
cache_summary="$run_dir/cache-summary.json"
layer30_patch="$base_run/layer30-heldout-selected.safetensors"
layer29_patch="$run_dir/layer29-heldout-selected.safetensors"
combined_patch="$run_dir/layers29-30-heldout-selected.safetensors"
merge_receipt="$run_dir/patch-merge.json"
training_report="$run_dir/layer29-heldout-qat.json"
reconstruction_report="$run_dir/layer29-reconstruction-eval.json"
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
if [[ ! -s "$run_dir/CACHE_COMPLETE" ]]; then
  echo "frozen layer-29 cache is incomplete" >&2
  exit 2
fi
if [[ -n "$(git status --short)" ]]; then
  echo "two-layer QAT requires a clean source checkout" >&2
  exit 2
fi

require_sha256 "$prereg" 7c150b23475029df2e64e4731aa34945e0c2a30cb5d69fc532f377792fd7b5c0
require_sha256 "$cache_freeze" 0d7d6ae6f046259dad13acee6d2aa06f9d7c7aae0ee14c7e2c7d9d947d7753f9
require_sha256 "$source_jsonl" e2a355b62c9e456aa22f0fa609705cf17eb66948d80a6470b49bc1132db40cb8
require_sha256 "$split_manifest" a253a051e7945e099d8e5057e51ff0935434f228b901e1c12b5ee6162ae823f2
require_sha256 "$activation_cache" e4685b8a225cf2aa72a2b0cae60703f0ae5a81706c391d82b237b38535ea340d
require_sha256 "$cache_summary" cd8a9a955707ffc317add217e32cf00fcd991027c1383295e9e2221c2f7f6fd1
require_sha256 "$layer30_patch" 793cfc84332a6749a07cae2cb72ef16e4bc2b6e50577755b362cb2074f25182e
require_sha256 "$inventory" 793d872a2b75fb51e640afec9472fba2c605e03b610ffae9c5d171d33851f344

"$python_bin" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["valid"] and p["rows"] == 48 and p["layer"] == 29 and p["topk_alignment_matches"] == p["topk_alignment_total"] == 1029' \
  "$cache_summary"

git rev-parse HEAD > "$run_dir/training-source.commit"
sha256sum \
  "$prereg" \
  "$cache_freeze" \
  "$source_jsonl" \
  "$split_manifest" \
  "$activation_cache" \
  "$cache_summary" \
  "$layer30_patch" \
  "$inventory" \
  "$model_dir/model.safetensors.index.json" \
  "$agent_dir/train_mlp_triplet_qat.py" \
  "$runtime_dir/merge_safetensors_weight_patches.py" \
  "$runtime_dir/eval_mlp_triplet_reconstruction.py" \
  > "$run_dir/training-inputs.sha256"

if [[ ! -s "$layer29_patch" || ! -s "$training_report" ]]; then
  resume_args=()
  if [[ -s "$checkpoint_dir/layer-29-latest.pt" ]]; then
    resume_args+=(--resume)
  fi
  /usr/bin/time -v env PYTHONPATH="$repo" "$python_bin" "$agent_dir/train_mlp_triplet_qat.py" \
    --model-dir "$model_dir" \
    --inventory "$inventory" \
    --activation-cache "$activation_cache" \
    --base-patch "$layer30_patch" \
    --out "$layer29_patch" \
    --report "$training_report" \
    --layers 29 \
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

if [[ ! -s "$combined_patch" || ! -s "$merge_receipt" ]]; then
  PYTHONPATH="$repo" "$python_bin" "$runtime_dir/merge_safetensors_weight_patches.py" \
    --base "$layer30_patch" \
    --addition "$layer29_patch" \
    --out "$combined_patch" \
    --receipt "$merge_receipt" \
    --expected-tensor-count 6 \
    --required-key-regex 'model[.]language_model[.]layers[.](29|30)[.]mlp[.](gate|up|down)_proj[.]weight' \
    > "$run_dir/patch-merge.log" 2>&1
fi

PYTHONPATH="$repo" "$python_bin" "$runtime_dir/eval_mlp_triplet_reconstruction.py" \
  --model-dir "$model_dir" \
  --inventory "$inventory" \
  --activation-cache "$activation_cache" \
  --candidate-patch "$combined_patch" \
  --candidate-label heldout-selected-layers29-30-v1 \
  --layer 29 \
  --train-row-indices "$train_rows" \
  --validation-row-indices "$validation_rows" \
  --test-row-indices "$test_rows" \
  --group-size 128 \
  --batch-size 32 \
  --torch-threads 4 \
  --out "$reconstruction_report" \
  > "$run_dir/reconstruction-eval.log" 2>&1

"$python_bin" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); t=p["splits"]["test"]; assert p["candidate_exact_ternary"] and t["positions"] > 0 and t["candidate_relative_improvement_vs_direct_ptq_percent"] >= 10.0' \
  "$reconstruction_report"

sha256sum \
  "$layer29_patch" \
  "$layer30_patch" \
  "$combined_patch" \
  "$merge_receipt" \
  "$training_report" \
  "$checkpoint_dir/layer-29-latest.pt" \
  "$reconstruction_report" \
  > "$run_dir/training-artifacts.sha256"
printf 'complete\n' > "$run_dir/TRAINING_COMPLETE"
