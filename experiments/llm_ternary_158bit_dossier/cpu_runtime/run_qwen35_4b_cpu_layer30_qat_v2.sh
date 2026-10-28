#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
model_dir=${MODEL_DIR:-/workspace/models/Qwen3.5-4B}
cache_dir=${CACHE_DIR:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v1}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v2}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
agent_dir="$repo/experiments/qwen_agentworld"
source_jsonl="$runtime_dir/qwen35_4b_cpu_layer30_train_v1.jsonl"
prereg="$runtime_dir/qwen35_4b_cpu_layer30_qat_preregistration_v2.json"
inventory="$cache_dir/model-inventory.json"
teacher_cache="$cache_dir/teacher-completions.pt"
activation_cache="$cache_dir/layer30-activations.pt"
local_patch="$run_dir/layer30-local-qat.safetensors"
final_patch="$run_dir/layer30-survival-qat.safetensors"
checkpoint_dir="$run_dir/checkpoints"

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
  echo "completed marker already exists: $run_dir/TRAINING_COMPLETE" >&2
  exit 2
fi

require_sha256 "$source_jsonl" 89542ace1ac7496f82727cc82985b70decb757dff4be9350d25d135cb594a31c
require_sha256 "$teacher_cache" 074209fd094cb83393e64143c799dc616dc570aba56d3ba7819a1e9656206138
require_sha256 "$activation_cache" 9a3daa21df6fe627245cf17917404cae190b43726a58d781bb81e13a08bf4d9e
require_sha256 "$inventory" 793d872a2b75fb51e640afec9472fba2c605e03b610ffae9c5d171d33851f344

sha256sum "$prereg" "$source_jsonl" "$model_dir/model.safetensors.index.json" \
  "$teacher_cache" "$activation_cache" "$inventory" > "$run_dir/inputs.sha256"
git rev-parse HEAD > "$run_dir/source.commit"
git status --short > "$run_dir/source.status"
if [[ -s "$run_dir/source.status" ]]; then
  echo "corrected QAT requires a clean source checkout" >&2
  exit 2
fi

if [[ ! -s "$local_patch" ]]; then
  resume_args=()
  if [[ -s "$checkpoint_dir/layer-30-latest.pt" ]]; then
    resume_args+=(--resume)
  fi
  /usr/bin/time -v "$python_bin" "$agent_dir/train_mlp_triplet_qat.py" \
    --model-dir "$model_dir" \
    --inventory "$inventory" \
    --activation-cache "$activation_cache" \
    --out "$local_patch" \
    --report "$run_dir/layer30-local-qat.json" \
    --layers 30 \
    --mode ternary \
    --group-size 128 \
    --artifact-exact-ternary \
    --steps-per-layer 8 \
    --batch-size 32 \
    --lr 2e-5 \
    --code-weight 0.001 \
    --grad-clip 1 \
    --log-every 1 \
    --seed 20260817 \
    --device cpu \
    --torch-threads 4 \
    --checkpoint-dir "$checkpoint_dir" \
    --checkpoint-every 2 \
    "${resume_args[@]}" \
    > "$run_dir/local-qat.log" 2>&1
fi

if [[ ! -s "$final_patch" ]]; then
  /usr/bin/time -v "$python_bin" "$agent_dir/train_block_survival_qat.py" \
    --model-dir "$model_dir" \
    --cache "$teacher_cache" \
    --base-patch "$local_patch" \
    --out-patch "$final_patch" \
    --out-report "$run_dir/layer30-survival-qat.json" \
    --layers 30 \
    --max-train-tokens 256 \
    --max-train-rows 2 \
    --mode ternary \
    --group-size 128 \
    --artifact-exact-ternary \
    --steps-per-layer 2 \
    --lr 5e-6 \
    --hidden-weight 0.05 \
    --anti-weight 0.1 \
    --code-weight 0.002 \
    --no-train-projections \
    --no-train-norms \
    --device cpu \
    --torch-threads 4 \
    > "$run_dir/survival-qat.log" 2>&1
fi

sha256sum "$local_patch" "$final_patch" "$checkpoint_dir/layer-30-latest.pt" \
  > "$run_dir/artifacts.sha256"
printf 'complete\n' > "$run_dir/TRAINING_COMPLETE"
