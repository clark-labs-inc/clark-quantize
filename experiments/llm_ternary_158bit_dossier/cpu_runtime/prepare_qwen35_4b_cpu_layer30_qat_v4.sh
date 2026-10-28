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
quality_policy="$runtime_dir/QUALITY_PREFLIGHT.md"
teacher_cache="$run_dir/teacher-completions.pt"
activation_cache="$run_dir/layer30-activations.pt"
split_receipt="$run_dir/split-validation.json"
cache_summary="$run_dir/cache-summary.json"

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

mkdir -p "$run_dir"
cd "$repo"

if [[ -e "$run_dir/CACHE_COMPLETE" ]]; then
  echo "cache completion marker already exists: $run_dir/CACHE_COMPLETE"
  exit 0
fi

require_sha256 "$prereg" 5c17a5e430a5237806e1b7433c7f804935f436739f21802566b433948efa97dc
require_sha256 "$source_jsonl" e2a355b62c9e456aa22f0fa609705cf17eb66948d80a6470b49bc1132db40cb8
require_sha256 "$split_manifest" a253a051e7945e099d8e5057e51ff0935434f228b901e1c12b5ee6162ae823f2
require_sha256 "$quality_policy" 4b0eaacd0478da1cd26bc3caf61d211e08f8c0e802b23cd069fe1bf246b9ab69
require_sha256 "$inventory" 793d872a2b75fb51e640afec9472fba2c605e03b610ffae9c5d171d33851f344

git rev-parse HEAD > "$run_dir/source.commit"
git status --short > "$run_dir/source.status"
if [[ -s "$run_dir/source.status" ]]; then
  echo "cache preparation requires a clean source checkout" >&2
  exit 2
fi
sha256sum \
  "$prereg" \
  "$source_jsonl" \
  "$split_manifest" \
  "$quality_policy" \
  "$inventory" \
  "$model_dir/model.safetensors.index.json" \
  "$agent_dir/collect_teacher_rollout_cache.py" \
  "$agent_dir/cache_mlp_activations.py" \
  "$runtime_dir/validate_qwen_calibration_split.py" \
  "$runtime_dir/summarize_qwen_cpu_caches.py" \
  > "$run_dir/inputs.sha256"

PYTHONPATH="$repo" "$python_bin" "$runtime_dir/validate_qwen_calibration_split.py" \
  --manifest "$split_manifest" \
  --out "$split_receipt" \
  > "$run_dir/split-validation.log" 2>&1

if [[ ! -s "$teacher_cache" || ! -s "$teacher_cache.json" ]]; then
  resume_args=()
  if [[ -s "$teacher_cache.partial" || -s "$teacher_cache" ]]; then
    resume_args+=(--resume)
  fi
  /usr/bin/time -v env PYTHONPATH="$repo" "$python_bin" "$agent_dir/collect_teacher_rollout_cache.py" \
    --model-dir "$model_dir" \
    --input "$source_jsonl" \
    --out "$teacher_cache" \
    --max-rows 48 \
    --max-model-len 256 \
    --generate-tokens 24 \
    --top-k 128 \
    --hidden-layers 30 \
    --projection-dim 64 \
    --seed 20260817 \
    --device cpu \
    --torch-threads 4 \
    --checkpoint-every 2 \
    "${resume_args[@]}" \
    > "$run_dir/teacher-cache.log" 2>&1
fi

if [[ ! -s "$activation_cache" ]]; then
  /usr/bin/time -v env PYTHONPATH="$repo" "$python_bin" "$agent_dir/cache_mlp_activations.py" \
    --model-dir "$model_dir" \
    --cache "$teacher_cache" \
    --out "$activation_cache" \
    --layers 30 \
    --positions 24 \
    --device cpu \
    --torch-threads 4 \
    > "$run_dir/activation-cache.log" 2>&1
fi

PYTHONPATH="$repo" "$python_bin" "$runtime_dir/summarize_qwen_cpu_caches.py" \
  --teacher-cache "$teacher_cache" \
  --activation-cache "$activation_cache" \
  --split-manifest "$split_manifest" \
  --layer 30 \
  --positions-per-row 24 \
  --out "$cache_summary" \
  > "$run_dir/cache-summary.log" 2>&1

"$python_bin" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["valid"] and p["rows"] == 48 and p["topk_alignment_matches"] == p["topk_alignment_total"]' \
  "$cache_summary"

sha256sum \
  "$teacher_cache" \
  "$teacher_cache.json" \
  "$activation_cache" \
  "$split_receipt" \
  "$cache_summary" \
  > "$run_dir/cache-artifacts.sha256"
printf 'complete\n' > "$run_dir/CACHE_COMPLETE"
