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
source_jsonl="$runtime_dir/qwen35_4b_cpu_layer30_train_v4.jsonl"
split_manifest="$runtime_dir/qwen35_4b_cpu_layer30_split_v4.json"
quality_policy="$runtime_dir/QUALITY_PREFLIGHT.md"
teacher_cache="$base_run/teacher-completions.pt"
teacher_receipt="$base_run/teacher-completions.pt.json"
layer30_patch="$base_run/layer30-heldout-selected.safetensors"
activation_cache="$run_dir/layer29-activations.pt"
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
if [[ -n "$(git status --short)" ]]; then
  echo "cache preparation requires a clean source checkout" >&2
  exit 2
fi

require_sha256 "$prereg" 7c150b23475029df2e64e4731aa34945e0c2a30cb5d69fc532f377792fd7b5c0
require_sha256 "$source_jsonl" e2a355b62c9e456aa22f0fa609705cf17eb66948d80a6470b49bc1132db40cb8
require_sha256 "$split_manifest" a253a051e7945e099d8e5057e51ff0935434f228b901e1c12b5ee6162ae823f2
require_sha256 "$quality_policy" 4b0eaacd0478da1cd26bc3caf61d211e08f8c0e802b23cd069fe1bf246b9ab69
require_sha256 "$inventory" 793d872a2b75fb51e640afec9472fba2c605e03b610ffae9c5d171d33851f344
require_sha256 "$teacher_cache" 79cc7281cb5d64ac08b636e9ca542346ba0c5f6ed95037b9f05df3c25654a00c
require_sha256 "$teacher_receipt" 463fa52b7f81309e57a7ad3cfe628d06ab660f03990f5cad5a05f04201efaf19
require_sha256 "$layer30_patch" 793cfc84332a6749a07cae2cb72ef16e4bc2b6e50577755b362cb2074f25182e

git rev-parse HEAD > "$run_dir/source.commit"
sha256sum \
  "$prereg" \
  "$source_jsonl" \
  "$split_manifest" \
  "$quality_policy" \
  "$inventory" \
  "$teacher_cache" \
  "$teacher_receipt" \
  "$layer30_patch" \
  "$model_dir/model.safetensors.index.json" \
  "$agent_dir/cache_mlp_activations.py" \
  "$runtime_dir/summarize_qwen_cpu_caches.py" \
  > "$run_dir/cache-inputs.sha256"

if [[ ! -s "$activation_cache" ]]; then
  /usr/bin/time -v env PYTHONPATH="$repo" "$python_bin" "$agent_dir/cache_mlp_activations.py" \
    --model-dir "$model_dir" \
    --cache "$teacher_cache" \
    --out "$activation_cache" \
    --layers 29 \
    --positions 24 \
    --device cpu \
    --torch-threads 4 \
    > "$run_dir/activation-cache.log" 2>&1
fi

PYTHONPATH="$repo" "$python_bin" "$runtime_dir/summarize_qwen_cpu_caches.py" \
  --teacher-cache "$teacher_cache" \
  --activation-cache "$activation_cache" \
  --split-manifest "$split_manifest" \
  --layer 29 \
  --positions-per-row 24 \
  --out "$cache_summary" \
  > "$run_dir/cache-summary.log" 2>&1

"$python_bin" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["valid"] and p["rows"] == 48 and p["topk_alignment_matches"] == p["topk_alignment_total"] == 1029 and p["layer"] == 29' \
  "$cache_summary"

sha256sum "$activation_cache" "$cache_summary" > "$run_dir/cache-artifacts.sha256"
printf 'complete\n' > "$run_dir/CACHE_COMPLETE"
