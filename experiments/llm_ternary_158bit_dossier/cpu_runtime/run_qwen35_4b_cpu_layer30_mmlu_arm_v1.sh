#!/usr/bin/env bash
set -euo pipefail

model=${1:?model GGUF is required}
model_label=${2:?model label is required}
expected_sha256=${3:?expected model SHA-256 is required}
port=${4:?server port is required}
repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen35-4b-cpu-layer30-mmlu-v1}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
manifest="$runtime_dir/qwen35_4b_cpu_layer30_mmlu_v1.json"
prereg="$runtime_dir/qwen35_4b_cpu_layer30_mmlu_preregistration_v1.json"
evaluator="$runtime_dir/eval_llama_cpp_mmlu.py"
launcher="$runtime_dir/launch_llama_cpu_eval_server.sh"
out="$run_dir/mmlu-${model_label}.json"
marker="$run_dir/MMLU_${model_label}_COMPLETE"
server_unit="llama-cpu-eval-${model_label}.service"

if [[ ! "$model_label" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "model label contains unsupported characters" >&2
    exit 2
fi

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
if [[ -e "$marker" ]]; then
    echo "completion marker already exists: $marker"
    exit 0
fi
if [[ -n "$(git status --short)" ]]; then
    echo "MMLU evaluation requires a clean source checkout" >&2
    exit 2
fi

require_sha256 "$manifest" e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba
require_sha256 "$model" "$expected_sha256"
git rev-parse HEAD > "$run_dir/${model_label}-source.commit"
git -C /workspace/src/llama.cpp rev-parse HEAD > "$run_dir/${model_label}-llama.commit"
sha256sum "$prereg" "$manifest" "$evaluator" "$launcher" "$model" \
    /workspace/build/llama.cpp/bin/llama-server \
    > "$run_dir/${model_label}-inputs.sha256"

cleanup() {
    systemctl stop "$server_unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT

CPU_EVAL_RUN_DIR="$run_dir" \
LLAMA_PARALLEL=4 \
LLAMA_CONTEXT=8192 \
LLAMA_THREADS=4 \
    "$launcher" "$model" "$model_label" "$port"

if [[ ! -s "$out" ]]; then
    resume_args=()
    if [[ -s "$out.partial" ]]; then
        resume_args+=(--resume)
    fi
    PYTHONPATH="$repo" "$python_bin" "$evaluator" \
        --manifest "$manifest" \
        --model "$model" \
        --model-label "$model_label" \
        --model-sha256 "$expected_sha256" \
        --server "http://127.0.0.1:${port}" \
        --workers 4 \
        --checkpoint-every 4 \
        --out "$out" \
        "${resume_args[@]}" \
        > "$run_dir/${model_label}.log" 2>&1
fi

"$python_bin" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["valid"] and p["status"] == "completed" and p["summary"]["n"] == 64 and p["contract"]["model_sha256"] == sys.argv[2]' \
    "$out" "$expected_sha256"
sha256sum "$out" > "$run_dir/${model_label}-artifacts.sha256"
printf 'complete\n' > "$marker"
