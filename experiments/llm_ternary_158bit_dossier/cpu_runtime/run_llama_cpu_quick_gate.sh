#!/usr/bin/env bash
set -euo pipefail

model=${1:?candidate GGUF is required}
label=${2:?short candidate label is required}
baseline_math=${3:?frozen BF16 math report is required}
baseline_cute=${4:?frozen BF16 CUTE report is required}
out_dir=${5:?output directory is required}
port=${6:-8091}

if [[ ! "$label" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "label must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
fi
for required in "$model" "$baseline_math" "$baseline_cute"; do
    if [[ ! -f "$required" ]]; then
        echo "required input not found: $required" >&2
        exit 2
    fi
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
model=$(realpath "$model")
baseline_math=$(realpath "$baseline_math")
baseline_cute=$(realpath "$baseline_cute")
out_dir=$(realpath -m "$out_dir")
mkdir -p "$out_dir"

model_sha256=$(sha256sum "$model" | awk '{print $1}')
server_unit="llama-cpu-eval-${label}.service"

cleanup() {
    systemctl stop "$server_unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT

CPU_EVAL_RUN_DIR="$out_dir" \
    "$script_dir/launch_llama_cpu_eval_server.sh" "$model" "$label" "$port"

python3 "$script_dir/eval_llama_cpp_cute_quick.py" \
    --manifest "$script_dir/cute_quick_v1.json" \
    --server "http://127.0.0.1:${port}" \
    --model-label "$label" \
    --model-sha256 "$model_sha256" \
    --out "$out_dir/cute-${label}.json"

python3 "$script_dir/eval_llama_cpp_math_code_canary.py" \
    --manifest "$script_dir/math_code_canary_v1.json" \
    --server "http://127.0.0.1:${port}" \
    --model-label "$label" \
    --model-sha256 "$model_sha256" \
    --out "$out_dir/math-code-${label}.json"

python3 "$script_dir/compare_cpu_capability_tripwire.py" \
    --baseline-math "$baseline_math" \
    --candidate-math "$out_dir/math-code-${label}.json" \
    --baseline-cute "$baseline_cute" \
    --candidate-cute "$out_dir/cute-${label}.json" \
    --candidate-label "$label" \
    --out "$out_dir/gate-${label}.json"

