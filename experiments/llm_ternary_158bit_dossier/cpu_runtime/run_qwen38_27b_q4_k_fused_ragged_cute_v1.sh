#!/usr/bin/env bash
set -euo pipefail

runtime_dir=${RUNTIME_DIR:-/workspace/clark-quantize-row-scale-v3/experiments/llm_ternary_158bit_dossier/cpu_runtime}
server=${SERVER_BIN:-/workspace/build/llama.cpp-q4k-fused-ragged-avx2-v2/bin/llama-server}
model=${MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q4-k-fused-ragged-cute-v1}
port=${PORT:-19000}
model_sha256=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34

mkdir -p "$state"
if [[ -s "$state/CUTE_COMPLETE" && -s "$state/cute.json" ]]; then
    printf '{"event":"terminal_resumed"}\n'
    exit 0
fi
if pgrep -x llama-server >/dev/null; then
    echo "refusing to overlap an existing llama-server" >&2
    exit 1
fi

cleanup() {
    if [[ -n "${server_pid:-}" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

OMP_NUM_THREADS=4 taskset -c 0,2,4,6 "$server" \
    -m "$model" -t 4 -tb 4 -c 4096 -np 1 -b 512 -ub 512 -ngl 0 \
    --host 127.0.0.1 --port "$port" --no-webui \
    > "$state/server.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "candidate server exited before health check passed" >&2
        exit 1
    fi
    sleep 1
done
if [[ "$ready" -ne 1 ]]; then
    echo "candidate server did not become healthy" >&2
    exit 1
fi

/workspace/venvs/quant-cpu/bin/python "$runtime_dir/eval_llama_cpp_cute_quick.py" \
    --manifest "$runtime_dir/cute_quick_v1.json" \
    --server "http://127.0.0.1:${port}" \
    --model-label qwen38-27b-q4-k-fused-ragged-avx2-v2 \
    --model-sha256 "$model_sha256" \
    --out "$state/cute.json" \
    > "$state/cute.log" 2>&1

printf 'complete\n' > "$state/CUTE_COMPLETE"
