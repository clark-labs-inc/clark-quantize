#!/usr/bin/env bash
set -euo pipefail

model=${1:?model GGUF is required}
label=${2:?short label is required}
port=${3:-8091}
unit="llama-cpu-eval-${label}"
run_dir=${CPU_EVAL_RUN_DIR:-/workspace/quant-runs/cq3-4b-ptq-scope-20260817}
server_bin=${LLAMA_SERVER_BIN:-/workspace/build/llama.cpp/bin/llama-server}

if systemctl is-active --quiet "$unit.service"; then
    echo "$unit.service is already active"
    exit 0
fi

systemctl reset-failed "$unit.service" 2>/dev/null || true
systemd-run --unit="$unit" --collect \
    --property=MemoryMax=28G \
    --property=CPUQuota=800% \
    --property=Nice=5 \
    "$server_bin" \
    -m "$model" -t 4 -tb 4 -c 2048 -b 512 -ub 512 \
    --host 127.0.0.1 --port "$port" --no-warmup \
    --log-file "$run_dir/server-${label}.log"

for _ in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null; then
        echo "$unit.service ready on port $port"
        exit 0
    fi
    if ! systemctl is-active --quiet "$unit.service"; then
        echo "$unit.service exited before becoming healthy" >&2
        systemctl status "$unit.service" --no-pager >&2 || true
        exit 1
    fi
    sleep 1
done

echo "$unit.service did not become healthy within 180 seconds" >&2
exit 1
