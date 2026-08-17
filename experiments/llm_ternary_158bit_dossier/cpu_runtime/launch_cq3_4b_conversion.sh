#!/usr/bin/env bash
set -euo pipefail

unit=${CQ3_UNIT:-quant-cq3-4b-ptq-scope-20260817}
worker=${CQ3_WORKER:-/workspace/clark-quantize/experiments/llm_ternary_158bit_dossier/cpu_runtime/run_cq3_4b_conversion_worker.sh}
input=${CQ3_INPUT:-/workspace/models/controls/Qwen3.5-4B-BF16.gguf}
output=${CQ3_OUTPUT:-/workspace/models/controls/Qwen3.5-4B-CQ3_0-QATScope.gguf}
run_dir=${CQ3_RUN_DIR:-/workspace/quant-runs/cq3-4b-ptq-scope-20260817}

if systemctl is-active --quiet "$unit.service"; then
    echo "$unit.service is already active"
    exit 0
fi

if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 1
fi

systemctl reset-failed "$unit.service" 2>/dev/null || true
systemd-run --unit="$unit" --collect \
    --property=MemoryMax=28G \
    --property=CPUQuota=800% \
    --property=Nice=5 \
    "$worker" "$input" "$output" "$run_dir"

systemctl show "$unit.service" \
    -p ActiveState -p SubState -p MemoryMax -p CPUQuotaPerSecUSec \
    -p ExecMainStartTimestamp
