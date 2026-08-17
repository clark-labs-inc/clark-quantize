#!/usr/bin/env bash
set -euo pipefail

unit=${UNIT:-quant-qwen35-cpu-layer30-qat-v1}
worker=${WORKER:-/workspace/clark-quantize/experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen35_4b_cpu_layer30_qat_v1.sh}

if systemctl is-active --quiet "$unit.service"; then
  echo "$unit.service is already active"
  exit 0
fi

systemctl reset-failed "$unit.service" 2>/dev/null || true
systemd-run --unit="$unit" --collect \
  --property=MemoryMax=28G \
  --property=CPUQuota=400% \
  --property=Nice=5 \
  "$worker"

systemctl show "$unit.service" \
  -p ActiveState -p SubState -p MemoryMax -p CPUQuotaPerSecUSec \
  -p ExecMainStartTimestamp
