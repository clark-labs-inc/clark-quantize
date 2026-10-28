#!/usr/bin/env bash
set -euo pipefail
repo=${REPO:-/workspace/src/clark-quantize-qwen38-v67}
runner=experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_sensitivity_selected_cq3_half_v67.sh
unit=quant-qwen38-27b-sensitivity-selected-cq3-half-v67
if systemctl list-units --type=service --state=running --no-legend | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -x "$repo/$runner"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
systemd-run \
  --unit="$unit" \
  --property=Type=exec \
  --property=CPUQuota=400% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=MemoryMax=30000000000 \
  --property=MemorySwapMax=0 \
  --property=Restart=on-failure \
  --property=RestartPreventExitStatus=2 \
  --property=RestartSec=10s \
  --property=WorkingDirectory="$repo" \
  /usr/bin/bash "$repo/$runner"
