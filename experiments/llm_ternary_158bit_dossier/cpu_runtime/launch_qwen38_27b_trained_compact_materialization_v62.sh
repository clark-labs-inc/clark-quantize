#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v62}
runner=experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_trained_compact_materialization_v62.sh
unit=quant-qwen38-27b-trained-compact-materialize-v62

if systemctl is-active --quiet "$unit.service"; then
  systemctl show "$unit.service" -p ActiveState -p SubState -p MainPID -p NRestarts
  exit 0
fi
if [[ $(systemctl show "$unit.service" -p LoadState --value 2>/dev/null || true) != not-found ]]; then
  echo "$unit.service already exists but is not active; inspect its terminal state" >&2
  exit 2
fi
if systemctl list-units --type=service --state=running --no-legend | grep -Eq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -x "$repo/$runner"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"

systemd-run \
  --unit="$unit" \
  --description="Qwen3.8-27B trained CQ3 AFF3 residual-free materialization v62" \
  --property=Type=exec \
  --property=CPUQuota=400% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=MemoryMax=20000000000 \
  --property=MemorySwapMax=0 \
  --property=Restart=on-failure \
  --property=RestartPreventExitStatus=2 \
  --property=RestartSec=10s \
  --property=StartLimitIntervalSec=6h \
  --property=StartLimitBurst=3 \
  --property=WorkingDirectory="$repo" \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 \
  /usr/bin/taskset -c 0,2,4,6 /usr/bin/bash "$repo/$runner"

systemctl show "$unit.service" \
  -p ActiveState -p SubState -p MainPID -p NRestarts \
  -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax -p AllowedCPUs
