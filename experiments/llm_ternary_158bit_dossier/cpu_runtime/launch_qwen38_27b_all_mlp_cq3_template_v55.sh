#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v55}
runner=experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_all_mlp_cq3_template_v55.sh
unit=${UNIT:-quant-qwen38-27b-all-mlp-cq3-template-v55}

for parent in \
  quant-qwen38-27b-all-layer-mixed-trained-v44.service \
  quant-qwen38-27b-all-layer-sensitivity-residual-v45.service; do
  if systemctl is-active --quiet "$parent"; then
    echo "refusing to overlap v55 template conversion with $parent" >&2
    exit 2
  fi
done
if systemctl is-active --quiet "$unit.service"; then
  systemctl show "$unit.service" -p ActiveState -p SubState -p MainPID -p NRestarts
  exit 0
fi
if [[ $(systemctl show "$unit.service" -p LoadState --value 2>/dev/null || true) != not-found ]]; then
  echo "$unit.service already exists but is not active; inspect before relaunch" >&2
  exit 2
fi
test -f "$repo/$runner"
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]; then
  echo "v55 launcher requires a clean tracked worktree" >&2
  exit 2
fi
if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a Qwen 4B-era service is active; refusing to share the node" >&2
  exit 2
fi
llama_conflicts=$(ps -eo pid=,comm=,args= | awk \
  '$2 ~ /^llama-(server|cli|bench|perplexi|quantize)/ {print}')
if [[ -n "$llama_conflicts" ]]; then
  echo "a llama.cpp process is active; refusing concurrent conversion" >&2
  exit 2
fi

systemd-run \
  --unit="$unit" \
  --description="Qwen3.8-27B all-MLP CQ3 structural template v55" \
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
  /usr/bin/taskset -c 0,2,4,6 /usr/bin/bash "$repo/$runner"

systemctl show "$unit.service" \
  -p ActiveState -p SubState -p MainPID -p NRestarts \
  -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax -p AllowedCPUs
