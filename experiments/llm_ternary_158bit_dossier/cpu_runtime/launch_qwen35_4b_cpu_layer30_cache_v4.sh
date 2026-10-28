#!/usr/bin/env bash
set -euo pipefail

unit=${UNIT:-quant-qwen35-cpu-layer30-cache-v4}
repo=${REPO:-/workspace/clark-quantize-cpu-qat}
worker=${WORKER:-$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime/prepare_qwen35_4b_cpu_layer30_qat_v4.sh}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
model_dir=${MODEL_DIR:-/workspace/models/Qwen3.5-4B}
inventory=${INVENTORY:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v1/model-inventory.json}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v4}

if systemctl is-active --quiet "$unit.service"; then
  echo "$unit.service is already active"
  exit 0
fi

systemctl reset-failed "$unit.service" 2>/dev/null || true
systemd-run --unit="$unit" --collect \
  --property=MemoryMax=28G \
  --property=CPUQuota=400% \
  --property=Nice=5 \
  --setenv=REPO="$repo" \
  --setenv=PYTHON_BIN="$python_bin" \
  --setenv=MODEL_DIR="$model_dir" \
  --setenv=INVENTORY="$inventory" \
  --setenv=RUN_DIR="$run_dir" \
  --setenv=PYTHONPATH="$repo" \
  --setenv=PYTHONNOUSERSITE=1 \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 \
  --setenv=TOKENIZERS_PARALLELISM=false \
  "$worker"

systemctl show "$unit.service" \
  -p ActiveState -p SubState -p MemoryMax -p CPUQuotaPerSecUSec \
  -p ExecMainStartTimestamp
