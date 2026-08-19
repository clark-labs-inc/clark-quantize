#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v58}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
unit=quant-qwen38-27b-cq3-exact-lut32-v58
runner=experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_cq3_exact_lut32_avx2_v58.py
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-cq3-exact-lut32-avx2-v58}
build_dir=${BUILD_DIR:-/workspace/build/qwen38-cq3-exact-lut32-avx2-v58}

for parent in \
  quant-qwen38-27b-all-layer-mixed-trained-v44.service \
  quant-qwen38-27b-all-layer-sensitivity-residual-v45.service; do
  if systemctl is-active --quiet "$parent"; then
    echo "refusing to overlap v58 with $parent" >&2
    exit 2
  fi
done
if systemctl is-active --quiet "$unit.service"; then
  systemctl show "$unit.service" -p ActiveState -p SubState -p MainPID -p NRestarts
  exit 0
fi
if [[ $(systemctl show "$unit.service" -p LoadState --value 2>/dev/null || true) != not-found ]]; then
  echo "$unit.service already exists but is not active; inspect its terminal receipt" >&2
  exit 2
fi

test -x "$python_bin"
test -f "$repo/$runner"
test -x "$repo/scripts/cpu/build_qwen38_cq3_exact_lut_benchmark_v58.sh"

systemd-run \
  --unit="$unit" \
  --description="Qwen3.8-27B exact CQ3 LUT32 Milan benchmark v58" \
  --property=Type=exec \
  --property=CPUQuota=100% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=MemoryMax=2000000000 \
  --property=MemorySwapMax=0 \
  --property=Restart=no \
  --property=WorkingDirectory="$repo" \
  --setenv=OMP_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  "$python_bin" -u "$runner" \
  --state-dir "$state_dir" \
  --build-dir "$build_dir" \
  --cpu-set 0,2,4,6

systemctl show "$unit.service" \
  -p ActiveState -p SubState -p MainPID -p CPUQuotaPerSecUSec \
  -p MemoryMax -p MemorySwapMax -p AllowedCPUs
