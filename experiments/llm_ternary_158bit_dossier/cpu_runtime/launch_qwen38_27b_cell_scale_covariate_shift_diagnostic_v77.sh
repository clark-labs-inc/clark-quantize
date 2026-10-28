#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v68}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime/run_qwen38_27b_cell_scale_covariate_shift_diagnostic_v77.py"
prereg="$runtime/qwen38_27b_cell_scale_covariate_shift_diagnostic_preregistration_v77.json"
v76_prereg="$runtime/qwen38_27b_optimal_cell_scale_fresh_replication_preregistration_v76.json"
v76_receipt="$runtime/receipts/qwen38_27b_optimal_cell_scale_fresh_replication_v76/receipt.json"
v76_checkpoint=/workspace/quant-state/qwen38-optimal-cell-scale-fresh-replication-v76/layer59-input-checkpoint.pt
v75_artifact=/workspace/quant-runs/qwen38-calibration-optimal-cell-scale-v75/blk.59.ffn_down.cellmix-cq3-int4-opt-scale
state=/workspace/quant-state/qwen38-cell-scale-covariate-shift-diagnostic-v77
out="$state/receipt.json"
unit=quant-qwen38-27b-cell-scale-covariate-shift-diagnostic-v77
expected_runner=3a14998ceabedd296cf727416f8a24a3825d18a1b7bb7b8c65e34cf711730a6e
expected_prereg=d3804dc1352ee9ab252500434cc2830269911d1c2e84b9d9f094aca98204638e

if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -x "$python_bin" -a -f "$runner" -a -f "$prereg"
test -f "$v76_prereg" -a -f "$v76_receipt" -a -f "$v76_checkpoint"
test -f "$v75_artifact"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
test "$(sha256sum "$runner" | awk '{print $1}')" = "$expected_runner"
test "$(sha256sum "$prereg" | awk '{print $1}')" = "$expected_prereg"
mkdir -p "$state"

systemd-run \
  --unit="$unit" \
  --property=Type=exec \
  --property=CPUQuota=400% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=MemoryMax=30000000000 \
  --property=MemorySwapMax=0 \
  --property=Restart=on-failure \
  --property=RestartSec=10s \
  --property=WorkingDirectory="$repo" \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 \
  --setenv=TOKENIZERS_PARALLELISM=false \
  /usr/bin/taskset -c 0,2,4,6 \
  "$python_bin" -u "$runner" \
    --preregistration "$prereg" \
    --v76-preregistration "$v76_prereg" \
    --v76-receipt "$v76_receipt" \
    --v76-checkpoint "$v76_checkpoint" \
    --v75-artifact "$v75_artifact" \
    --out "$out" \
    --threads 4
