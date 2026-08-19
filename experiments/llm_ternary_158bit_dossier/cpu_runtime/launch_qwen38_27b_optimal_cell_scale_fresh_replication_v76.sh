#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v68}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime/run_qwen38_27b_optimal_cell_scale_fresh_replication_v76.py"
prereg="$runtime/qwen38_27b_optimal_cell_scale_fresh_replication_preregistration_v76.json"
v42_receipt="$runtime/receipts/qwen38_27b_full_context_block_cache_v42/receipt.json"
v42_context=/workspace/quant-runs/qwen38-full-context-block-cache-v42/context.pt
v44_receipt="$runtime/receipts/qwen38_27b_all_layer_mixed_train_v44/terminal-receipt.json"
v75_receipt="$runtime/receipts/qwen38_27b_calibration_optimal_cell_scale_v75/receipt.json"
v75_artifact=/workspace/quant-runs/qwen38-calibration-optimal-cell-scale-v75/blk.59.ffn_down.cellmix-cq3-int4-opt-scale
state=/workspace/quant-state/qwen38-optimal-cell-scale-fresh-replication-v76
checkpoint="$state/layer59-input-checkpoint.pt"
out="$state/receipt.json"
marker="$state/EXPERIMENT_COMPLETE.json"
unit=quant-qwen38-27b-optimal-cell-scale-fresh-replication-v76
expected_runner=586ea9344b4782dba003fd1e82c0540c0a7e3c7317c4ab9f3f4534e1eae43ca1
expected_prereg=118f4a48201df6eae5973334f894fd6f9cf24eb093a4b5b97ebba79cb962e544

if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -x "$python_bin" -a -f "$runner" -a -f "$prereg"
test -f "$v42_receipt" -a -f "$v42_context" -a -f "$v44_receipt"
test -f "$v75_receipt" -a -f "$v75_artifact"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
test "$(sha256sum "$runner" | awk '{print $1}')" = "$expected_runner"
test "$(sha256sum "$prereg" | awk '{print $1}')" = "$expected_prereg"
available=$(df --output=avail -B1 /workspace | tail -1)
if (( available < 8589934592 )); then
  echo "v76 requires at least 8 GiB free disk" >&2
  exit 2
fi
mkdir -p "$state"

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
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 \
  --setenv=TOKENIZERS_PARALLELISM=false \
  /usr/bin/taskset -c 0,2,4,6 \
  "$python_bin" -u "$runner" \
    --preregistration "$prereg" \
    --v42-receipt "$v42_receipt" \
    --v42-context "$v42_context" \
    --v44-receipt "$v44_receipt" \
    --v75-receipt "$v75_receipt" \
    --v75-artifact "$v75_artifact" \
    --checkpoint "$checkpoint" \
    --out "$out" \
    --marker "$marker" \
    --threads 4
