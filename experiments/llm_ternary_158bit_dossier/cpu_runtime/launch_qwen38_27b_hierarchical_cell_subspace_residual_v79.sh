#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v68}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime/run_qwen38_27b_hierarchical_cell_subspace_residual_v79.py"
prereg="$runtime/qwen38_27b_hierarchical_cell_subspace_residual_preregistration_v79.json"
full_context_receipt="$runtime/receipts/qwen38_27b_full_context_block_cache_v42/receipt.json"
v44_receipt="$runtime/receipts/qwen38_27b_all_layer_mixed_train_v44/terminal-receipt.json"
v75_receipt="$runtime/receipts/qwen38_27b_calibration_optimal_cell_scale_v75/receipt.json"
v75_artifact=/workspace/quant-runs/qwen38-calibration-optimal-cell-scale-v75/blk.59.ffn_down.cellmix-cq3-int4-opt-scale
v76_prereg="$runtime/qwen38_27b_optimal_cell_scale_fresh_replication_preregistration_v76.json"
v76_receipt="$runtime/receipts/qwen38_27b_optimal_cell_scale_fresh_replication_v76/receipt.json"
v76_checkpoint=/workspace/quant-state/qwen38-optimal-cell-scale-fresh-replication-v76/layer59-input-checkpoint.pt
v77_receipt="$runtime/receipts/qwen38_27b_cell_scale_covariate_shift_diagnostic_v77/receipt.json"
v78_receipt="$runtime/receipts/qwen38_27b_prompt_robust_cell_support_v78/receipt.json"
v37_terminal="$runtime/receipts/qwen38_27b_sensitivity_gated_residual_v37/terminal-decision.json"
v39_receipt="$runtime/receipts/qwen38_27b_calibration_residual_stability_v39/receipt.json"
v40_receipt="$runtime/receipts/qwen38_27b_calibration_residual_ggml_v40/receipt.json"
artifact_root=/workspace/quant-runs/qwen38-hierarchical-cell-subspace-residual-v79
state=/workspace/quant-state/qwen38-hierarchical-cell-subspace-residual-v79
out="$state/receipt.json"
marker="$state/EXPERIMENT_COMPLETE.json"
unit=quant-qwen38-27b-hierarchical-cell-subspace-residual-v79
expected_runner=436b68e65b4e31ef124b13f7e60add7a7cd248855f98937ac124a17fa9d4514f
expected_prereg=21f12e777d3ee5ffc885495f7c9a54ed4201569a0c257a6acf819d910b31c386

if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -x "$python_bin" -a -f "$runner" -a -f "$prereg"
test -f "$full_context_receipt" -a -f "$v44_receipt" -a -f "$v75_receipt"
test -f "$v75_artifact" -a -f "$v76_prereg" -a -f "$v76_receipt"
test -f "$v76_checkpoint" -a -f "$v77_receipt" -a -f "$v78_receipt"
test -f "$v37_terminal" -a -f "$v39_receipt" -a -f "$v40_receipt"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
test "$(sha256sum "$runner" | awk '{print $1}')" = "$expected_runner"
test "$(sha256sum "$prereg" | awk '{print $1}')" = "$expected_prereg"
available=$(df --output=avail -B1 /workspace | tail -1)
if (( available < 8589934592 )); then
  echo "v79 requires at least 8 GiB free disk" >&2
  exit 2
fi
mkdir -p "$state" "$artifact_root"

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
    --full-context-receipt "$full_context_receipt" \
    --v44-receipt "$v44_receipt" \
    --v75-receipt "$v75_receipt" \
    --v75-artifact "$v75_artifact" \
    --v76-preregistration "$v76_prereg" \
    --v76-receipt "$v76_receipt" \
    --v76-checkpoint "$v76_checkpoint" \
    --v77-receipt "$v77_receipt" \
    --v78-receipt "$v78_receipt" \
    --v37-terminal "$v37_terminal" \
    --v39-receipt "$v39_receipt" \
    --v40-receipt "$v40_receipt" \
    --artifact-root "$artifact_root" \
    --out "$out" \
    --marker "$marker" \
    --threads 4
