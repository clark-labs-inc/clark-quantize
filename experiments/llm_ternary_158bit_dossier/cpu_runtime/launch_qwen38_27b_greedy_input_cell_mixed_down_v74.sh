#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v68}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime/run_qwen38_27b_greedy_input_cell_mixed_down_v74.py"
prereg="$runtime/qwen38_27b_greedy_input_cell_mixed_down_preregistration_v74.json"
full_context_receipt="$runtime/receipts/qwen38_27b_full_context_block_cache_v42/receipt.json"
v44_receipt="$runtime/receipts/qwen38_27b_all_layer_mixed_train_v44/terminal-receipt.json"
v73_receipt="$runtime/receipts/qwen38_27b_exact_marginal_tile_mixed_down_v73/receipt.json"
artifact_root=/workspace/quant-runs/qwen38-greedy-input-cell-mixed-down-v74
state=/workspace/quant-state/qwen38-greedy-input-cell-mixed-down-v74
out="$state/receipt.json"
marker="$state/EXPERIMENT_COMPLETE.json"
unit=quant-qwen38-27b-greedy-input-cell-mixed-down-v74
expected_runner=f29a8d8677b43bc09f2faffcc1ec95b95f8cec632f0bc162e1f28ab45ad9244c
expected_prereg=3aafd039391a4396bb97cfca4abf3608fa5e4cc24f4024832036af6377240e03

if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -x "$python_bin" -a -f "$runner" -a -f "$prereg"
test -f "$full_context_receipt" -a -f "$v44_receipt" -a -f "$v73_receipt"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
test "$(sha256sum "$runner" | awk '{print $1}')" = "$expected_runner"
test "$(sha256sum "$prereg" | awk '{print $1}')" = "$expected_prereg"
available=$(df --output=avail -B1 /workspace | tail -1)
if (( available < 8589934592 )); then
  echo "v74 requires at least 8 GiB free disk" >&2
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
    --v73-receipt "$v73_receipt" \
    --artifact-root "$artifact_root" \
    --out "$out" \
    --marker "$marker" \
    --threads 4
