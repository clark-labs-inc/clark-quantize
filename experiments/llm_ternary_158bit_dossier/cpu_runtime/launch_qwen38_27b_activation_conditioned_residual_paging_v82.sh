#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v68}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime/run_qwen38_27b_activation_conditioned_residual_paging_v82.py"
prereg="$runtime/qwen38_27b_activation_conditioned_residual_paging_preregistration_v82.json"
receipt="$runtime/receipts/qwen38_27b_full_context_block_cache_v42/receipt.json"
v44="$runtime/receipts/qwen38_27b_all_layer_mixed_train_v44/terminal-receipt.json"
v81="$runtime/receipts/qwen38_27b_imatrix_cq3_hybrid_quality_v81/summary.json"
state=/workspace/quant-state/qwen38-activation-conditioned-residual-paging-v82
unit=quant-qwen38-27b-activation-conditioned-residual-paging-v82

require_sha256() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

if systemctl list-units --type=service --state=running --no-legend | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
require_sha256 "$runner" d90c83645e6902283683de2ad6baeb1ffbf6efca58f93631e38f2459d247a5af
require_sha256 "$prereg" a0863dda656deef0484b7c8ef81d9c63bf8dcebcd774faf30b73fc2da7387285
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
  /workspace/venvs/quant-cpu/bin/python "$runner" \
    --preregistration "$prereg" \
    --full-context-receipt "$receipt" \
    --v44-receipt "$v44" \
    --v81-receipt "$v81" \
    --out "$state/receipt.json" \
    --marker "$state/EXPERIMENT_COMPLETE.json" \
    --threads 4
