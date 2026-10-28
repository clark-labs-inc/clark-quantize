#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v69}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime/run_qwen38_27b_tensor_nmse_selected_cq3_v69.sh"
unit=quant-qwen38-27b-tensor-nmse-selected-cq3-v69
expected_runner=fad23921b92baa246b0a403eb2caaa0fde9e787fe37b479d88ef63b1d2904d89

if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -x "$runner"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
actual_runner=$(sha256sum "$runner" | awk '{print $1}')
if [[ "$actual_runner" != "$expected_runner" ]]; then
  echo "runner SHA-256 mismatch: expected $expected_runner, got $actual_runner" >&2
  exit 2
fi

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
  /usr/bin/env REPO="$repo" /usr/bin/bash "$runner"
