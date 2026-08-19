#!/usr/bin/env bash
set -euo pipefail
repo=${REPO:-/workspace/src/clark-quantize-qwen38-v65}
runner=experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_trained_compact_http_canary_v65.sh
if systemctl list-units --type=service --state=running --no-legend | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -x "$repo/$runner"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
/usr/bin/bash "$repo/$runner"
