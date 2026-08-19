#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v68}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime/run_qwen38_27b_imatrix_cq3_hybrid_quality_v81.py"
prereg="$runtime/qwen38_27b_imatrix_cq3_hybrid_quality_preregistration_v81.json"
unit=quant-qwen38-27b-imatrix-cq3-hybrid-quality-v81

require_sha256() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

if pgrep -x llama-server >/dev/null || pgrep -x llama-cli >/dev/null || pgrep -x llama-bench >/dev/null || pgrep -x llama-perplexity >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi
if systemctl list-units --type=service --state=running --no-legend | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
require_sha256 "$runner" 9fe768bcff4cf3a46a57594ff1edbdee1e29c176032b21b91835b934a03727bc
require_sha256 "$prereg" e152af61f0526c74684b559db0fb3d4d5f268457874ac0fb4c1b27a9ac02aef6

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
  /workspace/venvs/quant-cpu/bin/python "$runner"
