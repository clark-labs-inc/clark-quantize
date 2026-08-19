#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v68}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime/run_qwen38_27b_imatrix_cq3_hybrid_materialization_v80.py"
prereg="$runtime/qwen38_27b_imatrix_cq3_hybrid_materialization_preregistration_v80.json"
unit=quant-qwen38-27b-imatrix-cq3-hybrid-materialization-v80

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
require_sha256 "$runner" d93b92676c38cb1f32928b717c5907fa043b4542c4e1f33a8606d102f4826bba
require_sha256 "$prereg" dcf9b9a99f37230c3578dcdec3a3059579a2d374f29c82bb3f793803fdb5bbf3

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
