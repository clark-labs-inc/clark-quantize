#!/usr/bin/env bash
# Receipt-gated postlude: conditional 100-step QAT, target-1.59 gate, final audit.
set -euo pipefail

REPO="${QUANT_REPO:-/workspace/clark-quantize}"
PYTHON="${QUANT_PYTHON:-/workspace/venvs/quant/bin/python}"
STATE="${QUANT_STATE_ROOT:-/workspace/quant-state}"
RUNS="${QUANT_RUN_ROOT:-/workspace/quant-runs}"
HARNESS="${REPO}/scripts/spot/run_spot_job.sh"
CHECKPOINTS="${RUNS}/qwen38-qat/checkpoints"

test -x "${PYTHON}"
test -x "${HARNESS}"
mkdir -p "${STATE}" "${RUNS}" "${CHECKPOINTS}"

receipt_schema() {
  "${PYTHON}" - "$1" "$2" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = sys.argv[2]
try:
    receipt = json.loads(path.read_text())
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if receipt.get("schema") == expected else 1)
PY
}

receipt_promoted() {
  "${PYTHON}" - "$1" <<'PY'
import json
from pathlib import Path
import sys

try:
    receipt = json.loads(Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if receipt.get("promotion", {}).get("passed") is True else 1)
PY
}

run_scientific_job() {
  local name="$1"
  local receipt="$2"
  local schema="$3"
  shift 3
  if [[ -e "${receipt}" ]]; then
    if receipt_schema "${receipt}" "${schema}"; then
      echo "${name}: terminal receipt already exists; refusing automatic rerun"
      return 0
    fi
    echo "${name}: malformed or wrong-schema receipt requires inspection: ${receipt}" >&2
    return 4
  fi
  set +e
  runuser --user ubuntu -- "${HARNESS}" "${name}" -- "$@"
  local rc=$?
  set -e
  if receipt_schema "${receipt}" "${schema}"; then
    echo "${name}: accepted terminal scientific receipt after exit ${rc}"
    return 0
  fi
  echo "${name}: exited ${rc} without a valid terminal receipt" >&2
  if [[ "${rc}" -eq 0 ]]; then
    return 1
  fi
  return "${rc}"
}

# systemd ordering normally handles this. The loop also protects direct/manual
# invocation and a pipeline restarted while this postlude is queued.
while systemctl is-active --quiet quant-qwen38-pipeline.service; do
  sleep 30
done

# `After=` orders systemd jobs but does not prove that the upstream oneshot
# succeeded.  Fail closed before doing any post-pipeline science if a required
# infrastructure/selection stage is absent or non-promoting.  Behavioral PTQ,
# radix-5 performance, and ten-step QAT receipts are terminal science even on
# non-promotion; the downstream rescue rung only requires their sealed schemas.
for requirement in \
  "${STATE}/qwen38-base3-full.json" \
  "${STATE}/qwen38-retained-export.json" \
  "${STATE}/qwen38-layer-sensitivity.json" \
  "${STATE}/qwen38-qat-teacher-cache.json"
do
  if ! receipt_promoted "${requirement}"; then
    echo "upstream pipeline receipt is absent or non-promoting: ${requirement}" >&2
    exit 4
  fi
done

for requirement in \
  "${STATE}/qwen38-base3-quality.json:clark_qwen38_base3_behavioral_quality_v1" \
  "${STATE}/qwen38-radix5-performance.json:clark_qwen38_radix5_performance_v1"
do
  receipt="${requirement%%:*}"
  schema="${requirement#*:}"
  if ! receipt_schema "${receipt}" "${schema}"; then
    echo "upstream terminal scientific receipt is absent or invalid: ${receipt}" >&2
    exit 4
  fi
done

FIRST_QAT="${STATE}/qwen38-one-layer-qat.json"
QAT100="${STATE}/qwen38-one-layer-qat-100.json"
TARGET158="${STATE}/qwen38-target-158-reconstruction.json"

if receipt_schema "${FIRST_QAT}" "clark_qwen38_one_layer_qat_v1"; then
  if receipt_promoted "${FIRST_QAT}"; then
    run_scientific_job \
      qwen38-one-layer-qat-100 "${QAT100}" clark_qwen38_one_layer_qat_100_v1 \
      "${PYTHON}" "${REPO}/experiments/qwen38_27b_spot/one_layer_qat_100.py"
  else
    echo "ten-step QAT did not promote; the 100-step rung is correctly skipped"
  fi
else
  echo "ten-step QAT receipt is absent or invalid; refusing post-pipeline work" >&2
  exit 4
fi

run_scientific_job \
  qwen38-target-158-reconstruction "${TARGET158}" clark_qwen38_target_158_reconstruction_v1 \
  "${PYTHON}" "${REPO}/experiments/qwen38_27b_spot/target_158_reconstruction_gate.py"

"${PYTHON}" "${REPO}/scripts/spot/prune_qat_checkpoints.py" \
  --root "${CHECKPOINTS}" --keep 2

"${PYTHON}" "${REPO}/experiments/qwen38_27b_spot/audit_completion.py" \
  --out "${STATE}/qwen38-completion-audit.json"
