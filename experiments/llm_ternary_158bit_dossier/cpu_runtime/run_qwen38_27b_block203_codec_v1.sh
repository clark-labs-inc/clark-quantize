#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/clark-quantize-cpu-qat}"
PYTHON="${PYTHON:-/workspace/venvs/quant-cpu/bin/python}"
STATE="${STATE:-/workspace/quant-state/qwen38-27b-block203-codec-v1}"
RECEIPT="${RECEIPT:-/workspace/quant-state/qwen38-27b-block203-codec-v1.json}"

mkdir -p "${STATE}"
test -z "$(git -C "${REPO}" status --porcelain)"
"${REPO}/scripts/cpu/build_ternary_cpu.sh" >"${STATE}/native-library.path"

CPU_ID="$(${PYTHON} - <<'PY'
import os
print(min(os.sched_getaffinity(0)))
PY
)"
taskset -c "${CPU_ID}" "${PYTHON}" \
  "${REPO}/experiments/llm_ternary_158bit_dossier/cpu_runtime/benchmark_qwen38_27b_block203_codec_v1.py" \
  --receipt "${RECEIPT}"

"${PYTHON}" - "${RECEIPT}" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1]))
if receipt.get("promotion", {}).get("passed") is not True:
    raise SystemExit("block-203 codec did not pass the preregistered gate")
PY
printf '%s\n' "$(git -C "${REPO}" rev-parse HEAD)" >"${STATE}/source.commit"
printf 'complete\n' >"${STATE}/COMPLETE"
