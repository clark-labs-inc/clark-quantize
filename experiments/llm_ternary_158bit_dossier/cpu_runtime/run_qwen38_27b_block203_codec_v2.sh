#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/clark-quantize-cpu-qat}"
PYTHON="${PYTHON:-/workspace/venvs/quant-cpu/bin/python}"
STATE="${STATE:-/workspace/quant-state/qwen38-27b-block203-codec-v2}"
RECEIPT="${RECEIPT:-/workspace/quant-state/qwen38-27b-block203-codec-v2.json}"
PARENT="${PARENT:-/workspace/quant-state/qwen38-27b-block203-codec-v1.json}"
PREREG="${REPO}/experiments/llm_ternary_158bit_dossier/cpu_runtime/qwen38_27b_block203_codec_preregistration_v2.json"

mkdir -p "${STATE}"
test -z "$(git -C "${REPO}" status --porcelain)"
"${PYTHON}" - "${PARENT}" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1]))
if (
    receipt.get("schema") != "clark_qwen38_27b_block203_codec_receipt_v1"
    or receipt.get("promotion", {}).get("passed") is not True
):
    raise SystemExit("the frozen v1 codec baseline is not a promoting receipt")
PY
"${REPO}/scripts/cpu/build_ternary_cpu.sh" >"${STATE}/native-library.path"

CPU_ID="$(${PYTHON} - <<'PY'
import os
print(min(os.sched_getaffinity(0)))
PY
)"
set +e
taskset -c "${CPU_ID}" "${PYTHON}" \
  "${REPO}/experiments/llm_ternary_158bit_dossier/cpu_runtime/benchmark_qwen38_27b_block203_codec_v1.py" \
  --preregistration "${PREREG}" \
  --receipt "${RECEIPT}"
BENCHMARK_STATUS=$?
set -e

OUTCOME="$(${PYTHON} - "${RECEIPT}" "${BENCHMARK_STATUS}" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if receipt.get("schema") != "clark_qwen38_27b_block203_codec_receipt_v2":
    raise SystemExit("invalid v2 codec receipt schema")
passed = receipt.get("promotion", {}).get("passed")
if not isinstance(passed, bool) or status != (0 if passed else 1):
    raise SystemExit("v2 codec process status and terminal receipt disagree")
print("promotion" if passed else "valid_nonpromotion")
PY
)"
printf '%s\n' "$(git -C "${REPO}" rev-parse HEAD)" >"${STATE}/source.commit"
printf '%s\n' "${OUTCOME}" >"${STATE}/OUTCOME"
if [[ "${OUTCOME}" == promotion ]]; then
  printf 'pass\n' >"${STATE}/PROMOTION_PASS"
else
  printf 'valid nonpromotion\n' >"${STATE}/VALID_NONPROMOTION"
fi
printf 'complete\n' >"${STATE}/OBSERVATION_COMPLETE"
