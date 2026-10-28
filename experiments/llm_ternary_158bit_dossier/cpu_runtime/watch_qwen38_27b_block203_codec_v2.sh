#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/clark-quantize-cpu-qat}"
BUNDLE="${BUNDLE:?BUNDLE must name the authenticated staged Git bundle}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}"
PARENT_UNIT="${PARENT_UNIT:-quant-qwen38-27b-block203-codec-watch-v2.service}"
PARENT_RECEIPT="${PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-block203-codec-v1.json}"

while systemctl is-active --quiet "${PARENT_UNIT}"; do
  sleep 30
done

/workspace/venvs/quant-cpu/bin/python - "${PARENT_RECEIPT}" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1]))
if (
    receipt.get("schema") != "clark_qwen38_27b_block203_codec_receipt_v1"
    or receipt.get("promotion", {}).get("passed") is not True
):
    raise SystemExit("the v1 codec parent did not promote")
PY

cd "${REPO}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty research repository" >&2
  exit 73
fi
git bundle verify "${BUNDLE}"
git fetch "${BUNDLE}" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "unexpected deployed commit" >&2
  exit 65
fi
exec experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_block203_codec_v2.sh
