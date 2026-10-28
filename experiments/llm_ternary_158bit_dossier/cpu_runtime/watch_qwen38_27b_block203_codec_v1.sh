#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/clark-quantize-cpu-qat}"
BUNDLE="${BUNDLE:?BUNDLE must name the authenticated staged Git bundle}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}"

WAIT_UNITS=(
  quant-qwen38-27b-cpu-q4-baseline-v1.service
  quant-qwen38-27b-postbaseline-chain-v1.service
  quant-qwen38-27b-paired-throughput-watch-v1.service
  quant-qwen38-27b-cq3-x86-pair-watch-v1.service
  quant-qwen38-27b-scale-codebook4-watch-v1.service
)

while :; do
  active=false
  for unit in "${WAIT_UNITS[@]}"; do
    if systemctl is-active --quiet "${unit}"; then
      active=true
      break
    fi
  done
  if [[ "${active}" == false ]]; then
    break
  fi
  sleep 30
done

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
exec experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_block203_codec_v1.sh
