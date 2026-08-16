#!/usr/bin/env bash
set -euo pipefail

VENV="${QUANT_VENV:-/workspace/venvs/quant}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
MODEL_REVISION="${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.8-27B}"
REPO="${QUANT_REPO:-/workspace/clark-quantize}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1

source "${VENV}/bin/activate"
mkdir -p "${MODEL_DIR}"
python "${REPO}/experiments/qwen38_27b_spot/download_model.py" \
  --model-id "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --model-dir "${MODEL_DIR}"
