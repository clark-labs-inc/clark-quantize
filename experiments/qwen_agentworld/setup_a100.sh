#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-/workspace/agentworld-venv}"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV/bin/python" -m pip install \
  "accelerate>=1.0" \
  "fastapi>=0.115" \
  "huggingface_hub>=0.24" \
  "openai>=1.0" \
  "safetensors>=0.4" \
  "sentencepiece>=0.2" \
  "transformers>=4.57.0" \
  "uvicorn[standard]>=0.30"

echo "agentworld venv ready: $VENV"
