#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-/workspace/agentworld-venv}"
PYTHON="${PYTHON:-python3}"
BACKEND="${BACKEND:-vllm}"
INSTALL_BACKEND="${INSTALL_BACKEND:-1}"
REQUIRED_GPUS="${REQUIRED_GPUS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
echo "detected_gpus=${gpu_count}"
if [ "$gpu_count" -lt "$REQUIRED_GPUS" ]; then
  cat >&2 <<EOF
This node is not full-fidelity capable for Qwen-AgentWorld.
Found ${gpu_count} GPU(s), need ${REQUIRED_GPUS}.

Official Qwen-AgentWorld serving uses tensor parallelism 4 with
context/max-model-len ${MAX_MODEL_LEN}. Use this node only for reduced-context
smokes, packing, and allocator/debug iterations.
EOF
  exit 42
fi

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

if [ "$INSTALL_BACKEND" = "1" ]; then
  case "$BACKEND" in
    vllm)
      "$VENV/bin/python" -m pip install "vllm>=0.8"
      ;;
    sglang)
      "$VENV/bin/python" -m pip install "sglang[all]>=0.4"
      ;;
    none)
      ;;
    *)
      echo "unsupported BACKEND=$BACKEND; expected vllm, sglang, or none" >&2
      exit 2
      ;;
  esac
fi

case "$BACKEND" in
  vllm)
    if [ ! -x "$VENV/bin/vllm" ] && ! command -v vllm >/dev/null 2>&1; then
      echo "vLLM command not found after setup" >&2
      exit 43
    fi
    ;;
  sglang)
    if ! "$VENV/bin/python" -c "import sglang" >/dev/null 2>&1; then
      echo "SGLang import failed after setup" >&2
      exit 43
    fi
    ;;
  none)
    ;;
esac

echo "full-fidelity node ready: backend=${BACKEND} venv=${VENV}"
