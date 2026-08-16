#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/clark-quantize}
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/quant/bin/python}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3.8-27B}
MODEL_REVISION=${MODEL_REVISION:-1d4bf0e8f5fccd42e6efbc05b30b680ae4aeeef9}
MANIFEST=${MANIFEST:-/workspace/quant-state/qwen38-global-pilot-v1/capability_manifest.json}
PATCH=${PATCH:?PATCH must identify the hard-weight patch to evaluate}
OUT=${OUT:?OUT must identify the atomic capability receipt}

restore_power() {
  sudo nvidia-smi --power-limit=300 >/dev/null 2>&1 || true
}
trap restore_power EXIT INT TERM

if [[ ! -f "${MANIFEST}" || ! -f "${PATCH}" ]]; then
  echo "required manifest or patch is missing" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "refusing to share the GPU with an existing compute process" >&2
  exit 75
fi
available_gib=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if [[ -z "${available_gib}" || "${available_gib}" -lt 300 ]]; then
  echo "less than 300 GiB free on /workspace" >&2
  exit 76
fi

sudo nvidia-smi --power-limit=250 >/dev/null
mkdir -p "$(dirname "${OUT}")"
cd "${REPO_ROOT}"
export PYTHONPATH="experiments/qwen_agentworld:."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON_BIN}" experiments/llm_ternary_158bit_dossier/qwen38_27b_global_pilot/evaluate_capabilities.py \
  --model "${MODEL_DIR}" \
  --model-revision "${MODEL_REVISION}" \
  --arm qat_patch \
  --patch "${PATCH}" \
  --manifest "${MANIFEST}" \
  --out "${OUT}" \
  --max-new-tokens 96 \
  --generation-batch-size 4 \
  --decode-workers 4 \
  --max-in-flight 8
chmod 0644 "${OUT}"
