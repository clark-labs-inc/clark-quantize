#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/clark-quantize}
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/quant/bin/python}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3.8-27B}
CACHE=${CACHE:-/workspace/quant-runs/qwen38-global-pilot-v1/hidden_cache/teacher_cache.pt}
PATCH=${PATCH:?PATCH must identify the hard-weight patch to evaluate}
ARM=${ARM:?ARM must identify this evaluation arm}
OUT=${OUT:?OUT must identify the atomic JSON receipt}
INCLUDE_REGEX=${INCLUDE_REGEX:-}
EXCLUDE_REGEX=${EXCLUDE_REGEX:-}

restore_power() {
  sudo nvidia-smi --power-limit=300 >/dev/null 2>&1 || true
}
trap restore_power EXIT INT TERM

if [[ ! -f "${CACHE}" || ! -f "${PATCH}" ]]; then
  echo "required cache or patch is missing" >&2
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

arguments=(
  --model-dir "${MODEL_DIR}"
  --cache "${CACHE}"
  --patch "${PATCH}"
  --arm "${ARM}"
  --out "${OUT}"
  --max-rows-per-task 2
  --max-tokens 256
  --max-new-tokens 48
)
if [[ -n "${INCLUDE_REGEX}" ]]; then
  arguments+=(--include-regex "${INCLUDE_REGEX}")
fi
if [[ -n "${EXCLUDE_REGEX}" ]]; then
  arguments+=(--exclude-regex "${EXCLUDE_REGEX}")
fi
"${PYTHON_BIN}" experiments/llm_ternary_158bit_dossier/qwen38_27b_global_pilot/forensic_checkpoint_ablation.py "${arguments[@]}"
chmod 0644 "${OUT}"
