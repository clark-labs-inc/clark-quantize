#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/clark-quantize}
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/quant/bin/python}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3.8-27B}
INPUT=${INPUT:-/workspace/quant-runs/qwen38-global-pilot-v1/teacher_prompts.jsonl}
OUT=${OUT:-/workspace/quant-state/qwen38-global-pilot-v1/teacher-rollout-cache-v1/teacher_cache.pt}
POWER_LIMIT_WATTS=${POWER_LIMIT_WATTS:-250}

restore_power() {
  sudo nvidia-smi --power-limit=300 >/dev/null 2>&1 || true
}
trap restore_power EXIT INT TERM

mkdir -p "$(dirname "${OUT}")"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "refusing to share the GPU with an existing compute process" >&2
  exit 75
fi
available_gib=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if [[ -z "${available_gib}" || "${available_gib}" -lt 300 ]]; then
  echo "less than 300 GiB free on /workspace" >&2
  exit 76
fi

sudo nvidia-smi --power-limit="${POWER_LIMIT_WATTS}" >/dev/null
cd "${REPO_ROOT}"
export PYTHONPATH="experiments/qwen_agentworld:."
"${PYTHON_BIN}" experiments/qwen_agentworld/collect_teacher_rollout_cache.py \
  --model-dir "${MODEL_DIR}" \
  --input "${INPUT}" \
  --out "${OUT}" \
  --max-rows 224 \
  --max-model-len 2048 \
  --generate-tokens 96 \
  --top-k 128 \
  --hidden-layers 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63 \
  --projection-dim 128 \
  --projection-seed 20260701
chmod 0644 "${OUT}" "${OUT}.json"
