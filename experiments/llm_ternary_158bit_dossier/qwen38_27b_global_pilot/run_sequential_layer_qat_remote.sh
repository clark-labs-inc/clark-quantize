#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/clark-quantize}
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/quant/bin/python}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3.8-27B}
CACHE=${CACHE:-/workspace/quant-runs/qwen38-global-pilot-v1/hidden_cache/teacher_cache.pt}
RUN_DIR=${RUN_DIR:?RUN_DIR must name an isolated output directory}
LAYERS=${LAYERS:-0}
STEPS_PER_LAYER=${STEPS_PER_LAYER:-1}
MAX_ROWS_PER_TASK=${MAX_ROWS_PER_TASK:-1}
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-256}
POWER_LIMIT_WATTS=${POWER_LIMIT_WATTS:-250}
BASE_PATCH=${BASE_PATCH:-}

restore_power() {
  sudo nvidia-smi --power-limit=300 >/dev/null 2>&1 || true
}
trap restore_power EXIT INT TERM

mkdir -p "${RUN_DIR}"
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

arguments=(
  --model-dir "${MODEL_DIR}" \
  --cache "${CACHE}" \
  --out-patch "${RUN_DIR}/layer-patch.safetensors" \
  --out-report "${RUN_DIR}/train-report.json" \
  --layers "${LAYERS}" \
  --max-train-tokens "${MAX_TRAIN_TOKENS}" \
  --max-rows-per-task "${MAX_ROWS_PER_TASK}" \
  --shuffle-rows \
  --interleave-tasks \
  --row-seed 20260816 \
  --steps-per-layer "${STEPS_PER_LAYER}" \
  --group-size 128 \
  --lr 5e-6 \
  --hidden-weight 1.0 \
  --anti-weight 0.1 \
  --code-weight 0.001 \
  --no-train-norms \
  --log-every 4
)
if [[ -n "${BASE_PATCH}" ]]; then
  arguments+=(--base-patch "${BASE_PATCH}" --include-base-in-output)
fi
"${PYTHON_BIN}" experiments/qwen_agentworld/train_block_survival_qat.py "${arguments[@]}"
chmod 0644 "${RUN_DIR}/layer-patch.safetensors" "${RUN_DIR}/train-report.json"
