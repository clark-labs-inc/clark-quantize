#!/usr/bin/env bash
set -euo pipefail

CLARK_DIR="${CLARK_DIR:-/workspace/clark-quantize}"
ACTIVE_PID="${ACTIVE_PID:?set ACTIVE_PID to the currently running training PID}"
OLD_SWEEP_PID="${OLD_SWEEP_PID:-}"
OLD_WAITER_PID="${OLD_WAITER_PID:-}"
TRAIN_VENV="${TRAIN_VENV:-/workspace/venv}"
CACHE="${CACHE:-/workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019/dense_text_reason_vllm_onepass_len8192_dense_top64_rows10000.pt}"
OUT_ROOT="${OUT_ROOT:-/workspace/agentworld_runs/qwen35_global_hard_vllm_lr_sweep_corrected_shuffle_$(date -u +%Y%m%d_%H%M%S)}"
LRS="${LRS:-2e-5 1e-4 5e-4}"
STEPS="${STEPS:-2000}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-8192}"
ROW_SEED="${ROW_SEED:-17}"

cd "${CLARK_DIR}"

echo "waiting_for_active_pid=${ACTIVE_PID} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  stat="$(ps -p "${ACTIVE_PID}" -o stat= 2>/dev/null || true)"
  stat="${stat//[[:space:]]/}"
  if [[ -z "${stat}" || "${stat}" == Z* ]]; then
    break
  fi
  sleep 60
done
echo "active_pid_done=${ACTIVE_PID} $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ -n "${OLD_SWEEP_PID}" ]]; then
  kill -TERM "${OLD_SWEEP_PID}" 2>/dev/null || true
  kill -CONT "${OLD_SWEEP_PID}" 2>/dev/null || true
fi
if [[ -n "${OLD_WAITER_PID}" ]]; then
  kill -TERM "${OLD_WAITER_PID}" 2>/dev/null || true
fi

if [[ -d "${TRAIN_VENV}" ]]; then
  # shellcheck source=/dev/null
  source "${TRAIN_VENV}/bin/activate"
fi

export CACHE
export OUT_ROOT
export LRS
export STEPS
export MAX_TRAIN_TOKENS
export SHUFFLE_ROWS=1
export ROW_SEED
export SAVE_FINAL_PATCH=1
export LOG_EVERY="${LOG_EVERY:-10}"

echo "starting_corrected_lr_sweep out_root=${OUT_ROOT} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash experiments/qwen_agentworld/run_vllm_cache_lr_sweep.sh
echo "corrected_lr_sweep_done out_root=${OUT_ROOT} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
