#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/clark-quantize}"
RUN_DIR="${RUN_DIR:-/workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019}"
CACHE="${CACHE:-${RUN_DIR}/dense_text_reason_vllm_onepass_len8192_dense_top64_rows10000.pt}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
GPU_FREE_CHECK_SECONDS="${GPU_FREE_CHECK_SECONDS:-15}"
GPU_USED_READY_MIB="${GPU_USED_READY_MIB:-5000}"
OUT_ROOT="${OUT_ROOT:-/workspace/agentworld_runs/qwen35_global_hard_vllm_lr_sweep_after_smoke_$(date -u +%Y%m%d_%H%M%S)}"
LOG="${LOG:-${RUN_DIR}/post_smoke_lr_sweep_$(date -u +%Y%m%d_%H%M%S).log}"

cd "${PROJECT_DIR}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

log "post_smoke_sweep_wait_start run_dir=${RUN_DIR} cache=${CACHE} out_root=${OUT_ROOT}"

smoke_dir=""
while [[ -z "${smoke_dir}" ]]; do
  handoff_log="${RUN_DIR}/handoff_latest.log"
  if [[ ! -f "${handoff_log}" ]]; then
    handoff_log="$(ls -t "${RUN_DIR}"/handoff_*.log 2>/dev/null | head -1 || true)"
  fi
  if [[ -n "${handoff_log}" && -f "${handoff_log}" ]]; then
    smoke_dir="$(
      awk '/train_smoke_done out_dir=/ {
        line=$0
        sub(/^.*train_smoke_done out_dir=/, "", line)
        print line
      }' "${handoff_log}" | tail -1
    )"
  fi
  if [[ -n "${smoke_dir}" && -f "${smoke_dir}/report.json" ]]; then
    log "smoke_done smoke_dir=${smoke_dir}"
    break
  fi
  smoke_dir=""
  log "smoke_not_done sleeping_seconds=${CHECK_INTERVAL_SECONDS}"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

while pgrep -af "train_all2d_ternary_qat.py" \
  | awk -v self="$$" '$1 != self && $0 !~ /wait_smoke_then_lr_sweep/ && $0 !~ /pgrep -af/ {print}' \
  >/tmp/qwen_qat_training_processes.txt && [[ -s /tmp/qwen_qat_training_processes.txt ]]; do
  log "training_process_still_alive sleeping_seconds=${GPU_FREE_CHECK_SECONDS}"
  cat /tmp/qwen_qat_training_processes.txt | tee -a "${LOG}"
  sleep "${GPU_FREE_CHECK_SECONDS}"
done

while true; do
  gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
  if [[ -z "${gpu_used}" ]]; then
    log "gpu_memory_query_empty; continuing"
    break
  fi
  if (( gpu_used <= GPU_USED_READY_MIB )); then
    log "gpu_ready used_mib=${gpu_used} threshold_mib=${GPU_USED_READY_MIB}"
    break
  fi
  log "gpu_still_busy used_mib=${gpu_used} threshold_mib=${GPU_USED_READY_MIB} sleeping_seconds=${GPU_FREE_CHECK_SECONDS}"
  sleep "${GPU_FREE_CHECK_SECONDS}"
done

if [[ ! -f "${CACHE}" ]]; then
  log "cache_missing_after_smoke cache=${CACHE}; refusing_lr_sweep"
  exit 2
fi

log "starting_lr_sweep out_root=${OUT_ROOT}"
CACHE="${CACHE}" OUT_ROOT="${OUT_ROOT}" bash experiments/qwen_agentworld/run_vllm_cache_lr_sweep.sh | tee -a "${LOG}"
log "lr_sweep_done out_root=${OUT_ROOT}"
