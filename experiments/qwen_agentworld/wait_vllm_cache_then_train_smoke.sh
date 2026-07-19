#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/clark-quantize}"
RUN_DIR="${RUN_DIR:-/workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019}"
MAX_ROWS="${MAX_ROWS:-10000}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
GPU_FREE_CHECK_SECONDS="${GPU_FREE_CHECK_SECONDS:-15}"
GPU_USED_READY_MIB="${GPU_USED_READY_MIB:-5000}"
TRAIN_VENV="${TRAIN_VENV:-/workspace/venv}"
CACHE="${CACHE:-${RUN_DIR}/dense_text_reason_vllm_onepass_len8192_dense_top64_rows10000.pt}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-/workspace/agentworld_runs/qwen35_global_hard_vllm_smoke_after_cache_$(date -u +%Y%m%d_%H%M%S)}"
LOG="${LOG:-${RUN_DIR}/handoff_$(date -u +%Y%m%d_%H%M%S).log}"

cd "${PROJECT_DIR}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

build_pid=""
if [[ -f "${RUN_DIR}/build.pid" ]]; then
  build_pid="$(tr -d '[:space:]' < "${RUN_DIR}/build.pid")"
fi

log "handoff_start run_dir=${RUN_DIR} cache=${CACHE} max_rows=${MAX_ROWS} build_pid=${build_pid:-none}"

while [[ -n "${build_pid}" ]] && kill -0 "${build_pid}" 2>/dev/null; do
  python3 experiments/qwen_agentworld/status_vllm_cache_run.py "${RUN_DIR}" --max-rows "${MAX_ROWS}" | tee -a "${LOG}"
  log "cache_builder_alive sleeping_seconds=${CHECK_INTERVAL_SECONDS}"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

log "cache_builder_exited_or_missing"
python3 experiments/qwen_agentworld/status_vllm_cache_run.py "${RUN_DIR}" --max-rows "${MAX_ROWS}" | tee -a "${LOG}" || true

while pgrep -af "collect_text_teacher_cache_vllm|VLLM::EngineCore" \
  | awk -v self="$$" '$1 != self && $0 !~ /wait_vllm_cache_then_train_smoke/ && $0 !~ /pgrep -af/ {print}' \
  >/tmp/vllm_cache_handoff_processes.txt && [[ -s /tmp/vllm_cache_handoff_processes.txt ]]; do
  log "vllm_processes_still_alive sleeping_seconds=${GPU_FREE_CHECK_SECONDS}"
  cat /tmp/vllm_cache_handoff_processes.txt | tee -a "${LOG}"
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
  partial_meta="${CACHE}.partial.json"
  if [[ -f "${partial_meta}" ]]; then
    log "final_cache_missing; continuation_plan_from=${partial_meta}"
    python3 experiments/qwen_agentworld/plan_vllm_cache_continuation.py \
      "${partial_meta}" \
      --aya-target 2000 \
      --openr1-target 4000 \
      --fineweb-per-config-target 450 | tee -a "${LOG}" || true
  fi
  log "final_cache_missing cache=${CACHE}; refusing_to_train_from_partial"
  exit 2
fi

if [[ -d "${TRAIN_VENV}" ]]; then
  # shellcheck source=/dev/null
  source "${TRAIN_VENV}/bin/activate"
fi

python - "${CACHE}" "${MAX_ROWS}" <<'PY' | tee -a "${LOG}"
import json
import sys
import torch

cache_path = sys.argv[1]
min_rows = int(sys.argv[2])
cache = torch.load(cache_path, map_location="cpu")
rows = cache.get("rows") or []
if cache.get("format") != "qwen_agentworld_teacher_behavior_v1":
    raise SystemExit(f"bad format: {cache.get('format')!r}")
if len(rows) < min_rows:
    raise SystemExit(f"too few rows: {len(rows)} < {min_rows}")
row = rows[0]
values = row.get("teacher_top_logprobs")
indices = row.get("top_indices")
if values is None or indices is None:
    raise SystemExit("missing teacher_top_logprobs/top_indices")
if tuple(values.shape) != tuple(indices.shape):
    raise SystemExit(f"shape mismatch: {tuple(values.shape)} vs {tuple(indices.shape)}")
print(json.dumps({
    "cache_validated": cache_path,
    "rows": len(rows),
    "shape0": list(values.shape),
    "top_mass0": float(values.float().exp().sum(-1).mean()),
}, indent=2))
PY

log "starting_train_smoke out_dir=${SMOKE_OUT_DIR}"
CACHE="${CACHE}" OUT_DIR="${SMOKE_OUT_DIR}" bash experiments/qwen_agentworld/run_vllm_cache_train_smoke.sh
log "train_smoke_done out_dir=${SMOKE_OUT_DIR}"
