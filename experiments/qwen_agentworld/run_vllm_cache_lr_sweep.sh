#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
CACHE="${CACHE:-/workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019/dense_text_reason_vllm_onepass_len8192_dense_top64_rows10000.pt}"
OUT_ROOT="${OUT_ROOT:-/workspace/agentworld_runs/qwen35_global_hard_vllm_lr_sweep_$(date -u +%Y%m%d_%H%M%S)}"
LRS="${LRS:-2e-5 1e-4 5e-4}"
STEPS="${STEPS:-2000}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-8192}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-}"
HEAD_CHUNK_ROWS="${HEAD_CHUNK_ROWS:-8192}"
LOG_EVERY="${LOG_EVERY:-10}"
MAX_UPDATE_RMS="${MAX_UPDATE_RMS:-0.5}"
CODE_FORENSICS_GROUPS_PER_MODULE="${CODE_FORENSICS_GROUPS_PER_MODULE:-64}"
MASTER_DELTA_SAMPLE_SIZE="${MASTER_DELTA_SAMPLE_SIZE:-65536}"
TRAIN_VENV="${TRAIN_VENV:-/workspace/venv}"
SAVE_FINAL_PATCH="${SAVE_FINAL_PATCH:-1}"
SHUFFLE_ROWS="${SHUFFLE_ROWS:-1}"
ROW_SEED="${ROW_SEED:-17}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${CACHE}" ]]; then
  echo "cache_missing=${CACHE}" >&2
  exit 2
fi

if [[ -d "${TRAIN_VENV}" ]]; then
  # shellcheck source=/dev/null
  source "${TRAIN_VENV}/bin/activate"
fi

if [[ -z "${CODE_PERSISTENCE_SHORT_STEP:-}" || -z "${CODE_PERSISTENCE_LONG_STEP:-}" ]]; then
  if (( STEPS >= 1500 )); then
    CODE_PERSISTENCE_SHORT_STEP="${CODE_PERSISTENCE_SHORT_STEP:-500}"
    CODE_PERSISTENCE_LONG_STEP="${CODE_PERSISTENCE_LONG_STEP:-1500}"
  elif (( STEPS >= 30 )); then
    CODE_PERSISTENCE_SHORT_STEP="${CODE_PERSISTENCE_SHORT_STEP:-$(( STEPS / 3 ))}"
    CODE_PERSISTENCE_LONG_STEP="${CODE_PERSISTENCE_LONG_STEP:-$(( (2 * STEPS) / 3 ))}"
  else
    CODE_PERSISTENCE_SHORT_STEP="${CODE_PERSISTENCE_SHORT_STEP:-0}"
    CODE_PERSISTENCE_LONG_STEP="${CODE_PERSISTENCE_LONG_STEP:-0}"
  fi
fi

if [[ "${SHUFFLE_ROWS}" == "1" || "${SHUFFLE_ROWS}" == "true" ]]; then
  DATA_ORDER="deterministic_shuffle"
else
  DATA_ORDER="sequential_from_cache_start"
fi

mkdir -p "${OUT_ROOT}"
cat > "${OUT_ROOT}/sweep_config.json" <<JSON
{
  "cache": "${CACHE}",
  "model_dir": "${MODEL_DIR}",
  "lrs": "${LRS}",
  "steps": ${STEPS},
  "max_train_tokens": ${MAX_TRAIN_TOKENS},
  "max_train_rows": "${MAX_TRAIN_ROWS}",
  "head_chunk_rows": ${HEAD_CHUNK_ROWS},
  "max_update_rms": ${MAX_UPDATE_RMS},
  "code_persistence_short_step": ${CODE_PERSISTENCE_SHORT_STEP},
  "code_persistence_long_step": ${CODE_PERSISTENCE_LONG_STEP}
}
JSON

echo "out_root=${OUT_ROOT}"

for lr in ${LRS}; do
  lr_tag="$(printf '%s' "${lr}" | tr '+-' 'pm' | tr '.' '_')"
  out_dir="${OUT_ROOT}/lr_${lr_tag}"
  mkdir -p "${out_dir}"
  out_patch="${out_dir}/final_patch.safetensors"
  echo "starting_lr=${lr} out_dir=${out_dir}"

  cat > "${out_dir}/arm_manifest.json" <<JSON
{
  "cache": "${CACHE}",
  "model_dir": "${MODEL_DIR}",
  "lr": ${lr},
  "steps": ${STEPS},
  "max_train_tokens": ${MAX_TRAIN_TOKENS},
  "max_train_rows": "${MAX_TRAIN_ROWS}",
  "head_chunk_rows": ${HEAD_CHUNK_ROWS},
  "max_update_rms": ${MAX_UPDATE_RMS},
  "master_delta_sample_size": ${MASTER_DELTA_SAMPLE_SIZE},
  "code_forensics_groups_per_module": ${CODE_FORENSICS_GROUPS_PER_MODULE},
  "code_persistence_short_step": ${CODE_PERSISTENCE_SHORT_STEP},
  "code_persistence_long_step": ${CODE_PERSISTENCE_LONG_STEP},
  "fresh_ptq_init_expected": true,
  "init_weight_patch": null,
  "data_order": "${DATA_ORDER}",
  "shuffle_rows": "${SHUFFLE_ROWS}",
  "row_seed": ${ROW_SEED},
  "save_final_patch": "${SAVE_FINAL_PATCH}",
  "out_patch": "${out_patch}"
}
JSON

  args=(
    experiments/qwen_agentworld/train_all2d_ternary_qat.py
    --model-dir "${MODEL_DIR}"
    --cache "${CACHE}"
    --out-report "${out_dir}/report.json"
    --mode ternary
    --group-size 128
    --train-layers all
    --train-embed
    --train-head
    --train-norms
    --master-dtype float32
    --optimizer factored_adafactor
    --post-accumulate-step
    --grad-clip 0
    --explicit-layer-checkpointing
    --chunked-head-kd
    --topk-other-bucket
    --head-chunk-rows "${HEAD_CHUNK_ROWS}"
    --anti-weight 0
    --alpha-start 1
    --alpha-end 1
    --steps "${STEPS}"
    --lr "${lr}"
    --max-update-rms "${MAX_UPDATE_RMS}"
    --max-train-tokens "${MAX_TRAIN_TOKENS}"
    --code-forensics-groups-per-module "${CODE_FORENSICS_GROUPS_PER_MODULE}"
    --master-delta-sample-size "${MASTER_DELTA_SAMPLE_SIZE}"
    --code-persistence-short-step "${CODE_PERSISTENCE_SHORT_STEP}"
    --code-persistence-long-step "${CODE_PERSISTENCE_LONG_STEP}"
    --log-every "${LOG_EVERY}"
    --no-log-first-cycle
  )
  if [[ "${SHUFFLE_ROWS}" == "1" || "${SHUFFLE_ROWS}" == "true" ]]; then
    args+=(--shuffle-rows --row-seed "${ROW_SEED}")
  fi
  if [[ "${SAVE_FINAL_PATCH}" == "1" || "${SAVE_FINAL_PATCH}" == "true" ]]; then
    args+=(--out-patch "${out_patch}" --no-skip-final-save)
  else
    args+=(--skip-final-save)
  fi
  if [[ -n "${MAX_TRAIN_ROWS}" ]]; then
    args+=(--max-train-rows "${MAX_TRAIN_ROWS}")
  fi

  python "${args[@]}" > "${out_dir}/train.log" 2>&1
  echo "done_lr=${lr} report=${out_dir}/report.json"
done

echo "sweep_done out_root=${OUT_ROOT}"
