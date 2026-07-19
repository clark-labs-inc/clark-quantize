#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-/workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_$(date -u +%Y%m%d_%H%M%S)}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
TOP_K="${TOP_K:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
SAVE_EVERY_ROWS="${SAVE_EVERY_ROWS:-256}"
AYA_SAMPLES="${AYA_SAMPLES:-2000}"
AYA_OFFSET="${AYA_OFFSET:-0}"
OPENR1_SAMPLES="${OPENR1_SAMPLES:-4000}"
OPENR1_OFFSET="${OPENR1_OFFSET:-0}"
FINEWEB_PER_CONFIG="${FINEWEB_PER_CONFIG:-450}"
FINEWEB_OFFSET="${FINEWEB_OFFSET:-0}"
MAX_ROWS="${MAX_ROWS:-10000}"
FINEWEB_CONFIGS="${FINEWEB_CONFIGS:-spa_Latn,fra_Latn,deu_Latn,arb_Arab,hin_Deva,cmn_Hani,rus_Cyrl,jpn_Jpan,kor_Hang}"

export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "${OUT_DIR}"
echo "out_dir=${OUT_DIR}"

out_file="${OUT_DIR}/dense_text_reason_vllm_onepass_len${MAX_LENGTH}_dense_top${TOP_K}_rows${MAX_ROWS}.pt"
python experiments/qwen_agentworld/collect_text_teacher_cache_vllm.py \
  --model-dir "${MODEL_DIR}" \
  --out "${out_file}" \
  --max-length "${MAX_LENGTH}" \
  --top-k "${TOP_K}" \
  --batch-size "${BATCH_SIZE}" \
  --max-batch-tokens "${MAX_BATCH_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --save-every-rows "${SAVE_EVERY_ROWS}" \
  --aya-samples "${AYA_SAMPLES}" \
  --aya-offset "${AYA_OFFSET}" \
  --openr1-samples "${OPENR1_SAMPLES}" \
  --openr1-offset "${OPENR1_OFFSET}" \
  --fineweb-configs "${FINEWEB_CONFIGS}" \
  --fineweb-per-config "${FINEWEB_PER_CONFIG}" \
  --fineweb-offset "${FINEWEB_OFFSET}" \
  --max-rows "${MAX_ROWS}" \
  > "${OUT_DIR}/build.log" 2>&1

echo "out=${out_file}"
