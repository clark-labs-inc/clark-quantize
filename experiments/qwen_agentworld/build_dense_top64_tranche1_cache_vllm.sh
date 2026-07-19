#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-/workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_$(date -u +%Y%m%d_%H%M%S)}"
SHARDS="${SHARDS:-32}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
TOP_K="${TOP_K:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
SAVE_EVERY_ROWS="${SAVE_EVERY_ROWS:-256}"
AYA_PER_SHARD="${AYA_PER_SHARD:-256}"
OPENR1_PER_SHARD="${OPENR1_PER_SHARD:-512}"
FINEWEB_PER_CONFIG="${FINEWEB_PER_CONFIG:-64}"
FINEWEB_CONFIGS="${FINEWEB_CONFIGS:-spa_Latn,fra_Latn,deu_Latn,arb_Arab,hin_Deva,cmn_Hani,rus_Cyrl,jpn_Jpan,kor_Hang}"

export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "${OUT_DIR}"
echo "out_dir=${OUT_DIR}"

for ((shard=0; shard<SHARDS; shard++)); do
  aya_offset=$((shard * AYA_PER_SHARD))
  openr1_offset=$((shard * OPENR1_PER_SHARD))
  fineweb_offset=$((shard * FINEWEB_PER_CONFIG))
  max_rows=$((AYA_PER_SHARD + OPENR1_PER_SHARD + FINEWEB_PER_CONFIG * 9))
  out_file="${OUT_DIR}/dense_text_reason_vllm_shard_$(printf '%03d' "${shard}")_len${MAX_LENGTH}_dense_top${TOP_K}.pt"
  echo "shard=${shard}/${SHARDS} out=${out_file}"
  python experiments/qwen_agentworld/collect_text_teacher_cache_vllm.py \
    --model-dir "${MODEL_DIR}" \
    --out "${out_file}" \
    --max-length "${MAX_LENGTH}" \
    --top-k "${TOP_K}" \
    --batch-size "${BATCH_SIZE}" \
    --max-batch-tokens "${MAX_BATCH_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --save-every-rows "${SAVE_EVERY_ROWS}" \
    --aya-samples "${AYA_PER_SHARD}" \
    --aya-offset "${aya_offset}" \
    --openr1-samples "${OPENR1_PER_SHARD}" \
    --openr1-offset "${openr1_offset}" \
    --fineweb-configs "${FINEWEB_CONFIGS}" \
    --fineweb-per-config "${FINEWEB_PER_CONFIG}" \
    --fineweb-offset "${fineweb_offset}" \
    --max-rows "${max_rows}" \
    > "${OUT_DIR}/shard_$(printf '%03d' "${shard}").log" 2>&1
done
