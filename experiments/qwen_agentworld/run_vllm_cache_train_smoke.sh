#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
CACHE="${CACHE:-/workspace/agentworld_runs/qwen35_dense_top64_tranche1_vllm_onepass_safe_20260702_081019/dense_text_reason_vllm_onepass_len8192_dense_top64_rows10000.pt}"
OUT_DIR="${OUT_DIR:-/workspace/agentworld_runs/qwen35_global_hard_vllm_smoke_$(date -u +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-20}"
LR="${LR:-1e-4}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-8192}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-64}"
HEAD_CHUNK_ROWS="${HEAD_CHUNK_ROWS:-8192}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_DIR}"
echo "out_dir=${OUT_DIR}"

python experiments/qwen_agentworld/train_all2d_ternary_qat.py \
  --model-dir "${MODEL_DIR}" \
  --cache "${CACHE}" \
  --out-report "${OUT_DIR}/report.json" \
  --skip-final-save \
  --mode ternary \
  --group-size 128 \
  --train-layers all \
  --train-embed \
  --train-head \
  --train-norms \
  --master-dtype float32 \
  --optimizer factored_adafactor \
  --post-accumulate-step \
  --grad-clip 0 \
  --explicit-layer-checkpointing \
  --chunked-head-kd \
  --topk-other-bucket \
  --head-chunk-rows "${HEAD_CHUNK_ROWS}" \
  --anti-weight 0 \
  --alpha-start 1 \
  --alpha-end 1 \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --max-update-rms 0.5 \
  --max-train-tokens "${MAX_TRAIN_TOKENS}" \
  --max-train-rows "${MAX_TRAIN_ROWS}" \
  --code-forensics-groups-per-module 64 \
  --master-delta-sample-size 65536 \
  --log-every 1 \
  > "${OUT_DIR}/train.log" 2>&1

echo "report=${OUT_DIR}/report.json"
