#!/usr/bin/env bash
set -euo pipefail

SOURCE_CACHE="${SOURCE_CACHE:-/workspace/agentworld_runs/qwen35_multimodal_cache_20260630}"
OUT_CACHE="${OUT_CACHE:-/workspace/agentworld_runs/qwen35_multimodal_cache_hiddenproj_20260701}"
REPO="${REPO:-/workspace/clark-quantize}"
PYTHON="${PYTHON:-/workspace/venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-0,1,8,12,20,27,30,31}"
PROJECTION_DIM="${PROJECTION_DIM:-128}"
PROJECTION_SEED="${PROJECTION_SEED:-20260701}"
LIMIT_SHARDS="${LIMIT_SHARDS:-0}"

mkdir -p "$OUT_CACHE"
cd "$REPO"

"$PYTHON" experiments/qwen_agentworld/label_teacher_cache_hidden_proj.py \
  --model-dir "$MODEL_DIR" \
  --cache-glob "$SOURCE_CACHE/mm_shard_*.pt" \
  --cache-glob "$SOURCE_CACHE/mm_reason_shard_*.pt" \
  --cache-glob "$SOURCE_CACHE/mm_video_shard_*.pt" \
  --out-dir "$OUT_CACHE" \
  --hidden-layers "$HIDDEN_LAYERS" \
  --projection-dim "$PROJECTION_DIM" \
  --projection-seed "$PROJECTION_SEED" \
  --limit-shards "$LIMIT_SHARDS"

"$PYTHON" experiments/qwen_agentworld/summarize_teacher_cache_shards.py \
  "$OUT_CACHE/mm_shard_*.pt.json" \
  "$OUT_CACHE/mm_reason_shard_*.pt.json" \
  "$OUT_CACHE/mm_video_shard_*.pt.json" \
  --out "$OUT_CACHE/cache_index_latest.json"
