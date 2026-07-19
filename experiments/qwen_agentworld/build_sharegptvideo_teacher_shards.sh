#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-/workspace/agentworld_runs/qwen35_multimodal_cache_20260630}"
REPO="${REPO:-/workspace/clark-quantize}"
PYTHON="${PYTHON:-/workspace/venv/bin/python}"
START_SHARD="${START_SHARD:-0}"
END_SHARD="${END_SHARD:-15}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-128}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LAST_N="${LAST_N:-96}"
TOP_K="${TOP_K:-256}"
MAX_FRAMES="${MAX_FRAMES:-8}"
METADATA="${METADATA:-video_instruction/train/sft/video_caption_300k.jsonl}"

mkdir -p "$OUT"
cd "$REPO"

for shard in $(seq "$START_SHARD" "$END_SHARD"); do
  printf -v shard_id "%03d" "$shard"
  offset=$((shard * SAMPLES_PER_SHARD))
  cache="$OUT/mm_video_shard_${shard_id}_sharegptvideo${SAMPLES_PER_SHARD}_len${MAX_LENGTH}_last${LAST_N}_top${TOP_K}.pt"
  log="$OUT/build_mm_video_shard_${shard_id}.log"

  if [[ -s "${cache}.json" ]]; then
    echo "skip existing video shard=$shard_id cache=$cache"
    continue
  fi

  echo "building video shard=$shard_id offset=$offset cache=$cache"
  for attempt in 1 2 3; do
    echo "attempt=$attempt video_shard=$shard_id" | tee "$log"
    rm -f "$cache" "${cache}.json"
    if "$PYTHON" experiments/qwen_agentworld/collect_multimodal_teacher_cache.py \
      --model-dir /workspace/models/Qwen3.5-9B \
      --out "$cache" \
      --max-length "$MAX_LENGTH" \
      --last-n "$LAST_N" \
      --top-k "$TOP_K" \
      --aya-samples 0 \
      --openr1-samples 0 \
      --mm-openr1-samples 0 \
      --fineweb-per-config 0 \
      --cauldron-per-config 0 \
      --sharegptvideo-samples "$SAMPLES_PER_SHARD" \
      --sharegptvideo-offset "$offset" \
      --sharegptvideo-metadata "$METADATA" \
      --sharegptvideo-max-frames "$MAX_FRAMES" \
      --max-rows "$SAMPLES_PER_SHARD" \
      >> "$log" 2>&1; then
      df -h /workspace | tee -a "$log"
      break
    fi
    echo "video_shard=$shard_id attempt=$attempt failed" | tee -a "$log"
    sleep $((attempt * 30))
  done
done
