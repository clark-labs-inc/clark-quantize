#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/clark-quantize}"
PYTHON="${PYTHON:-/workspace/venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
OUT_CACHE="${OUT_CACHE:-/workspace/agentworld_runs/qwen35_dense_top64_tranche1_$(date +%Y%m%d_%H%M%S)}"

# Tranche 1 is intentionally text/reasoning/multilingual. Cauldron/doc-image
# rows are deferred until the hard text-side signal is stable.
SHARDS="${SHARDS:-32}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
TOP_K="${TOP_K:-64}"
AYA_PER_SHARD="${AYA_PER_SHARD:-256}"
OPENR1_PER_SHARD="${OPENR1_PER_SHARD:-512}"
FINEWEB_PER_CONFIG="${FINEWEB_PER_CONFIG:-64}"
FINEWEB_CONFIGS="${FINEWEB_CONFIGS:-spa_Latn,fra_Latn,deu_Latn,arb_Arab,hin_Deva,cmn_Hani,rus_Cyrl,jpn_Jpan,kor_Hang}"

mkdir -p "$OUT_CACHE"
cd "$REPO"

IFS=',' read -r -a fineweb_configs_array <<< "$FINEWEB_CONFIGS"
FINEWEB_CONFIG_COUNT="${#fineweb_configs_array[@]}"
ROWS_PER_SHARD=$((AYA_PER_SHARD + OPENR1_PER_SHARD + FINEWEB_PER_CONFIG * FINEWEB_CONFIG_COUNT))

{
  echo "run=qwen35_dense_top64_tranche1"
  echo "out_cache=$OUT_CACHE"
  echo "shards=$SHARDS rows_per_shard=$ROWS_PER_SHARD max_length=$MAX_LENGTH top_k=$TOP_K"
  echo "mix aya=$AYA_PER_SHARD openr1=$OPENR1_PER_SHARD fineweb_per_config=$FINEWEB_PER_CONFIG fineweb_configs=$FINEWEB_CONFIGS"
  echo "cauldron=0 mm_openr1=0 sharegptvideo=0"
} | tee "$OUT_CACHE/launch.log"

for ((shard = 0; shard < SHARDS; shard++)); do
  shard_id="$(printf "%03d" "$shard")"
  out="$OUT_CACHE/dense_text_reason_shard_${shard_id}_len${MAX_LENGTH}_dense_top${TOP_K}.pt"
  aya_offset=$((shard * AYA_PER_SHARD))
  openr1_offset=$((shard * OPENR1_PER_SHARD))
  fineweb_offset=$((shard * FINEWEB_PER_CONFIG))
  echo "shard_start id=$shard_id out=$out $(date -Is)" | tee -a "$OUT_CACHE/launch.log"
  "$PYTHON" experiments/qwen_agentworld/collect_multimodal_teacher_cache.py \
    --model-dir "$MODEL_DIR" \
    --out "$out" \
    --max-length "$MAX_LENGTH" \
    --dense-positions \
    --last-n 0 \
    --top-k "$TOP_K" \
    --aya-samples "$AYA_PER_SHARD" \
    --aya-offset "$aya_offset" \
    --openr1-samples "$OPENR1_PER_SHARD" \
    --openr1-offset "$openr1_offset" \
    --fineweb-configs "$FINEWEB_CONFIGS" \
    --fineweb-per-config "$FINEWEB_PER_CONFIG" \
    --fineweb-offset "$fineweb_offset" \
    --mm-openr1-samples 0 \
    --sharegptvideo-samples 0 \
    --cauldron-per-config 0 \
    --max-rows "$ROWS_PER_SHARD" \
    > "$OUT_CACHE/shard_${shard_id}.log" 2>&1
  echo "shard_done id=$shard_id $(date -Is)" | tee -a "$OUT_CACHE/launch.log"
done

"$PYTHON" experiments/qwen_agentworld/summarize_teacher_cache_shards.py \
  "$OUT_CACHE/dense_text_reason_shard_*.pt.json" \
  --out "$OUT_CACHE/cache_index_latest.json" \
  > "$OUT_CACHE/cache_index_latest.stdout"

echo "cache_done out_cache=$OUT_CACHE $(date -Is)" | tee -a "$OUT_CACHE/launch.log"
