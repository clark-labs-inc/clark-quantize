#!/usr/bin/env bash
set -euo pipefail

CACHE="${CACHE:-/workspace/agentworld_runs/qwen35_multimodal_cache_hiddenproj_20260701}"
REPO="${REPO:-/workspace/clark-quantize}"
PYTHON="${PYTHON:-/workspace/venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
RUN_ROOT="${RUN_ROOT:-/workspace/agentworld_runs/qwen35_global_t2_fulltrain_v2}"
MIN_ROWS="${MIN_ROWS:-10000}"

PHASE1_STEPS="${PHASE1_STEPS:-15000}"
PHASE2_STEPS="${PHASE2_STEPS:-50000}"
CACHE_SHARD_STEPS="${CACHE_SHARD_STEPS:-128}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-1200}"
LR="${LR:-5e-6}"
MODE="${MODE:-ternary}"
GROUP_SIZE="${GROUP_SIZE:-128}"
MASTER_DTYPE="${MASTER_DTYPE:-bfloat16}"
FORMAT_BOOST_PHASE1="${FORMAT_BOOST_PHASE1:-2.0}"
FORMAT_BOOST_PHASE2="${FORMAT_BOOST_PHASE2:-2.5}"
HIDDEN_WEIGHT_PHASE1="${HIDDEN_WEIGHT_PHASE1:-0.05}"
HIDDEN_WEIGHT_PHASE2="${HIDDEN_WEIGHT_PHASE2:-0.10}"
HARD_SHADOW_START="${HARD_SHADOW_START:-0.05}"
HARD_SHADOW_END="${HARD_SHADOW_END:-0.25}"
HARD_SHADOW_EVERY="${HARD_SHADOW_EVERY:-2}"
AGENTWORLD_GATE_DATA_DIR="${AGENTWORLD_GATE_DATA_DIR:-/workspace/AgentWorldBench_round8_fixed10_20260629}"

mkdir -p "$RUN_ROOT"
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

refresh_index() {
  "$PYTHON" experiments/qwen_agentworld/summarize_teacher_cache_shards.py \
    "$CACHE/mm_shard_*.pt.json" \
    "$CACHE/mm_reason_shard_*.pt.json" \
    "$CACHE/mm_video_shard_*.pt.json" \
    --out "$CACHE/cache_index_latest.json" \
    > "$RUN_ROOT/cache_index_latest.stdout"
}

cache_rows() {
  "$PYTHON" - <<PY
import json, pathlib
p=pathlib.Path("$CACHE/cache_index_latest.json")
j=json.loads(p.read_text())["summary"]
print(j["rows"])
PY
}

bad_shards() {
  "$PYTHON" - <<PY
import json, pathlib
p=pathlib.Path("$CACHE/cache_index_latest.json")
j=json.loads(p.read_text())["summary"]
print(len(j["bad_shards"]))
PY
}

cache_task_summary() {
  "$PYTHON" - <<PY
import json, pathlib
p=pathlib.Path("$CACHE/cache_index_latest.json")
j=json.loads(p.read_text())["summary"]
print(json.dumps(j.get("tasks", j.get("task_counts", {})), sort_keys=True))
PY
}

hidden_available() {
  "$PYTHON" - <<PY
import glob
import torch
for pattern in ("$CACHE/mm_shard_*.pt", "$CACHE/mm_reason_shard_*.pt", "$CACHE/mm_video_shard_*.pt"):
    paths = sorted(glob.glob(pattern))
    if not paths:
        continue
    cache = torch.load(paths[0], map_location="cpu")
    rows = cache.get("rows") or []
    row = rows[0] if rows else {}
    ok = bool(cache.get("hidden_layers")) and bool(cache.get("hidden_projection_dim")) and bool(row.get("teacher_hidden_proj"))
    print(1 if ok else 0)
    break
else:
    print(0)
PY
}

sample_cache_preflight() {
  "$PYTHON" - <<PY
import glob
import torch
for pattern in ("$CACHE/mm_shard_*.pt", "$CACHE/mm_reason_shard_*.pt", "$CACHE/mm_video_shard_*.pt"):
    paths = sorted(glob.glob(pattern))
    if not paths:
        continue
    cache = torch.load(paths[0], map_location="cpu")
    rows = cache.get("rows") or []
    if not rows:
        continue
    row = rows[0]
    assert "top_values" in row and "top_indices" in row, row.keys()
    assert row["top_values"].shape == row["top_indices"].shape, (row["top_values"].shape, row["top_indices"].shape)
    assert int(row.get("tokens", 0)) > 0
    assert row.get("teacher_hidden_proj"), "missing projected hidden labels"
    print(f"sample_cache_ok path={paths[0]} format={cache.get('format')} rows={len(rows)} top_shape={tuple(row['top_values'].shape)} hidden_layers={cache.get('hidden_layers', [])} hidden_projection_dim={cache.get('hidden_projection_dim')}")
    break
else:
    raise SystemExit("no readable cache shard with rows")
PY
}

tokenizer_preflight() {
  "$PYTHON" - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$MODEL_DIR", trust_remote_code=True)
required = ["<think>", "</think>", "<predicted_observation>", "</predicted_observation>"]
ids = {token: tok.convert_tokens_to_ids(token) for token in required}
print(f"tokenizer_preflight chat_template={bool(getattr(tok, 'chat_template', None))} eos={tok.eos_token_id} required_ids={ids}")
PY
}

fixed10_preflight() {
  "$PYTHON" - <<PY
from pathlib import Path
import json
data_dir = Path("$AGENTWORLD_GATE_DATA_DIR")
files = [data_dir] if data_dir.is_file() else sorted(data_dir.glob("*_test.jsonl"))
rows = []
for path in files:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
if not rows:
    raise SystemExit(f"no AgentWorld gate rows loaded from {data_dir}")
print(f"fixed10_preflight rows={len(rows)} data_dir={data_dir}")
PY
}

refresh_index
rows="$(cache_rows)"
bad="$(bad_shards)"
tasks="$(cache_task_summary)"
hidden_ok="$(hidden_available)"
{
  echo "run=qwen35_global_t2_fulltrain_v2"
  echo "cache rows=$rows bad_shards=$bad min_rows=$MIN_ROWS tasks=$tasks"
  echo "config phase1_steps=$PHASE1_STEPS phase2_steps=$PHASE2_STEPS lr=$LR max_train_tokens=$MAX_TRAIN_TOKENS"
  echo "config optimizer=mirror_sgd group_adagrad=1 loss_adaptive_lr=1 task_family_normalize=1"
  echo "config hidden_required=1 hidden_available=$hidden_ok hidden_weight_phase1=$HIDDEN_WEIGHT_PHASE1 hidden_weight_phase2=$HIDDEN_WEIGHT_PHASE2"
  echo "config hard_shadow_start=$HARD_SHADOW_START hard_shadow_end=$HARD_SHADOW_END hard_shadow_every=$HARD_SHADOW_EVERY"
} | tee "$RUN_ROOT/launch.log"
if [[ "$bad" != "0" ]]; then
  echo "refusing to train: cache has bad shards" | tee -a "$RUN_ROOT/launch.log"
  exit 2
fi
if (( rows < MIN_ROWS )); then
  echo "refusing to train: cache rows below MIN_ROWS" | tee -a "$RUN_ROOT/launch.log"
  exit 3
fi
if [[ "$hidden_ok" != "1" ]]; then
  echo "refusing to train: v2 requires projected hidden labels" | tee -a "$RUN_ROOT/launch.log"
  exit 4
fi
sample_cache_preflight | tee -a "$RUN_ROOT/launch.log"
tokenizer_preflight | tee -a "$RUN_ROOT/launch.log"
fixed10_preflight | tee -a "$RUN_ROOT/launch.log"

common_args=(
  --model-dir "$MODEL_DIR"
  --cache-glob "$CACHE/mm_shard_*.pt"
  --cache-glob "$CACHE/mm_reason_shard_*.pt"
  --cache-glob "$CACHE/mm_video_shard_*.pt"
  --interleave-cache-globs
  --cache-shard-steps "$CACHE_SHARD_STEPS"
  --mode "$MODE"
  --group-size "$GROUP_SIZE"
  --master-dtype "$MASTER_DTYPE"
  --train-layers all
  --train-embed
  --train-head
  --train-norms
  --max-train-tokens "$MAX_TRAIN_TOKENS"
  --optimizer mirror_sgd
  --group-adagrad
  --loss-adaptive-lr
  --loss-adaptive-min-scale 0.25
  --loss-adaptive-max-scale 1.0
  --task-family-normalize
  --task-family-min-scale 0.25
  --task-family-max-scale 2.0
  --lr "$LR"
  --weight-decay 0.0
  --anti-weight 0.1
  --code-weight 0.001
  --grad-clip 0.5
  --max-update-rms 0.5
  --log-every 10
  --save-latest-every 1000
)

phase1_dir="$RUN_ROOT/phase1_soft_landing"
mkdir -p "$phase1_dir"
echo "phase1_start $(date -Is)" | tee -a "$RUN_ROOT/launch.log"
"$PYTHON" experiments/qwen_agentworld/train_all2d_ternary_qat.py \
  "${common_args[@]}" \
  --steps "$PHASE1_STEPS" \
  --alpha-start 0.0 \
  --alpha-end 1.0 \
  --prox-rho 0.0 \
  --format-boost "$FORMAT_BOOST_PHASE1" \
  --hidden-weight "$HIDDEN_WEIGHT_PHASE1" \
  --hard-shadow-weight-start "$HARD_SHADOW_START" \
  --hard-shadow-weight-end "$HARD_SHADOW_END" \
  --hard-shadow-every "$HARD_SHADOW_EVERY" \
  --task-max-frac cauldron:0.10 \
  --out-patch "$phase1_dir/qwen35_all2d_t2_phase1_patch.safetensors" \
  --out-full-patch "$phase1_dir/qwen35_all2d_t2_phase1_full_patch.safetensors" \
  --out-report "$phase1_dir/report_phase1.json" \
  --latest-patch "$phase1_dir/latest_phase1_patch.safetensors" \
  --latest-report "$phase1_dir/latest_phase1_report.json" \
  > "$phase1_dir/train.log" 2>&1
echo "phase1_done $(date -Is)" | tee -a "$RUN_ROOT/launch.log"

phase2_dir="$RUN_ROOT/phase2_hard_ternary"
mkdir -p "$phase2_dir"
echo "phase2_start $(date -Is)" | tee -a "$RUN_ROOT/launch.log"
"$PYTHON" experiments/qwen_agentworld/train_all2d_ternary_qat.py \
  "${common_args[@]}" \
  --steps "$PHASE2_STEPS" \
  --alpha-start 1.0 \
  --alpha-end 1.0 \
  --prox-rho 0.0005 \
  --format-boost "$FORMAT_BOOST_PHASE2" \
  --hidden-weight "$HIDDEN_WEIGHT_PHASE2" \
  --task-max-frac cauldron:0.20 \
  --init-weight-patch "$phase1_dir/qwen35_all2d_t2_phase1_full_patch.safetensors" \
  --out-patch "$phase2_dir/qwen35_all2d_t2_phase2_patch.safetensors" \
  --out-full-patch "$phase2_dir/qwen35_all2d_t2_phase2_full_patch.safetensors" \
  --out-report "$phase2_dir/report_phase2.json" \
  --latest-patch "$phase2_dir/latest_phase2_patch.safetensors" \
  --latest-report "$phase2_dir/latest_phase2_report.json" \
  > "$phase2_dir/train.log" 2>&1
echo "phase2_done $(date -Is)" | tee -a "$RUN_ROOT/launch.log"
