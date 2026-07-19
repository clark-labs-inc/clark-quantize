#!/usr/bin/env bash
set -euo pipefail

# Domain-balanced 10-scenario smoke for the live Qwen3.5 BF16 vs Clark Air
# AgentWorld run. This uses a private subset data directory and private output
# dirs. It does not modify the full-run predictions.

RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
CLARK_DIR="${CLARK_DIR:-/workspace/clark-quantize}"
EVAL_VENV="${EVAL_VENV:-/workspace/agentworld-venv}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/workspace/AgentWorldBench}"
SUBSET_DIR="${SUBSET_DIR:-/workspace/AgentWorldBench_smoke10_balanced_20260628}"
SMOKE_DIR="${SMOKE_DIR:-$RESULT_ROOT/qwen35_bf16_vs_clark_air_balanced_smoke10_20260628}"

BF16_MODEL_NAME="${BF16_MODEL_NAME:-Qwen/Qwen3.5-9B}"
PACKED_MODEL_NAME="${PACKED_MODEL_NAME:-Qwen/Qwen3.5-9B-ClarkAir-Q2_0-g128}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://127.0.0.1:8001/v1}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY:-2}"

JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen/qwen3.5-flash-02-23}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-32768}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
OPENROUTER_ENV_FILE="${OPENROUTER_ENV_FILE:-/workspace/.openrouter.env}"

STREAM_INFER="$CLARK_DIR/experiments/qwen_agentworld/stream_official_infer.py"
STREAM_JUDGE="$CLARK_DIR/experiments/qwen_agentworld/stream_official_judge.py"
BF16_OUT="$SMOKE_DIR/bf16"
PACKED_OUT="$SMOKE_DIR/clark_air"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"
}

make_subset() {
  mkdir -p "$SUBSET_DIR"
  cp "$SOURCE_DATA_DIR/README.md" "$SUBSET_DIR/README.md" 2>/dev/null || true
  cp "$SOURCE_DATA_DIR/.gitattributes" "$SUBSET_DIR/.gitattributes" 2>/dev/null || true

  # Seven domains total. Select one from each domain, then add one extra from
  # the three largest domains for a 10-row balanced-but-small smoke.
  for domain in android mcp os web; do
    sed -n '1p' "$SOURCE_DATA_DIR/${domain}_test.jsonl" > "$SUBSET_DIR/${domain}_test.jsonl"
  done
  for domain in search swe terminal; do
    sed -n '1,2p' "$SOURCE_DATA_DIR/${domain}_test.jsonl" > "$SUBSET_DIR/${domain}_test.jsonl"
  done

  "$EVAL_VENV/bin/python" - "$SUBSET_DIR" <<'PY'
import json
import sys
from pathlib import Path

subset = Path(sys.argv[1])
rows = []
for path in sorted(subset.glob("*_test.jsonl")):
    count = sum(1 for line in path.read_text().splitlines() if line.strip())
    rows.append({"file": path.name, "rows": count})
manifest = {"rows": rows, "total": sum(row["rows"] for row in rows)}
(subset / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
}

wait_for_model_name() {
  local expected="$1"
  while true; do
    if curl -fsS "http://127.0.0.1:8001/v1/models" 2>/dev/null | grep -q "$expected"; then
      log "model endpoint is serving $expected"
      return 0
    fi
    log "waiting for model endpoint to serve $expected"
    sleep 120
  done
}

run_infer() {
  local label="$1"
  local model_name="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  local existing=0
  if [[ -f "$out_dir/predictions.jsonl" ]]; then
    existing=$(wc -l < "$out_dir/predictions.jsonl")
  fi
  if (( existing >= 10 )); then
    log "$label balanced smoke predictions already complete: $existing/10"
    return 0
  fi
  log "running $label balanced smoke inference with concurrency=$SMOKE_CONCURRENCY"
  "$EVAL_VENV/bin/python" "$STREAM_INFER" \
    --eval-dir /workspace/Qwen-AgentWorld/eval \
    --data-dir "$SUBSET_DIR" \
    --model-base-url "$MODEL_BASE_URL" \
    --model-name "$model_name" \
    --model-api-key "$MODEL_API_KEY" \
    --output-dir "$out_dir" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --concurrency "$SMOKE_CONCURRENCY" \
    > "$out_dir.infer.log" 2>&1
}

run_judge() {
  local label="$1"
  local out_dir="$2"
  local judge_dir="$out_dir/judge_qwen35_flash"
  mkdir -p "$judge_dir"
  local existing=0
  if [[ -f "$judge_dir/judged.jsonl" ]]; then
    existing=$(wc -l < "$judge_dir/judged.jsonl")
  fi
  if (( existing >= 10 )); then
    log "$label balanced smoke judge already complete: $existing/10"
    return 0
  fi
  log "running $label balanced smoke judge with $JUDGE_MODEL"
  "$EVAL_VENV/bin/python" "$STREAM_JUDGE" \
    --eval-dir /workspace/Qwen-AgentWorld/eval \
    --predictions "$out_dir/predictions.jsonl" \
    --output-dir "$judge_dir" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --max-tokens "$JUDGE_MAX_TOKENS" \
    --temperature "$JUDGE_TEMPERATURE" \
    --concurrency "$JUDGE_CONCURRENCY" \
    > "$out_dir.judge.log" 2>&1
}

write_summary() {
  "$EVAL_VENV/bin/python" - "$SMOKE_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dims = ["format", "factuality", "consistency", "realism", "quality"]

def load(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def key(row):
    return [row.get("task", ""), row.get("id", ""), row.get("turn_idx", ""), row.get("total_turns", "")]

def summarize(rows):
    return {
        "count": len(rows),
        "failed": sum(1 for row in rows if row.get("failed", 1.0) != 0.0),
        "avg_total_score": sum(float(row.get("total_score", 0.0) or 0.0) for row in rows) / max(len(rows), 1),
        "avg_dims": {
            dim: sum(float(row.get(dim, 0.0) or 0.0) for row in rows) / max(len(rows), 1)
            for dim in dims
        },
    }

bf16 = load(root / "bf16" / "judge_qwen35_flash" / "judged.jsonl")
clark = load(root / "clark_air" / "judge_qwen35_flash" / "judged.jsonl")
result = {
    "bf16": summarize(bf16),
    "clark_air": summarize(clark),
    "matched_keys": [key(row) for row in bf16] == [key(row) for row in clark],
    "keys": [key(row) for row in bf16],
}
if bf16 and clark:
    result["delta_avg_total_score"] = result["clark_air"]["avg_total_score"] - result["bf16"]["avg_total_score"]
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
}

if [[ -f "$OPENROUTER_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$OPENROUTER_ENV_FILE"
  set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  log "missing OPENROUTER_API_KEY or OPENAI_API_KEY; write $OPENROUTER_ENV_FILE first"
  exit 1
fi

cd "$CLARK_DIR"
mkdir -p "$SMOKE_DIR"
make_subset > "$SMOKE_DIR/subset_manifest.log"

wait_for_model_name "$BF16_MODEL_NAME"
run_infer BF16 "$BF16_MODEL_NAME" "$BF16_OUT"
run_judge BF16 "$BF16_OUT"

wait_for_model_name "$PACKED_MODEL_NAME"
run_infer "Clark Air" "$PACKED_MODEL_NAME" "$PACKED_OUT"
run_judge "Clark Air" "$PACKED_OUT"

write_summary
log "balanced smoke summary written to $SMOKE_DIR/summary.json"
