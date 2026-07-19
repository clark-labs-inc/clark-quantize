#!/usr/bin/env bash
set -euo pipefail

# Lightweight early signal for the full Qwen3.5 BF16-vs-Clark-Air AgentWorld
# run. This does not touch the GPU inference processes. It judges the first
# completed BF16 rows immediately, then waits for the first Clark Air rows after
# handoff and judges the matched prefix.

BF16_RUN="${BF16_RUN:-qwen35_9b_bf16_full_vllm_262k_32768_20260628}"
PACKED_RUN="${PACKED_RUN:-qwen35_9b_clark_air_q2_0_g128_round4_materialized_full_vllm_262k_32768_20260628}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
CLARK_DIR="${CLARK_DIR:-/workspace/clark-quantize}"
EVAL_VENV="${EVAL_VENV:-/workspace/agentworld-venv}"
SMOKE_ROWS="${SMOKE_ROWS:-10}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen/qwen3.5-flash-02-23}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-32768}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
OPENROUTER_ENV_FILE="${OPENROUTER_ENV_FILE:-/workspace/.openrouter.env}"
MAX_COLLAPSED_FRACTION="${MAX_COLLAPSED_FRACTION:-0.25}"
DEGENERACY_MIN_LEN="${DEGENERACY_MIN_LEN:-64}"

BF16_PREDICTIONS="$RESULT_ROOT/$BF16_RUN/predictions.jsonl"
PACKED_PREDICTIONS="$RESULT_ROOT/$PACKED_RUN/predictions.jsonl"
SMOKE_DIR="$RESULT_ROOT/qwen35_bf16_vs_clark_air_smoke${SMOKE_ROWS}_20260628"
BF16_SMOKE_PRED="$SMOKE_DIR/bf16_predictions_${SMOKE_ROWS}.jsonl"
PACKED_SMOKE_PRED="$SMOKE_DIR/clark_air_predictions_${SMOKE_ROWS}.jsonl"
STREAM_JUDGE="$CLARK_DIR/experiments/qwen_agentworld/stream_official_judge.py"
DEGENERACY_DETECTOR="$CLARK_DIR/experiments/qwen_agentworld/detect_generation_degeneracy.py"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"
}

row_count() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path"
  else
    echo 0
  fi
}

wait_for_rows() {
  local label="$1"
  local path="$2"
  local rows=0
  while true; do
    rows=$(row_count "$path")
    if (( rows >= SMOKE_ROWS )); then
      log "$label predictions have $rows rows; smoke prefix ready"
      return 0
    fi
    log "waiting for $label predictions: $rows/$SMOKE_ROWS rows"
    sleep 120
  done
}

write_prefix() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" ]] && (( $(wc -l < "$dst") >= SMOKE_ROWS )); then
    return 0
  fi
  head -n "$SMOKE_ROWS" "$src" > "$dst.tmp"
  mv "$dst.tmp" "$dst"
}

run_degeneracy_gate() {
  local label="$1"
  local predictions="$2"
  local output_dir="$3"
  local baseline="${4:-}"
  local report="$output_dir/degeneracy_gate.json"

  mkdir -p "$output_dir"
  local args=(
    "$DEGENERACY_DETECTOR"
    --predictions "$predictions"
    --out "$report"
    --min-len "$DEGENERACY_MIN_LEN"
    --fail-fraction "$MAX_COLLAPSED_FRACTION"
  )
  if [[ -n "$baseline" && -f "$baseline" ]]; then
    args+=(--baseline "$baseline")
  fi
  "$EVAL_VENV/bin/python" "${args[@]}"

  local failed
  failed="$("$EVAL_VENV/bin/python" - "$report" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())["summary"]
print("1" if summary["failed_gate"] else "0")
print(summary["collapsed_fraction"])
print(summary["collapsed"])
print(summary["count"])
PY
)"
  local failed_gate collapsed_fraction collapsed count
  failed_gate="$(printf '%s\n' "$failed" | sed -n '1p')"
  collapsed_fraction="$(printf '%s\n' "$failed" | sed -n '2p')"
  collapsed="$(printf '%s\n' "$failed" | sed -n '3p')"
  count="$(printf '%s\n' "$failed" | sed -n '4p')"
  if [[ "$failed_gate" == "1" ]]; then
    log "$label smoke degeneracy gate failed: collapsed=$collapsed/$count fraction=$collapsed_fraction max=$MAX_COLLAPSED_FRACTION report=$report"
    return 1
  fi
  log "$label smoke degeneracy gate passed: collapsed=$collapsed/$count fraction=$collapsed_fraction report=$report"
}

run_judge() {
  local label="$1"
  local predictions="$2"
  local output_dir="$3"
  local log_path="$4"
  local baseline="${5:-}"

  mkdir -p "$output_dir"
  local existing=0
  if [[ -f "$output_dir/judged.jsonl" ]]; then
    existing=$(wc -l < "$output_dir/judged.jsonl")
  fi
  if (( existing >= SMOKE_ROWS )); then
    log "$label smoke judge already complete: $existing/$SMOKE_ROWS"
    return 0
  fi

  if ! run_degeneracy_gate "$label" "$predictions" "$output_dir" "$baseline"; then
    log "not launching $label smoke judge because cheap collapse gate failed"
    return 2
  fi

  log "running $label smoke judge with $JUDGE_MODEL"
  "$EVAL_VENV/bin/python" "$STREAM_JUDGE" \
    --eval-dir /workspace/Qwen-AgentWorld/eval \
    --predictions "$predictions" \
    --output-dir "$output_dir" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --max-tokens "$JUDGE_MAX_TOKENS" \
    --temperature "$JUDGE_TEMPERATURE" \
    --concurrency "$JUDGE_CONCURRENCY" \
    > "$log_path" 2>&1
  log "$label smoke judge complete"
}

write_summary() {
  "$EVAL_VENV/bin/python" - "$SMOKE_DIR" "$SMOKE_ROWS" <<'PY'
import json
import sys
from pathlib import Path

smoke_dir = Path(sys.argv[1])
smoke_rows = int(sys.argv[2])

def load(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def key(row):
    return [row.get("task", ""), row.get("id", ""), row.get("turn_idx", ""), row.get("total_turns", "")]

def summarize(rows):
    dims = ["format", "factuality", "consistency", "realism", "quality"]
    count = len(rows)
    return {
        "count": count,
        "failed": sum(1 for row in rows if row.get("failed", 1.0) != 0.0),
        "avg_total_score": sum(float(row.get("total_score", 0.0) or 0.0) for row in rows) / max(count, 1),
        "avg_dims": {
            dim: sum(float(row.get(dim, 0.0) or 0.0) for row in rows) / max(count, 1)
            for dim in dims
        },
    }

bf16 = load(smoke_dir / "bf16_judge" / "judged.jsonl")
clark = load(smoke_dir / "clark_air_judge" / "judged.jsonl")
result = {
    "smoke_rows": smoke_rows,
    "bf16": summarize(bf16),
    "clark_air": summarize(clark),
    "matched_prefix": [key(a) for a in bf16] == [key(b) for b in clark],
    "keys": [key(row) for row in bf16[:smoke_rows]],
}
if bf16 and clark:
    result["delta_avg_total_score"] = (
        result["clark_air"]["avg_total_score"] - result["bf16"]["avg_total_score"]
    )
(smoke_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
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

wait_for_rows BF16 "$BF16_PREDICTIONS"
write_prefix "$BF16_PREDICTIONS" "$BF16_SMOKE_PRED"
run_judge \
  BF16 \
  "$BF16_SMOKE_PRED" \
  "$SMOKE_DIR/bf16_judge" \
  "$SMOKE_DIR/bf16_judge.log"

wait_for_rows "Clark Air" "$PACKED_PREDICTIONS"
write_prefix "$PACKED_PREDICTIONS" "$PACKED_SMOKE_PRED"
run_judge \
  "Clark Air" \
  "$PACKED_SMOKE_PRED" \
  "$SMOKE_DIR/clark_air_judge" \
  "$SMOKE_DIR/clark_air_judge.log" \
  "$BF16_SMOKE_PRED"

write_summary
log "smoke summary written to $SMOKE_DIR/summary.json"
