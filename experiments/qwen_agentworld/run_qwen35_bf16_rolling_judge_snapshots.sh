#!/usr/bin/env bash
set -euo pipefail

# Opportunistically judge BF16 rows while the long full-context inference is
# still running. The streaming judge is resumable by sample key, so each
# snapshot only scores rows not already present in judged.jsonl.

BF16_RUN="${BF16_RUN:-qwen35_9b_bf16_full_vllm_262k_32768_20260628}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
CLARK_DIR="${CLARK_DIR:-/workspace/clark-quantize}"
EVAL_VENV="${EVAL_VENV:-/workspace/agentworld-venv}"
EXPECTED_ROWS="${EXPECTED_ROWS:-2170}"
JUDGE_DIR_NAME="${JUDGE_DIR_NAME:-judge_qwen35_flash}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen/qwen3.5-flash-02-23}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-32768}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
MIN_NEW_ROWS="${MIN_NEW_ROWS:-160}"
POLL_SECONDS="${POLL_SECONDS:-120}"
OPENROUTER_ENV_FILE="${OPENROUTER_ENV_FILE:-/workspace/.openrouter.env}"

BF16_DIR="$RESULT_ROOT/$BF16_RUN"
PREDICTIONS="$BF16_DIR/predictions.jsonl"
JUDGE_DIR="$BF16_DIR/$JUDGE_DIR_NAME"
JUDGED="$JUDGE_DIR/judged.jsonl"
PID_PATH="$RESULT_ROOT/$BF16_RUN.judge_qwen35_flash.pid"
STREAM_JUDGE="$CLARK_DIR/experiments/qwen_agentworld/stream_official_judge.py"
SNAPSHOT_DIR="$BF16_DIR/judge_snapshots"

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

mkdir -p "$JUDGE_DIR" "$SNAPSHOT_DIR"
cd "$CLARK_DIR"

while true; do
  prediction_rows=$(row_count "$PREDICTIONS")
  judged_rows=$(row_count "$JUDGED")

  if (( prediction_rows >= EXPECTED_ROWS && judged_rows >= EXPECTED_ROWS )); then
    log "BF16 rolling judge complete: judged=$judged_rows predictions=$prediction_rows"
    exit 0
  fi

  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    log "judge already running pid $(cat "$PID_PATH"); predictions=$prediction_rows judged=$judged_rows"
    sleep "$POLL_SECONDS"
    continue
  fi

  if (( prediction_rows <= judged_rows )); then
    log "no unjudged rows yet; predictions=$prediction_rows judged=$judged_rows"
    sleep "$POLL_SECONDS"
    continue
  fi

  if (( prediction_rows < EXPECTED_ROWS && prediction_rows - judged_rows < MIN_NEW_ROWS )); then
    log "waiting for larger snapshot; predictions=$prediction_rows judged=$judged_rows min_new=$MIN_NEW_ROWS"
    sleep "$POLL_SECONDS"
    continue
  fi

  snapshot="$SNAPSHOT_DIR/predictions_snapshot_$(date -u +%Y%m%dT%H%M%SZ)_${prediction_rows}.jsonl"
  head -n "$prediction_rows" "$PREDICTIONS" > "$snapshot.tmp"
  mv "$snapshot.tmp" "$snapshot"

  log "starting rolling BF16 judge snapshot rows=$prediction_rows judged=$judged_rows"
  nohup env \
    "$EVAL_VENV/bin/python" "$STREAM_JUDGE" \
    --eval-dir /workspace/Qwen-AgentWorld/eval \
    --predictions "$snapshot" \
    --output-dir "$JUDGE_DIR" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --max-tokens "$JUDGE_MAX_TOKENS" \
    --temperature "$JUDGE_TEMPERATURE" \
    --concurrency "$JUDGE_CONCURRENCY" \
    > "$RESULT_ROOT/$BF16_RUN.rolling_judge_qwen35_flash.log" 2>&1 &
  echo $! > "$PID_PATH"
  sleep "$POLL_SECONDS"
done
