#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 infer|judge|score RUN_NAME" >&2
  exit 2
fi

CMD="$1"
RUN_NAME="$2"

VENV="${VENV:-/workspace/agentworld-venv}"
EVAL_DIR="${EVAL_DIR:-/workspace/Qwen-AgentWorld/eval}"
DATA_DIR="${DATA_DIR:-/workspace/AgentWorldBench}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen-AgentWorld-35B-A3B}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
OUT_DIR="$RESULT_ROOT/$RUN_NAME"

mkdir -p "$OUT_DIR"
cd "$EVAL_DIR"

case "$CMD" in
  infer)
    "$VENV/bin/python" eval.py infer \
      --data-dir "$DATA_DIR" \
      --model-base-url "$MODEL_BASE_URL" \
      --model-name "$MODEL_NAME" \
      --model-api-key "$MODEL_API_KEY" \
      --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --output-dir "$OUT_DIR"
    ;;
  judge)
    : "${JUDGE_BASE_URL:?set JUDGE_BASE_URL after user confirms judge provider}"
    : "${JUDGE_MODEL:?set JUDGE_MODEL after user confirms judge model}"
    judge_args=(
      eval.py judge
      --predictions "$OUT_DIR/predictions.jsonl" \
      --judge-base-url "$JUDGE_BASE_URL" \
      --judge-model "$JUDGE_MODEL" \
      --max-tokens "${JUDGE_MAX_TOKENS:-32768}" \
      --temperature "${JUDGE_TEMPERATURE:-0.6}" \
      --output-dir "$OUT_DIR"
    )
    if [ -n "${JUDGE_API_KEY:-}" ]; then
      export OPENAI_API_KEY="$JUDGE_API_KEY"
    fi
    "$VENV/bin/python" "${judge_args[@]}"
    ;;
  score)
    "$VENV/bin/python" eval.py score --predictions "$OUT_DIR/judged.jsonl"
    ;;
  *)
    echo "unknown command: $CMD" >&2
    exit 2
    ;;
esac
