#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-qwen35_9b_bf16_full_262k_32768}"

cd /workspace/clark-quantize
source experiments/qwen_agentworld/qwen35_9b_env.sh

if [ "${DOWNLOAD_ASSETS:-1}" = "1" ]; then
  bash experiments/qwen_agentworld/download_assets.sh
fi

# Use the lightweight Transformers endpoint by default. It is slower than
# vLLM/SGLang but avoids turning this helper into a CUDA-serving installer.
SERVER_LOG="$RESULT_ROOT/${RUN_NAME}.transformers_server.log"
SERVER_PID="$RESULT_ROOT/${RUN_NAME}.transformers_server.pid"
mkdir -p "$RESULT_ROOT"

cleanup() {
  if [ -f "$SERVER_PID" ]; then
    pid="$(cat "$SERVER_PID" 2>/dev/null || true)"
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${PORT:-8000}/v1/models" >/dev/null 2>&1; then
  PORT="${PORT:-8000}" \
  MODEL_DIR="$MODEL_DIR" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  nohup bash experiments/qwen_agentworld/serve_transformers_openai.sh > "$SERVER_LOG" 2>&1 &
  echo $! > "$SERVER_PID"

  echo "waiting for Qwen3.5-9B endpoint at http://127.0.0.1:${PORT:-8000}/v1"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT:-8000}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$(cat "$SERVER_PID")" 2>/dev/null; then
      echo "server exited before readiness; tailing log" >&2
      tail -160 "$SERVER_LOG" >&2 || true
      exit 4
    fi
    sleep 10
  done
  if [ "${ready:-0}" != "1" ]; then
    echo "endpoint did not become ready; tailing log" >&2
    tail -160 "$SERVER_LOG" >&2 || true
    exit 4
  fi
fi

MODEL_BASE_URL="${MODEL_BASE_URL:-http://127.0.0.1:${PORT:-8000}/v1}" \
MODEL_NAME="$MODEL_NAME" \
DATA_DIR="$DATA_DIR" \
MAX_TOKENS="$MAX_TOKENS" \
TEMPERATURE="${TEMPERATURE:-0.6}" \
bash experiments/qwen_agentworld/run_official_eval.sh infer "$RUN_NAME"

if [ "${RUN_JUDGE:-0}" = "1" ]; then
  : "${JUDGE_API_KEY:?set JUDGE_API_KEY to run paid AgentWorld judge}"
  JUDGE_API_KEY="$JUDGE_API_KEY" \
  JUDGE_BASE_URL="$JUDGE_BASE_URL" \
  JUDGE_MODEL="$JUDGE_MODEL" \
  bash experiments/qwen_agentworld/run_official_eval.sh judge "$RUN_NAME"

  JUDGE_API_KEY="$JUDGE_API_KEY" \
  JUDGE_BASE_URL="$JUDGE_BASE_URL" \
  JUDGE_MODEL="$JUDGE_MODEL" \
  bash experiments/qwen_agentworld/run_official_eval.sh score "$RUN_NAME" \
    > "$RESULT_ROOT/$RUN_NAME/score_openrouter_qwen35_flash.txt"
  cat "$RESULT_ROOT/$RUN_NAME/score_openrouter_qwen35_flash.txt"
fi
