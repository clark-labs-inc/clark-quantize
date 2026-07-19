#!/usr/bin/env bash
set -euo pipefail

# Queue the compressed Clark Air AgentWorld run after the BF16 baseline
# inference finishes. This intentionally evaluates a materialized HF checkpoint
# for behavior; the packed parent-pack remains the actual storage artifact.

BF16_RUN="${BF16_RUN:-qwen35_9b_bf16_full_vllm_262k_32768_20260628}"
PACKED_RUN="${PACKED_RUN:-qwen35_9b_clark_air_q2_0_g128_round4_materialized_full_vllm_262k_32768_20260628}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
CLARK_DIR="${CLARK_DIR:-/workspace/clark-quantize}"
VLLM_VENV="${VLLM_VENV:-/workspace/vllm-qwen35-venv}"
EVAL_VENV="${EVAL_VENV:-/workspace/agentworld-venv}"
BF16_SERVER_PID_FILE="${BF16_SERVER_PID_FILE:-/workspace/agentworld_runs/qwen35_9b_bf16_vllm_resume_server.pid}"
PACKED_MODEL_DIR="${PACKED_MODEL_DIR:-/workspace/models/Qwen3.5-9B-ClarkAir-Q2_0-g128-round4-materialized}"
PACKED_SERVED_NAME="${PACKED_SERVED_NAME:-Qwen/Qwen3.5-9B-ClarkAir-Q2_0-g128}"
PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
STREAM_CONCURRENCY="${STREAM_CONCURRENCY:-4}"
EXPECTED_ROWS="${EXPECTED_ROWS:-2170}"

BF16_PID_FILE="$RESULT_ROOT/$BF16_RUN.infer.pid"
BF16_STREAM_PID_FILE="${BF16_STREAM_PID_FILE:-$RESULT_ROOT/$BF16_RUN.stream.pid}"
BF16_PREDICTIONS="$RESULT_ROOT/$BF16_RUN/predictions.jsonl"
PACKED_SERVER_LOG="$RESULT_ROOT/$PACKED_RUN.vllm_server.log"
PACKED_SERVER_PID_FILE="$RESULT_ROOT/$PACKED_RUN.vllm_server.pid"
PACKED_INFER_LOG="$RESULT_ROOT/$PACKED_RUN.infer.log"
STREAM_INFER="$CLARK_DIR/experiments/qwen_agentworld/stream_official_infer.py"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"
}

process_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_pid_tree() {
  local pid="$1"
  if ! process_alive "$pid"; then
    return 0
  fi
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! process_alive "$pid"; then
      return 0
    fi
    sleep 1
  done
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

wait_for_health() {
  for i in $(seq 1 90); do
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      log "vLLM health ready after poll $i"
      return 0
    fi
    if [[ -f "$PACKED_SERVER_PID_FILE" ]] && ! process_alive "$(cat "$PACKED_SERVER_PID_FILE")"; then
      log "packed vLLM exited before health"
      tail -160 "$PACKED_SERVER_LOG" || true
      return 1
    fi
    sleep 10
  done
  log "packed vLLM did not become healthy"
  tail -160 "$PACKED_SERVER_LOG" || true
  return 1
}

if [[ ! -f "$BF16_PID_FILE" ]]; then
  log "missing BF16 pid file: $BF16_PID_FILE"
  exit 1
fi

log "waiting for BF16 inference run $BF16_RUN"
while process_alive "$(cat "$BF16_PID_FILE")"; do
  tail -5 "$RESULT_ROOT/$BF16_RUN.infer.log" || true
  sleep 120
done

bf16_rows=0
if [[ -f "$BF16_PREDICTIONS" ]]; then
  bf16_rows=$(wc -l < "$BF16_PREDICTIONS")
fi

if (( bf16_rows < EXPECTED_ROWS )) && [[ -f "$BF16_STREAM_PID_FILE" ]] && process_alive "$(cat "$BF16_STREAM_PID_FILE")"; then
  log "BF16 stream already running with $bf16_rows/$EXPECTED_ROWS rows; waiting for pid $(cat "$BF16_STREAM_PID_FILE")"
  while process_alive "$(cat "$BF16_STREAM_PID_FILE")"; do
    tail -5 "$RESULT_ROOT/$BF16_RUN.stream_fallback.log" || true
    sleep 120
  done
  bf16_rows=0
  if [[ -f "$BF16_PREDICTIONS" ]]; then
    bf16_rows=$(wc -l < "$BF16_PREDICTIONS")
  fi
fi

if (( bf16_rows < EXPECTED_ROWS )); then
  log "BF16 inference has $bf16_rows/$EXPECTED_ROWS rows; starting/resuming streaming BF16 fallback"
  tail -120 "$RESULT_ROOT/$BF16_RUN.infer.log" || true
  "$EVAL_VENV/bin/python" "$STREAM_INFER" \
    --eval-dir /workspace/Qwen-AgentWorld/eval \
    --data-dir /workspace/AgentWorldBench \
    --model-base-url "http://$HOST:$PORT/v1" \
    --model-name Qwen/Qwen3.5-9B \
    --model-api-key EMPTY \
    --output-dir "$RESULT_ROOT/$BF16_RUN" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --concurrency "$STREAM_CONCURRENCY" \
    > "$RESULT_ROOT/$BF16_RUN.stream_fallback.log" 2>&1
fi
bf16_rows=$(wc -l < "$BF16_PREDICTIONS")
if (( bf16_rows < EXPECTED_ROWS )); then
  log "BF16 streaming exited before completion: $bf16_rows/$EXPECTED_ROWS rows"
  tail -120 "$RESULT_ROOT/$BF16_RUN.stream_fallback.log" || true
  exit 1
fi
log "BF16 predictions complete: $bf16_rows/$EXPECTED_ROWS lines"

if [[ -f "$BF16_SERVER_PID_FILE" ]]; then
  log "stopping BF16 vLLM server pid $(cat "$BF16_SERVER_PID_FILE")"
  stop_pid_tree "$(cat "$BF16_SERVER_PID_FILE")"
fi

source "$VLLM_VENV/bin/activate"
export CUDA_HOME="$VLLM_VENV/lib/python3.10/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export XDG_CONFIG_HOME=/workspace/.config
export XDG_CACHE_HOME=/workspace/.cache
export VLLM_USE_FLASHINFER_SAMPLER=0

log "starting packed/materialized vLLM server from $PACKED_MODEL_DIR"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$PACKED_MODEL_DIR" \
  --served-model-name "$PACKED_SERVED_NAME" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --port "$PORT" \
  --host "$HOST" \
  > "$PACKED_SERVER_LOG" 2>&1 &
echo $! > "$PACKED_SERVER_PID_FILE"

wait_for_health

log "smoking packed/materialized endpoint"
curl -fsS "http://$HOST:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$PACKED_SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Return exactly: ok\"}],\"max_tokens\":8,\"temperature\":0}" \
  > "$RESULT_ROOT/$PACKED_RUN.smoke.json"

cd "$CLARK_DIR"
source experiments/qwen_agentworld/qwen35_9b_env.sh
log "starting packed/materialized AgentWorld inference run $PACKED_RUN"
nohup env \
  "$EVAL_VENV/bin/python" "$STREAM_INFER" \
  --eval-dir /workspace/Qwen-AgentWorld/eval \
  --data-dir /workspace/AgentWorldBench \
  --model-base-url "http://$HOST:$PORT/v1" \
  --model-name "$PACKED_SERVED_NAME" \
  --model-api-key EMPTY \
  --output-dir "$RESULT_ROOT/$PACKED_RUN" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --concurrency "$STREAM_CONCURRENCY" \
  > "$PACKED_INFER_LOG" 2>&1 &
echo $! > "$RESULT_ROOT/$PACKED_RUN.infer.pid"
log "packed/materialized inference launched pid $(cat "$RESULT_ROOT/$PACKED_RUN.infer.pid")"
