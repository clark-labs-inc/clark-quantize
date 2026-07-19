#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
RUN_NAME="${2:-}"

if [ -z "$MODE" ] || [ -z "$RUN_NAME" ]; then
  echo "usage: $0 bf16|endpoint RUN_NAME" >&2
  echo "bf16: start official vLLM/SGLang-style BF16 server, then infer/judge/score" >&2
  echo "endpoint: use an already-running OpenAI-compatible endpoint, then infer/judge/score" >&2
  exit 2
fi

VENV="${VENV:-/workspace/agentworld-venv}"
DATA_DIR="${DATA_DIR:-/workspace/AgentWorldBench}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen-AgentWorld-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen-AgentWorld-35B-A3B}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://127.0.0.1:8000/v1}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
REQUIRED_GPUS="${REQUIRED_GPUS:-$TP}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
SERVER_BACKEND="${SERVER_BACKEND:-vllm}"
VLLM_BIN="${VLLM_BIN:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
SERVER_LOG="$RESULT_ROOT/${RUN_NAME}.server.log"
SERVER_PID="$RESULT_ROOT/${RUN_NAME}.server.pid"

mkdir -p "$RESULT_ROOT"

gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [ "$MODE" = "bf16" ] && [ "$gpu_count" -lt "$REQUIRED_GPUS" ]; then
  cat >&2 <<EOF
Refusing full-fidelity BF16 run: found ${gpu_count} GPU(s), need ${REQUIRED_GPUS}.

Official Qwen-AgentWorld serving uses tensor parallelism 4 with
context/max-model-len ${MAX_MODEL_LEN}. Reduced-context single-A100 smoke runs
are not full validation.
EOF
  exit 42
fi

cleanup() {
  if [ -f "$SERVER_PID" ]; then
    pid="$(cat "$SERVER_PID" 2>/dev/null || true)"
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
}
if [ "$MODE" = "bf16" ]; then
  trap cleanup EXIT
fi

cd /workspace/clark-quantize

if [ "$MODE" = "bf16" ]; then
  nvidia_lib_root="$VENV/lib/python3.11/site-packages/nvidia"
  if [ -d "$nvidia_lib_root" ]; then
    while IFS= read -r cuda_lib; do
      export LD_LIBRARY_PATH="$cuda_lib:${LD_LIBRARY_PATH:-}"
    done < <(find "$nvidia_lib_root" -type d -name lib | sort)
  fi

  if curl -sf "$MODEL_BASE_URL/models" >/dev/null 2>&1; then
    echo "endpoint already running at $MODEL_BASE_URL; refusing to reuse it for bf16 mode" >&2
    exit 3
  fi
  case "$SERVER_BACKEND" in
    vllm)
      if [ -z "$VLLM_BIN" ]; then
        if [ -x "$VENV/bin/vllm" ]; then
          VLLM_BIN="$VENV/bin/vllm"
        elif command -v vllm >/dev/null 2>&1; then
          VLLM_BIN="$(command -v vllm)"
        fi
      fi
      if [ -z "$VLLM_BIN" ]; then
        echo "vLLM is not installed on this node" >&2
        exit 43
      fi
      nohup "$VLLM_BIN" serve "$MODEL_DIR" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --reasoning-parser qwen3 \
        --language-model-only \
        --trust-remote-code \
        > "$SERVER_LOG" 2>&1 &
      ;;
    sglang)
      if ! "$VENV/bin/python" -c "import sglang" >/dev/null 2>&1; then
        echo "SGLang is not installed in $VENV" >&2
        exit 43
      fi
      nohup "$VENV/bin/python" -m sglang.launch_server \
        --model-path "$MODEL_DIR" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --context-length "$MAX_MODEL_LEN" \
        --reasoning-parser qwen3 \
        > "$SERVER_LOG" 2>&1 &
      ;;
    *)
      echo "unsupported SERVER_BACKEND=$SERVER_BACKEND; expected vllm or sglang" >&2
      exit 2
      ;;
  esac
  echo $! > "$SERVER_PID"

  echo "waiting for full-fidelity endpoint at $MODEL_BASE_URL"
  for _ in $(seq 1 180); do
    if curl -sf "$MODEL_BASE_URL/models" >/dev/null 2>&1; then
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
elif [ "$MODE" = "endpoint" ]; then
  if ! curl -sf "$MODEL_BASE_URL/models" >/dev/null 2>&1; then
    echo "no endpoint is ready at $MODEL_BASE_URL" >&2
    exit 4
  fi
else
  echo "unsupported mode: $MODE" >&2
  exit 2
fi

DATA_DIR="$DATA_DIR" \
MODEL_BASE_URL="$MODEL_BASE_URL" \
MODEL_NAME="$MODEL_NAME" \
MAX_TOKENS="$MAX_TOKENS" \
TEMPERATURE="$TEMPERATURE" \
bash experiments/qwen_agentworld/run_official_eval.sh infer "$RUN_NAME"

: "${JUDGE_BASE_URL:?set JUDGE_BASE_URL for full-fidelity judging}"
: "${JUDGE_MODEL:?set JUDGE_MODEL for full-fidelity judging}"
: "${JUDGE_API_KEY:?set JUDGE_API_KEY for full-fidelity judging}"

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
