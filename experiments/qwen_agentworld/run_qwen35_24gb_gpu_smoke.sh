#!/usr/bin/env bash
set -euo pipefail

# 24GB GPU smoke path for Qwen3.5-9B Clark Air artifacts.
#
# This intentionally avoids full-context vLLM. The materialized HF checkpoint is
# roughly 19.3GB, and max-context serving with a large KV cache will not fit on a
# 24GB card. This script keeps the node useful for:
#   - packed artifact inspection / storage-lossless verification;
#   - optional materialization when disk allows;
#   - short-context inference smoke through the lightweight Transformers server;
#   - packed-runtime loader/kernel development.

CLARK_DIR="${CLARK_DIR:-/workspace/clark-quantize}"
VENV="${VENV:-/workspace/agentworld-venv}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
RUN_NAME="${RUN_NAME:-qwen35_9b_clark_air_24gb_smoke_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
PACK_DIR="${PACK_DIR:-/workspace/agentworld_runs/qwen35_9b_pack_q2_0_g128_round4_qat_patch}"
MANIFEST="${MANIFEST:-$PACK_DIR/manifest.json}"
MATERIALIZED_DIR="${MATERIALIZED_DIR:-/workspace/models/Qwen3.5-9B-ClarkAir-Q2_0-g128-round4-materialized}"

PORT="${PORT:-8002}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0}"
MIN_MATERIALIZE_FREE_GB="${MIN_MATERIALIZE_FREE_GB:-45}"

# Modes:
#   packed-runtime: load BF16 parent, then replace packed linears at runtime.
#   materialized: serve an already materialized HF checkpoint, or create it if
#                 MATERIALIZE=1/auto and there is enough disk.
MODE="${MODE:-packed-runtime}"
MATERIALIZE="${MATERIALIZE:-auto}"
SERVE="${SERVE:-1}"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

free_gb_for_path() {
  local path="$1"
  mkdir -p "$path"
  df -BG "$path" | awk 'NR == 2 {gsub("G","",$4); print $4}'
}

require_file() {
  [[ -f "$1" ]] || fail "missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || fail "missing required directory: $1"
}

mkdir -p "$RUN_DIR"

log "run_dir=$RUN_DIR"
log "gpu snapshot"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,driver_version --format=csv,noheader || true
df -h / /workspace 2>/dev/null || df -h /

require_dir "$CLARK_DIR"
require_file "$MANIFEST"

log "packed artifact inspection"
"$VENV/bin/python" - <<PY | tee "$RUN_DIR/pack_summary.json"
import json
from pathlib import Path

manifest = json.loads(Path("$MANIFEST").read_text())
lossless_path = Path("$PACK_DIR/lossless_verify.json")
lossless = json.loads(lossless_path.read_text()) if lossless_path.exists() else {}
summary = manifest.get("summary") or {}
print(json.dumps({
    "manifest": "$MANIFEST",
    "pack_dir": "$PACK_DIR",
    "group_size": manifest.get("group_size"),
    "packed_parent_count": len(manifest.get("packed", [])),
    "manifest_summary": summary,
    "lossless_summary": lossless.get("summary"),
}, indent=2))
PY

log "storage-lossless verification"
cd "$CLARK_DIR"
"$VENV/bin/python" experiments/qwen_agentworld/verify_parent_pack_lossless.py \
  --manifest "$MANIFEST" \
  --out "$RUN_DIR/lossless_verify.json"

if [[ "$MODE" == "materialized" ]]; then
  if [[ ! -d "$MATERIALIZED_DIR" ]]; then
    if [[ "$MATERIALIZE" == "0" || "$MATERIALIZE" == "false" ]]; then
      fail "materialized mode requested, but $MATERIALIZED_DIR does not exist and MATERIALIZE=$MATERIALIZE"
    fi
    require_dir "$BASE_MODEL_DIR"
    free_gb="$(free_gb_for_path "$(dirname "$MATERIALIZED_DIR")")"
    if (( free_gb < MIN_MATERIALIZE_FREE_GB )); then
      fail "not materializing: only ${free_gb}GB free, need at least ${MIN_MATERIALIZE_FREE_GB}GB"
    fi
    log "materializing packed artifact to $MATERIALIZED_DIR"
    "$VENV/bin/python" experiments/qwen_agentworld/materialize_parent_pack.py \
      --manifest "$MANIFEST" \
      --model-dir "$BASE_MODEL_DIR" \
      --out-dir "$MATERIALIZED_DIR" \
      --report "$RUN_DIR/materialize_report.json" \
      2>&1 | tee "$RUN_DIR/materialize.log"
  fi
  MODEL_DIR="$MATERIALIZED_DIR"
  PACKED_RUNTIME_MANIFEST_VALUE=""
else
  require_dir "$BASE_MODEL_DIR"
  MODEL_DIR="$BASE_MODEL_DIR"
  PACKED_RUNTIME_MANIFEST_VALUE="$MANIFEST"
fi

if [[ "$SERVE" != "1" ]]; then
  log "SERVE=$SERVE, stopping after artifact inspection"
  exit 0
fi

log "starting short-context Transformers smoke server mode=$MODE model_dir=$MODEL_DIR max_model_len=$MAX_MODEL_LEN"
SERVER_LOG="$RUN_DIR/server.log"
SERVER_PID="$RUN_DIR/server.pid"
rm -f "$SERVER_PID"

env_args=(
  VENV="$VENV"
  MODEL_DIR="$MODEL_DIR"
  PORT="$PORT"
  MAX_MODEL_LEN="$MAX_MODEL_LEN"
)

if [[ -n "$PACKED_RUNTIME_MANIFEST_VALUE" ]]; then
  env_args+=(
    PACKED_RUNTIME_MANIFEST="$PACKED_RUNTIME_MANIFEST_VALUE"
    PACKED_RUNTIME_CACHE="${PACKED_RUNTIME_CACHE:-active_lazy}"
    PACKED_RUNTIME_ACTIVE_CACHE_LIMIT="${PACKED_RUNTIME_ACTIVE_CACHE_LIMIT:-4}"
    PACKED_RUNTIME_ACTIVE_CPU_CACHE_LIMIT="${PACKED_RUNTIME_ACTIVE_CPU_CACHE_LIMIT:-0}"
    PACKED_RUNTIME_REPORT="$RUN_DIR/packed_runtime_report.json"
  )
fi

nohup env "${env_args[@]}" \
  bash experiments/qwen_agentworld/serve_transformers_openai.sh \
  > "$SERVER_LOG" 2>&1 &
echo $! > "$SERVER_PID"

for i in $(seq 1 90); do
  if curl -fsS "http://$HOST:$PORT/v1/models" > "$RUN_DIR/models.json" 2>/dev/null; then
    log "server ready after poll $i"
    break
  fi
  if ! kill -0 "$(cat "$SERVER_PID")" 2>/dev/null; then
    log "server exited before readiness"
    tail -160 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 5
done

curl -fsS "http://$HOST:$PORT/v1/models" | tee "$RUN_DIR/models.json"
echo

log "short-context chat smoke"
curl -fsS "http://$HOST:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_DIR\",\"messages\":[{\"role\":\"user\",\"content\":\"Return exactly: clark-air-ok\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":$TEMPERATURE}" \
  | tee "$RUN_DIR/smoke_chat.json"
echo

log "post-smoke gpu snapshot"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader || true

log "done; server pid $(cat "$SERVER_PID") is still running on $HOST:$PORT"
