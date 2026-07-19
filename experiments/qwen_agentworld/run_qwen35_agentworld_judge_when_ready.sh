#!/usr/bin/env bash
set -euo pipefail

# Wait for each prediction set, then run the official AgentWorld judge
# prompt/parser through the resumable streaming judge wrapper. BF16 is judged
# as soon as it is ready so that judging can overlap the Clark Air GPU pass.

BF16_RUN="${BF16_RUN:-qwen35_9b_bf16_full_vllm_262k_32768_20260628}"
PACKED_RUN="${PACKED_RUN:-qwen35_9b_clark_air_q2_0_g128_round4_materialized_full_vllm_262k_32768_20260628}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
CLARK_DIR="${CLARK_DIR:-/workspace/clark-quantize}"
EVAL_VENV="${EVAL_VENV:-/workspace/agentworld-venv}"
EXPECTED_ROWS="${EXPECTED_ROWS:-2170}"
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
    if (( rows >= EXPECTED_ROWS )); then
      log "$label predictions complete: $rows/$EXPECTED_ROWS rows"
      return 0
    fi
    log "waiting for $label predictions: $rows/$EXPECTED_ROWS rows"
    sleep 120
  done
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
    log "$label degeneracy gate failed: collapsed=$collapsed/$count fraction=$collapsed_fraction max=$MAX_COLLAPSED_FRACTION report=$report"
    return 1
  fi
  log "$label degeneracy gate passed: collapsed=$collapsed/$count fraction=$collapsed_fraction report=$report"
}

run_judge_for() {
  local label="$1"
  local predictions="$2"
  local output_dir="$3"
  local log_path="$4"
  local pid_path="$5"
  local baseline="${6:-}"

  mkdir -p "$output_dir"
  local existing=0
  if [[ -f "$output_dir/judged.jsonl" ]]; then
    existing=$(wc -l < "$output_dir/judged.jsonl")
  fi
  if (( existing >= EXPECTED_ROWS )); then
    log "$label judged rows already complete: $existing/$EXPECTED_ROWS"
    return 0
  fi
  if [[ -f "$pid_path" ]] && kill -0 "$(cat "$pid_path")" 2>/dev/null; then
    local running_pid
    running_pid=$(cat "$pid_path")
    local running_cmd
    running_cmd=$(tr '\0' ' ' < "/proc/$running_pid/cmdline" 2>/dev/null || true)
    if [[ "$running_cmd" == *"--predictions $predictions"* ]]; then
      log "$label judge already running pid $running_pid"
      return 0
    fi
    log "$label judge pid $running_pid is for another predictions file; waiting before full judge"
    while kill -0 "$running_pid" 2>/dev/null; do
      sleep 30
    done
    existing=0
    if [[ -f "$output_dir/judged.jsonl" ]]; then
      existing=$(wc -l < "$output_dir/judged.jsonl")
    fi
    if (( existing >= EXPECTED_ROWS )); then
      log "$label judged rows already complete after waiting: $existing/$EXPECTED_ROWS"
      return 0
    fi
  fi

  if ! run_degeneracy_gate "$label" "$predictions" "$output_dir" "$baseline"; then
    log "not launching $label judge because cheap collapse gate failed"
    return 2
  fi

  log "starting $label judge with $JUDGE_MODEL; existing=$existing/$EXPECTED_ROWS"
  nohup env \
    "$EVAL_VENV/bin/python" "$STREAM_JUDGE" \
    --eval-dir /workspace/Qwen-AgentWorld/eval \
    --predictions "$predictions" \
    --output-dir "$output_dir" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --max-tokens "$JUDGE_MAX_TOKENS" \
    --temperature "$JUDGE_TEMPERATURE" \
    --concurrency "$JUDGE_CONCURRENCY" \
    > "$log_path" 2>&1 &
  echo $! > "$pid_path"
  log "$label judge launched pid $(cat "$pid_path")"
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

wait_for_rows BF16 "$BF16_PREDICTIONS"
run_judge_for \
  BF16 \
  "$BF16_PREDICTIONS" \
  "$RESULT_ROOT/$BF16_RUN/judge_qwen35_flash" \
  "$RESULT_ROOT/$BF16_RUN.judge_qwen35_flash.log" \
  "$RESULT_ROOT/$BF16_RUN.judge_qwen35_flash.pid"

wait_for_rows "Clark Air" "$PACKED_PREDICTIONS"
run_judge_for \
  "Clark Air" \
  "$PACKED_PREDICTIONS" \
  "$RESULT_ROOT/$PACKED_RUN/judge_qwen35_flash" \
  "$RESULT_ROOT/$PACKED_RUN.judge_qwen35_flash.log" \
  "$RESULT_ROOT/$PACKED_RUN.judge_qwen35_flash.pid" \
  "$BF16_PREDICTIONS"
