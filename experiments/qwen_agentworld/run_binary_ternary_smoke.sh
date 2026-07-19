#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 RUN_NAME POLICY_JSON|none" >&2
  echo "example: $0 smoke_1x7_moe_ternary_stage256_no_think_64 /workspace/agentworld_runs/policy_qwen_agentworld_moe_ternary_stage256.json" >&2
  exit 2
fi

RUN_NAME="$1"
POLICY="$2"

VENV="${VENV:-/workspace/agentworld-venv}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
DATA_DIR="${DATA_DIR:-/workspace/AgentWorldBench_smoke_1x7}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen-AgentWorld-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen-AgentWorld-35B-A3B}"
BASELINE_PREDICTIONS="${BASELINE_PREDICTIONS-/workspace/agentworld_runs/smoke_1x7_bf16_8192_no_think_64/predictions.jsonl}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.6}"
FAKE_QUANT_BUCKETS="${FAKE_QUANT_BUCKETS:-moe_expert}"
FAKE_QUANT_RATES="${FAKE_QUANT_RATES:-ternary40_log8}"
FAKE_QUANT_INCLUDE_EXPERTS="${FAKE_QUANT_INCLUDE_EXPERTS:-1}"
RUN_JUDGE="${RUN_JUDGE:-0}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen/qwen3.5-flash-02-23}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"
MIN_SEQUENCE_RATIO="${MIN_SEQUENCE_RATIO:-0.25}"
MAX_COLLAPSED_FRACTION="${MAX_COLLAPSED_FRACTION:-0.0}"

OUT_DIR="$RESULT_ROOT/$RUN_NAME"
REPORT_DIR="$RESULT_ROOT/qwen_binary_ternary_runs"
SERVER_LOG="$REPORT_DIR/${RUN_NAME}.serve.log"
SERVER_PID="$REPORT_DIR/${RUN_NAME}.serve.pid"
APPLY_REPORT="$REPORT_DIR/${RUN_NAME}.fake_quant_apply.json"
DEGENERACY_JSON="$REPORT_DIR/${RUN_NAME}.degeneracy.json"
COMPARE_JSON="$REPORT_DIR/${RUN_NAME}.compare_bf16.json"
SUMMARY_JSON="$REPORT_DIR/${RUN_NAME}.summary.json"

mkdir -p "$OUT_DIR" "$REPORT_DIR"
USE_FAKE_QUANT=1
if [ "$POLICY" = "none" ] || [ "$POLICY" = "-" ] || [ "$POLICY" = "bf16" ]; then
  USE_FAKE_QUANT=0
fi

cleanup() {
  if [ -f "$SERVER_PID" ]; then
    local pid
    pid="$(cat "$SERVER_PID" 2>/dev/null || true)"
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

cd /workspace/clark-quantize

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "port ${PORT} already has a model endpoint; refusing to reuse it implicitly" >&2
  exit 3
fi

echo "[run_binary_ternary_smoke] starting endpoint for ${RUN_NAME}"
if [ "$USE_FAKE_QUANT" = "1" ]; then
  PORT="$PORT" \
  MODEL_DIR="$MODEL_DIR" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  FAKE_QUANT_POLICY="$POLICY" \
  FAKE_QUANT_BUCKETS="$FAKE_QUANT_BUCKETS" \
  FAKE_QUANT_RATES="$FAKE_QUANT_RATES" \
  FAKE_QUANT_INCLUDE_EXPERTS="$FAKE_QUANT_INCLUDE_EXPERTS" \
  FAKE_QUANT_REPORT="$APPLY_REPORT" \
  nohup bash experiments/qwen_agentworld/serve_transformers_openai.sh > "$SERVER_LOG" 2>&1 &
else
  APPLY_REPORT=""
  PORT="$PORT" \
  MODEL_DIR="$MODEL_DIR" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  nohup bash experiments/qwen_agentworld/serve_transformers_openai.sh > "$SERVER_LOG" 2>&1 &
fi
echo $! > "$SERVER_PID"

echo "[run_binary_ternary_smoke] waiting for endpoint"
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$(cat "$SERVER_PID")" 2>/dev/null; then
    echo "server exited before readiness; tailing log" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 4
  fi
  sleep 10
done
if [ "${ready:-0}" != "1" ]; then
  echo "endpoint did not become ready" >&2
  tail -120 "$SERVER_LOG" >&2 || true
  exit 4
fi

echo "[run_binary_ternary_smoke] running official inference"
DATA_DIR="$DATA_DIR" \
MODEL_NAME="$MODEL_NAME" \
MAX_TOKENS="$MAX_TOKENS" \
TEMPERATURE="$TEMPERATURE" \
bash experiments/qwen_agentworld/run_official_eval.sh infer "$RUN_NAME"

echo "[run_binary_ternary_smoke] running cheap gates"
degeneracy_args=(
  experiments/qwen_agentworld/detect_generation_degeneracy.py
  --predictions "$OUT_DIR/predictions.jsonl"
  --out "$DEGENERACY_JSON"
)
if [ -n "$BASELINE_PREDICTIONS" ] && [ -f "$BASELINE_PREDICTIONS" ]; then
  degeneracy_args+=(--baseline "$BASELINE_PREDICTIONS")
fi
"$VENV/bin/python" "${degeneracy_args[@]}"

if [ -n "$BASELINE_PREDICTIONS" ] && [ -f "$BASELINE_PREDICTIONS" ]; then
  "$VENV/bin/python" experiments/qwen_agentworld/compare_predictions.py \
    --baseline "$BASELINE_PREDICTIONS" \
    --candidate "$OUT_DIR/predictions.jsonl" \
    --out "$COMPARE_JSON"
else
  "$VENV/bin/python" - "$COMPARE_JSON" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "summary": {
        "count": 0,
        "baseline_empty": 0,
        "candidate_empty": 0,
        "exact_matches": 0,
        "avg_sequence_ratio": 1.0,
    },
    "rows": [],
}, indent=2) + "\n")
PY
fi

gate_status="$("$VENV/bin/python" - "$DEGENERACY_JSON" "$COMPARE_JSON" "$MIN_SEQUENCE_RATIO" "$MAX_COLLAPSED_FRACTION" <<'PY'
import json
import sys
from pathlib import Path

deg = json.loads(Path(sys.argv[1]).read_text())
cmp = json.loads(Path(sys.argv[2]).read_text())
min_ratio = float(sys.argv[3])
max_collapsed = float(sys.argv[4])

collapsed = float(deg["summary"]["collapsed_fraction"])
ratio = float(cmp["summary"]["avg_sequence_ratio"])
passed = collapsed <= max_collapsed and ratio >= min_ratio
print(json.dumps({
    "passed": passed,
    "collapsed_fraction": collapsed,
    "avg_sequence_ratio": ratio,
    "min_sequence_ratio": min_ratio,
    "max_collapsed_fraction": max_collapsed,
}))
PY
)"
echo "$gate_status"

judge_score_path="$OUT_DIR/score_openrouter_qwen35_flash.txt"
if [ "$RUN_JUDGE" = "1" ]; then
  gate_passed="$("$VENV/bin/python" - "$gate_status" <<'PY'
import json, sys
print("1" if json.loads(sys.argv[1])["passed"] else "0")
PY
)"
  if [ "$gate_passed" = "1" ]; then
    : "${JUDGE_API_KEY:?set JUDGE_API_KEY to run paid judge}"
    echo "[run_binary_ternary_smoke] cheap gates passed; running judge ${JUDGE_MODEL}"
    JUDGE_API_KEY="$JUDGE_API_KEY" \
    JUDGE_BASE_URL="$JUDGE_BASE_URL" \
    JUDGE_MODEL="$JUDGE_MODEL" \
    bash experiments/qwen_agentworld/run_official_eval.sh judge "$RUN_NAME"
    JUDGE_API_KEY="$JUDGE_API_KEY" \
    JUDGE_BASE_URL="$JUDGE_BASE_URL" \
    JUDGE_MODEL="$JUDGE_MODEL" \
    bash experiments/qwen_agentworld/run_official_eval.sh score "$RUN_NAME" > "$judge_score_path"
    cat "$judge_score_path"
  else
    echo "[run_binary_ternary_smoke] cheap gates failed; skipping paid judge"
  fi
fi

"$VENV/bin/python" - "$SUMMARY_JSON" "$RUN_NAME" "$POLICY" "$APPLY_REPORT" "$DEGENERACY_JSON" "$COMPARE_JSON" "$gate_status" "$RUN_JUDGE" "$judge_score_path" <<'PY'
import json
import sys
from pathlib import Path

summary_path, run_name, policy, apply_report, degeneracy, compare, gate_json, run_judge, score_path = sys.argv[1:]
score = Path(score_path)
payload = {
    "run_name": run_name,
    "policy": policy,
    "apply_report": apply_report,
    "predictions": f"/workspace/agentworld_runs/{run_name}/predictions.jsonl",
    "degeneracy": degeneracy,
    "compare_bf16": compare,
    "cheap_gate": json.loads(gate_json),
    "judge_requested": run_judge == "1",
    "judge_score": str(score) if score.exists() else None,
}
Path(summary_path).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
