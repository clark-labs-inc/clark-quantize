#!/usr/bin/env bash
set -euo pipefail

# Wait for streaming judge outputs, then run the official AgentWorld score
# aggregation for BF16 and Clark Air.

BF16_RUN="${BF16_RUN:-qwen35_9b_bf16_full_vllm_262k_32768_20260628}"
PACKED_RUN="${PACKED_RUN:-qwen35_9b_clark_air_q2_0_g128_round4_materialized_full_vllm_262k_32768_20260628}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/agentworld_runs}"
EVAL_VENV="${EVAL_VENV:-/workspace/agentworld-venv}"
EXPECTED_ROWS="${EXPECTED_ROWS:-2170}"
JUDGE_DIR_NAME="${JUDGE_DIR_NAME:-judge_qwen35_flash}"

BF16_JUDGED="$RESULT_ROOT/$BF16_RUN/$JUDGE_DIR_NAME/judged.jsonl"
PACKED_JUDGED="$RESULT_ROOT/$PACKED_RUN/$JUDGE_DIR_NAME/judged.jsonl"
BF16_SCORE="$RESULT_ROOT/$BF16_RUN/$JUDGE_DIR_NAME/score.txt"
PACKED_SCORE="$RESULT_ROOT/$PACKED_RUN/$JUDGE_DIR_NAME/score.txt"
COMPARE_JSON="$RESULT_ROOT/qwen35_9b_bf16_vs_clark_air_q2_0_g128_agentworld_compare.json"

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
      log "$label judged rows complete: $rows/$EXPECTED_ROWS"
      return 0
    fi
    log "waiting for $label judged rows: $rows/$EXPECTED_ROWS"
    sleep 120
  done
}

score_one() {
  local label="$1"
  local judged="$2"
  local score="$3"
  if [[ -s "$score" ]]; then
    log "$label score already exists: $score"
    return 0
  fi
  log "scoring $label judged rows"
  cd /workspace/Qwen-AgentWorld/eval
  "$EVAL_VENV/bin/python" eval.py score --predictions "$judged" > "$score"
  tail -30 "$score" || true
}

write_compare_json() {
  "$EVAL_VENV/bin/python" - "$BF16_SCORE" "$PACKED_SCORE" "$COMPARE_JSON" <<'PY'
import json
import re
import sys
from pathlib import Path

bf16_score, packed_score, out = map(Path, sys.argv[1:])

def parse(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    result = {
        "score_file": str(path),
        "overall": None,
        "total_samples": None,
        "valid_samples": None,
        "failed_samples": None,
        "domains": {},
        "raw_text": text,
    }
    current_domain = None
    for line in text.splitlines():
        header = re.match(
            r"---\s+(.+?)\s+\((\d+)/(\d+) valid,\s+(\d+) failed\)\s+---",
            line,
        )
        if header:
            current_domain = header.group(1)
            result["domains"][current_domain] = {
                "valid_samples": int(header.group(2)),
                "total_samples": int(header.group(3)),
                "failed_samples": int(header.group(4)),
                "scores": {},
            }
            continue

        overall = re.match(r"Overall:\s*([-+]?\d+(?:\.\d+)?)\s*$", line)
        if overall:
            result["overall"] = float(overall.group(1))
            current_domain = None
            continue

        totals = re.match(
            r"Total samples:\s*(\d+),\s*Valid:\s*(\d+),\s*Failed:\s*(\d+)",
            line,
        )
        if totals:
            result["total_samples"] = int(totals.group(1))
            result["valid_samples"] = int(totals.group(2))
            result["failed_samples"] = int(totals.group(3))
            continue

        m = re.match(r"\s*([A-Za-z_]+):\s*([-+]?\d+(?:\.\d+)?)\s*$", line)
        if m:
            if current_domain is not None:
                result["domains"][current_domain]["scores"][m.group(1)] = float(m.group(2))
    return result

bf16 = parse(bf16_score)
packed = parse(packed_score)

def score_delta(left: dict, right: dict) -> dict:
    domains = {}
    for domain in sorted(set(left["domains"]) | set(right["domains"])):
        l_scores = left["domains"].get(domain, {}).get("scores", {})
        r_scores = right["domains"].get(domain, {}).get("scores", {})
        keys = sorted(set(l_scores) | set(r_scores))
        domains[domain] = {
            key: r_scores.get(key, 0.0) - l_scores.get(key, 0.0)
            for key in keys
        }
    return {
        "overall": (
            None
            if left["overall"] is None or right["overall"] is None
            else right["overall"] - left["overall"]
        ),
        "domains": domains,
    }

out.write_text(
    json.dumps(
        {
            "bf16_score_file": str(bf16_score),
            "clark_air_score_file": str(packed_score),
            "bf16": bf16,
            "clark_air": packed,
            "clark_air_minus_bf16": score_delta(bf16, packed),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(out)
PY
}

wait_for_rows BF16 "$BF16_JUDGED"
wait_for_rows "Clark Air" "$PACKED_JUDGED"
score_one BF16 "$BF16_JUDGED" "$BF16_SCORE"
score_one "Clark Air" "$PACKED_JUDGED" "$PACKED_SCORE"
write_compare_json
log "comparison written to $COMPARE_JSON"
