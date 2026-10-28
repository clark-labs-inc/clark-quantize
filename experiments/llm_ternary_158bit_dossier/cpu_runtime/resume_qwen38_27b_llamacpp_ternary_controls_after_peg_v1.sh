#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-tq-controls-v1}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
amendment="$runtime/qwen38_27b_llamacpp_ternary_controls_scoring_amendment_v2.json"
adapter_dir="$runtime/scoring_adapter_v1"
original_runner="$runtime/run_qwen38_27b_llamacpp_ternary_controls_v1.sh"
main_state=${MAIN_STATE_DIR:-/workspace/quant-state/qwen38-27b-llamacpp-ternary-controls-v1}
eval_state=${TQ1_EVAL_STATE_DIR:-/workspace/quant-state/qwen38-27b-tq1_0-control-v1}
eval_run=${TQ1_EVAL_RUN_DIR:-/workspace/quant-runs/qwen38-27b-tq1_0-control-v1}
failed_watch=${FAILED_WATCH_STATE_DIR:-/workspace/quant-state/qwen38-27b-llamacpp-ternary-controls-watch-v1}
resume_state=${RESUME_STATE_DIR:-/workspace/quant-state/qwen38-27b-llamacpp-ternary-controls-peg-resume-v1}
attempt="$resume_state/attempt1-peg-native-error"
result="$main_state/receipt.json"
tq1_model=${TQ1_MODEL:-/workspace/models/controls/Qwen3.8-27B-TQ1_0.gguf}
tq1_sha=68896f948b0ac299e8f1e67101fffcc84f04eda2d716fac15f78c52008b645d3
tq1_bytes=10243615104
server_log="$eval_run/server-qwen38-27b-tq1_0-v1.log"
evaluator_log="$eval_run/cute.log"
events="$resume_state/peg-native-scoring-events.jsonl"

require_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

mkdir -p "$resume_state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "PEG scoring continuation requires a clean source checkout" >&2
  exit 73
fi

"$python_bin" - "$repo" "$amendment" <<'PY'
import hashlib, json, sys
from pathlib import Path
repo, amendment = map(Path, sys.argv[1:])
payload = json.loads(amendment.read_text())
assert payload["post_observation_amendment"] is True
assert payload["continuation_authorized"] is True
assert payload["observation"]["first_attempt"]["infrastructure_invalid"] is False
assert payload["scoring_clarification"]["promotability_increased"] is False
assert payload["scoring_clarification"]["thresholds_changed"] is False
for label, descriptor in payload["frozen_sources"].items():
    path = repo / descriptor["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != descriptor["sha256"]:
        raise SystemExit(f"changed scoring-continuation source: {label}")
PY

if [[ "$(stat -c %s "$tq1_model")" != "$tq1_bytes" ]]; then
  echo "authenticated TQ1 artifact has an unexpected byte count" >&2
  exit 2
fi
require_sha256 "$tq1_model" "$tq1_sha"

if [[ ! -e "$attempt/ATTEMPT_ARCHIVED" ]]; then
  require_sha256 "$server_log" 187b4c258de7e3aee468582551e36df63a819de754c6905e45905acc6584cd22
  require_sha256 "$evaluator_log" ea66da489c104f69569d734afa14e0d65ab868ba194db68edc2e7c4cd760196c
  if ! grep -Fq 'unparsed peg-native output:' "$server_log"; then
    echo "initial server log lacks the observed model-originated PEG failure" >&2
    exit 2
  fi
  if ! grep -Fq 'HTTP Error 500: Internal Server Error' "$evaluator_log"; then
    echo "initial evaluator log lacks the observed HTTP 500" >&2
    exit 2
  fi
  mkdir -p "$attempt/eval-state" "$attempt/eval-run" "$attempt/watch-state" "$attempt/main-state"
  cp -a "$eval_state"/. "$attempt/eval-state/"
  cp -a "$eval_run"/. "$attempt/eval-run/"
  cp -a "$failed_watch"/. "$attempt/watch-state/"
  cp -a "$main_state"/. "$attempt/main-state/"
  "$python_bin" - "$attempt" "$tq1_model" "$tq1_sha" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
model, model_sha = sys.argv[2:]
files = {}
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name not in {"receipt.json", "ATTEMPT_ARCHIVED"}:
        content = path.read_bytes()
        files[str(path.relative_to(root))] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
payload = {
    "schema": "qwen38_27b_tq1_0_peg_native_attempt_v1",
    "archived_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "format": "TQ1_0",
    "artifact": model,
    "artifact_sha256": model_sha,
    "load_smoke_passed": True,
    "attempted_cute_items": 1,
    "scored_cute_items": 0,
    "model_originated_capability_failure_observed": True,
    "infrastructure_invalid": False,
    "suite_complete": False,
    "terminal_scientific_result": False,
    "reason": "the generated output violated the model's PEG-native grammar and the frozen scorer aborted on llama.cpp HTTP 500",
    "artifacts": files,
}
temporary = root / "receipt.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "receipt.json")
(root / "ATTEMPT_ARCHIVED").write_text("archived\n")
PY
fi

if [[ -e "$events" ]] && [[ ! -s "$result" ]]; then
  echo "unmarked PEG scoring events already exist and require inspection: $events" >&2
  exit 76
fi

git rev-parse HEAD > "$resume_state/repository.commit"
date -u +%FT%TZ > "$resume_state/started-at.txt"
sha256sum "$amendment" "$adapter_dir/sitecustomize.py" "$original_runner" \
  "$attempt/receipt.json" > "$resume_state/source.sha256"
free -b > "$resume_state/memory-before.txt"
df -B1 /workspace > "$resume_state/disk-before.txt"

PYTHONPATH="$adapter_dir${PYTHONPATH:+:$PYTHONPATH}" \
CLARK_SCORE_PEG_NATIVE_HTTP_500_AS_INCORRECT=1 \
CLARK_PEG_ERROR_EVENTS="$events" \
REPO="$repo" PYTHON_BIN="$python_bin" STATE_DIR="$main_state" \
  bash "$original_runner"

free -b > "$resume_state/memory-after.txt"
df -B1 /workspace > "$resume_state/disk-after.txt"
date -u +%FT%TZ > "$resume_state/finished-at.txt"
if [[ ! -s "$result" ]]; then
  echo "scoring continuation exited without the terminal controls receipt" >&2
  exit 70
fi

"$python_bin" - "$resume_state" "$result" "$events" "$amendment" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
state, result, events, amendment = map(Path, sys.argv[1:])
content = result.read_bytes()
payload = json.loads(content)
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("TQ controls result is not terminal")
if not events.is_file():
    raise SystemExit("missing PEG-native scoring event ledger")
event_rows = [json.loads(line) for line in events.read_text().splitlines() if line]
if not event_rows or not all(row.get("scored_as_incorrect") is True for row in event_rows):
    raise SystemExit("invalid PEG-native scoring event ledger")
receipt = {
    "schema": "qwen38_27b_llamacpp_ternary_controls_peg_resume_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "complete": True,
    "result_receipt": str(result),
    "result_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "execution_amendment": str(amendment),
    "execution_amendment_sha256": hashlib.sha256(amendment.read_bytes()).hexdigest(),
    "initial_attempt_receipt": str(state / "attempt1-peg-native-error/receipt.json"),
    "initial_attempt_receipt_sha256": hashlib.sha256(
        (state / "attempt1-peg-native-error/receipt.json").read_bytes()
    ).hexdigest(),
    "peg_native_errors_scored_incorrect": len(event_rows),
    "peg_native_event_ledger": str(events),
    "peg_native_event_ledger_sha256": hashlib.sha256(events.read_bytes()).hexdigest(),
    "thresholds_changed": False,
    "claim_boundary": "post-observation conservative scorer continuation; scientific claims remain in the original control result and linked evaluation receipts",
}
temporary = state / "receipt.json.tmp"
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, state / "receipt.json")
(state / "RESUME_TERMINAL").write_text("terminal\n")
PY
