#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-tq-controls-v1}
bundle=${BUNDLE:?BUNDLE must name the frozen TQ-controls bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-binary-low-rank-residual-v1.service}
parent_receipt=${PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-binary-low-rank-watch-v1/receipt.json}
state=${WATCH_STATE_DIR:-/workspace/quant-state/qwen38-27b-llamacpp-ternary-controls-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-llamacpp-ternary-controls-v1/receipt.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

# shellcheck source=wait_for_qwen38_terminal_receipt.sh
source "$(dirname "$0")/wait_for_qwen38_terminal_receipt.sh"
qwen38_wait_for_terminal_receipt "$parent_unit" "$parent_receipt" 30

if [[ -s "$result" ]] && "$python_bin" - "$result" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("terminal") is True else 1)
PY
then
  echo "authenticated TQ-controls receipt already exists"
  exit 0
fi

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen TQ-controls bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty TQ-controls repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed TQ-controls commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_llamacpp_ternary_controls_preregistration_v1.json" \
  "$runtime/run_qwen38_27b_llamacpp_ternary_controls_v1.sh" \
  "$runtime/eval_qwen38_27b_llamacpp_ternary_control_v1.sh" \
  clark_quantize/test_qwen38_llamacpp_ternary_controls_preregistration.py \
  > "$state/source.sha256"

"$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_llamacpp_ternary_controls_preregistration.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"

while :; do
  conflicts=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^llama-(server|cli|bench|perplexi|quantize)/ ||
     ($2 ~ /^(python|python3|torchrun)$/ && $0 ~ /qwen38.*(run|train|collect|convert)/) {print}')
  available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if [[ -z "$conflicts" ]] && (( available_kib >= 12 * 1024 * 1024 )); then
    break
  fi
  sleep 30
done

free -b > "$state/memory-before.txt"
df -B1 /workspace > "$state/disk-before.txt"
REPO="$repo" PYTHON_BIN="$python_bin" \
  "$runtime/run_qwen38_27b_llamacpp_ternary_controls_v1.sh" \
  > "$state/run.stdout" 2> "$state/run.stderr"
free -b > "$state/memory-after.txt"
df -B1 /workspace > "$state/disk-after.txt"

if [[ ! -s "$result" ]]; then
  echo "TQ-controls runner exited without a terminal receipt" >&2
  exit 70
fi
"$python_bin" - "$state" "$result" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, result = map(Path, sys.argv[1:])
content = result.read_bytes()
payload = json.loads(content)
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("TQ-controls receipt is not terminal")
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
receipt = {
    "schema": "qwen38_27b_llamacpp_ternary_controls_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "result_receipt": str(result),
    "result_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "artifacts": files,
    "claim_boundary": "orchestration receipt only; control claims remain in the result receipts",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
