#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen hash storage bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-hash-storage-budget-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-hash-storage-budget-v1.json}
hash_unit=${HASH_UNIT:-quant-qwen38-27b-hash-residual-screen-v1.service}
hash_receipt=${HASH_RECEIPT:-/workspace/quant-state/qwen38-27b-hash-residual-screen-v1.json}
codec_unit=${CODEC_UNIT:-quant-qwen38-27b-fixed-support-codec-v1.service}
codec_receipt=${CODEC_RECEIPT:-/workspace/quant-state/qwen38-27b-fixed-support-codec-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

# shellcheck source=wait_for_qwen38_terminal_receipt.sh
source "$(dirname "$0")/wait_for_qwen38_terminal_receipt.sh"
qwen38_wait_for_terminal_receipt "$hash_unit" "$hash_receipt" 30
qwen38_wait_for_terminal_receipt "$codec_unit" "$codec_receipt" 30

if [[ -s "$result" ]] && "$python_bin" - "$result" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("terminal") is True else 1)
PY
then
  echo "authenticated complete hash storage receipt already exists"
  exit 0
fi

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen Qwen3.8-27B hash storage bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B hash storage repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B hash storage commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_hash_storage_budget_preregistration_v1.json" \
  "$runtime/run_qwen38_27b_hash_storage_budget_v1.py" \
  clark_quantize/qwen38_hash_storage_budget.py \
  > "$state/source.sha256"

"$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_hash_storage_budget.py \
  clark_quantize/test_qwen38_hash_storage_preregistration.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"

if ! "$python_bin" - "$hash_receipt" "$codec_receipt" <<'PY'
import json, sys
parents = [json.load(open(path)) for path in sys.argv[1:]]
raise SystemExit(0 if all(parent.get("gate", {}).get("passed") is True for parent in parents) else 1)
PY
then
  "$python_bin" - "$hash_receipt" "$codec_receipt" "$result" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
hash_parent, codec_parent, result = map(Path, sys.argv[1:])
receipt = {
    "schema": "qwen38_27b_hash_storage_budget_skip_receipt_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hash_parent_receipt_sha256": hashlib.sha256(hash_parent.read_bytes()).hexdigest(),
    "codec_parent_receipt_sha256": hashlib.sha256(codec_parent.read_bytes()).hexdigest(),
    "complete": True,
    "terminal": True,
    "valid_nonpromotion": True,
    "reason": "at least one frozen hash representation parent gate rejected",
    "complete_inventory_scanned": False,
    "gate": {"passed": False},
    "claim_boundary": "valid parent rejection only; no complete hash storage result",
}
temporary = result.with_suffix(result.suffix + ".tmp")
result.parent.mkdir(parents=True, exist_ok=True)
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
os.replace(temporary, result)
PY
  printf 'valid parent rejection\n' > "$state/PARENT_REJECTED"
else
  llama_conflicts=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^llama-(server|cli|bench|perplexi)/ {print}')
  if [[ -n "$llama_conflicts" ]]; then
    echo "another llama.cpp workload is active" >&2
    printf '%s\n' "$llama_conflicts" >&2
    exit 2
  fi
  if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((2 * 1024 * 1024)) ]]; then
    echo "less than 2 GiB is available for complete hash storage inventory" >&2
    exit 72
  fi
  if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
    echo "swap is active during hash storage admission" >&2
    exit 74
  fi
  if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
    echo "less than 10 GiB disk is free for hash storage inventory" >&2
    exit 75
  fi
  free -b > "$state/memory-before.txt"
  set +e
  taskset -c 0 env PYTHONPATH="$repo" "$python_bin" \
    "$runtime/run_qwen38_27b_hash_storage_budget_v1.py" \
    --hash-receipt "$hash_receipt" \
    --codec-receipt "$codec_receipt" \
    --out "$result" \
    > "$state/budget.stdout" 2> "$state/budget.stderr"
  budget_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "hash storage runner exited without a terminal receipt" >&2
    exit 70
  fi
  if (( budget_status != 0 && budget_status != 2 )); then
    echo "hash storage runner had unexpected exit $budget_status" >&2
    exit "$budget_status"
  fi
  "$python_bin" - "$result" "$budget_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("hash storage receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("hash storage process status and gate disagree")
PY
  if (( budget_status == 0 )); then
    printf 'pass\n' > "$state/HASH_STORAGE_PASS"
  else
    printf 'valid rejection\n' > "$state/HASH_STORAGE_REJECTED"
  fi
  free -b > "$state/memory-after.txt"
fi

"$python_bin" - "$state" "$result" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, result = map(Path, sys.argv[1:])
content = result.read_bytes()
payload = json.loads(content)
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
receipt = {
    "schema": "qwen38_27b_hash_storage_budget_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "result_receipt": str(result),
    "result_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "scientific_gate_passed": payload.get("gate", {}).get("passed") is True,
    "valid_nonpromotion": payload.get("gate", {}).get("passed") is not True,
    "artifacts": files,
    "claim_boundary": "orchestration receipt only; scientific claims remain in the result receipt",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
