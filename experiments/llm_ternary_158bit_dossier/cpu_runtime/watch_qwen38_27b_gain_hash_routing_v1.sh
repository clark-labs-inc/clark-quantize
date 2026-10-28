#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen gain-hash bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-half-support-complete-ledger-v1.service}
parent_receipt=${PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-half-support-complete-ledger-watch-v1/receipt.json}
model=${MODEL_DIR:-/workspace/models/Qwen3.8-27B}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-gain-hash-routing-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-gain-hash-routing-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

# shellcheck source=experiments/llm_ternary_158bit_dossier/cpu_runtime/wait_for_qwen38_terminal_receipt.sh
source "$(dirname "$0")/wait_for_qwen38_terminal_receipt.sh"
qwen38_wait_for_terminal_receipt "$parent_unit" "$parent_receipt" 30

if [[ -s "$result" ]] && "$python_bin" - "$result" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("terminal") is True else 1)
PY
then
  echo "authenticated gain-hash receipt already exists"
  exit 0
fi

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen gain-hash bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B gain-hash repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B gain-hash commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_gain_hash_routing_preregistration_v1.json" \
  "$runtime/run_qwen38_27b_gain_hash_routing_v1.py" \
  clark_quantize/qwen38_gain_hash_residual.py \
  clark_quantize/qwen38_learned_hash_residual.py \
  clark_quantize/qwen38_hash_residual.py \
  clark_quantize/qwen38_fixed_support_codec.py \
  clark_quantize/test_qwen38_gain_hash_residual.py \
  clark_quantize/test_qwen38_gain_hash_preregistration.py \
  > "$state/source.sha256"

llama_conflicts=$(ps -eo pid=,comm=,args= | awk \
  '$2 ~ /^llama-(server|cli|bench|perplexi)/ {print}')
if [[ -n "$llama_conflicts" ]]; then
  echo "another llama.cpp workload is active" >&2
  printf '%s\n' "$llama_conflicts" >&2
  exit 2
fi
if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((4 * 1024 * 1024)) ]]; then
  echo "less than 4 GiB is available for gain-hash screen" >&2
  exit 72
fi
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "swap is active during gain-hash admission" >&2
  exit 74
fi
if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
  echo "less than 10 GiB disk is free for gain-hash screen" >&2
  exit 75
fi

"$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_gain_hash_residual.py \
  clark_quantize/test_qwen38_gain_hash_preregistration.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"
free -b > "$state/memory-before.txt"
set +e
taskset -c 0-3 env PYTHONPATH="$repo" "$python_bin" \
  "$runtime/run_qwen38_27b_gain_hash_routing_v1.py" \
  --model-dir "$model" \
  --out "$result" \
  --threads 4 \
  > "$state/run.stdout" 2> "$state/run.stderr"
run_status=$?
set -e
if [[ ! -s "$result" ]]; then
  echo "gain-hash screen exited without a terminal receipt" >&2
  exit 70
fi
if (( run_status != 0 && run_status != 2 )); then
  echo "gain-hash screen had unexpected exit $run_status" >&2
  exit "$run_status"
fi
"$python_bin" - "$result" "$run_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("gain-hash receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("gain-hash process status and gate disagree")
PY
if (( run_status == 0 )); then
  printf 'pass\n' > "$state/GAIN_HASH_PASS"
else
  printf 'valid rejection\n' > "$state/GAIN_HASH_REJECTED"
fi
free -b > "$state/memory-after.txt"

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
    "schema": "qwen38_27b_gain_hash_routing_watch_receipt_v1",
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
