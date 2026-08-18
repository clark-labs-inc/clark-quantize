#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-binary-low-rank-v1}
bundle=${BUNDLE:?BUNDLE must name the frozen binary-low-rank bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-infrastructure-retry-v1.service}
parent_receipt=${PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-infrastructure-retry-v1/receipt.json}
model=${MODEL_DIR:-/workspace/models/Qwen3.8-27B}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-binary-low-rank-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-binary-low-rank-v1.json}
artifact=${FACTOR_ARTIFACT:-/workspace/quant-state/qwen38-27b-binary-low-rank-v1.pt}
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
  echo "authenticated binary-low-rank receipt already exists"
  exit 0
fi

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen binary-low-rank bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty binary-low-rank repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed binary-low-rank commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_binary_low_rank_residual_preregistration_v1.json" \
  "$runtime/run_qwen38_27b_binary_low_rank_residual_v1.py" \
  clark_quantize/qwen38_binary_low_rank_residual.py \
  clark_quantize/test_qwen38_binary_low_rank_residual.py \
  clark_quantize/test_qwen38_binary_low_rank_preregistration.py \
  > "$state/source.sha256"

"$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_binary_low_rank_residual.py \
  clark_quantize/test_qwen38_binary_low_rank_preregistration.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"

while :; do
  conflicts=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^llama-(server|cli|bench|perplexi)/ ||
     ($2 ~ /^(python|python3|torchrun)$/ &&
      $0 ~ /((run|train|collect|convert).*qwen38|qwen38.*(run|train|collect|convert))/) {print}')
  available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if [[ -z "$conflicts" ]] && (( available_kib >= 12 * 1024 * 1024 )); then
    break
  fi
  sleep 30
done
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "swap is active during binary-low-rank admission" >&2
  exit 74
fi
if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
  echo "less than 10 GiB disk is free for binary-low-rank screen" >&2
  exit 75
fi

free -b > "$state/memory-before.txt"
df -B1 /workspace > "$state/disk-before.txt"
set +e
taskset -c 0-7 env PYTHONPATH="$repo" "$python_bin" \
  "$runtime/run_qwen38_27b_binary_low_rank_residual_v1.py" \
  --model-dir "$model" \
  --out "$result" \
  --artifact "$artifact" \
  --threads 8 \
  > "$state/run.stdout" 2> "$state/run.stderr"
run_status=$?
set -e
if [[ ! -s "$result" ]]; then
  echo "binary-low-rank screen exited without a terminal receipt" >&2
  exit 70
fi
if (( run_status != 0 && run_status != 2 )); then
  echo "binary-low-rank screen had unexpected exit $run_status" >&2
  exit "$run_status"
fi
"$python_bin" - "$result" "$run_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("binary-low-rank receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("binary-low-rank process status and gate disagree")
PY
if (( run_status == 0 )); then
  printf 'pass\n' > "$state/BINARY_LOW_RANK_PASS"
else
  printf 'valid rejection\n' > "$state/BINARY_LOW_RANK_REJECTED"
fi
free -b > "$state/memory-after.txt"
df -B1 /workspace > "$state/disk-after.txt"

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
    "schema": "qwen38_27b_binary_low_rank_watch_receipt_v1",
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
