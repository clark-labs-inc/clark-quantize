#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen activation gain-hash bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-activation-gain-hash-residual-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-activation-gain-hash-residual-v1.json}
gain_unit=${GAIN_UNIT:-quant-qwen38-27b-gain-hash-routing-v1.service}
gain_receipt=${GAIN_RECEIPT:-/workspace/quant-state/qwen38-27b-gain-hash-routing-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

# shellcheck source=wait_for_qwen38_terminal_receipt.sh
source "$(dirname "$0")/wait_for_qwen38_terminal_receipt.sh"
qwen38_wait_for_terminal_receipt "$gain_unit" "$gain_receipt" 30

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen activation gain-hash bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty activation gain-hash repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed activation gain-hash commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_activation_gain_hash_residual_preregistration_v1.json" \
  "$runtime/run_qwen38_27b_activation_gain_hash_residual_v1.py" \
  "$runtime/collect_qwen38_27b_layer0_mlp_activations_v1.py" \
  clark_quantize/qwen38_activation_gain_hash_residual.py \
  clark_quantize/qwen38_gain_hash_residual.py \
  > "$state/source.sha256"

"$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_activation_gain_hash_residual.py \
  clark_quantize/test_qwen38_activation_gain_hash_preregistration.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"

if ! "$python_bin" - "$gain_receipt" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("gate", {}).get("passed") is True else 1)
PY
then
  "$python_bin" - "$state" "$gain_receipt" "$result" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
state, parent, result = map(Path, sys.argv[1:])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_activation_gain_hash_residual_skip_receipt_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_gain_receipt": str(parent),
    "parent_gain_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "complete": True,
    "terminal": True,
    "valid_nonpromotion": True,
    "reason": "the frozen real-weight gain-route gate rejected",
    "activation_cache_launched": False,
    "activation_training_launched": False,
    "gate": {"passed": False},
    "claim_boundary": "valid parent rejection only; no activation gain-hash result",
}
temporary = result.with_suffix(result.suffix + ".tmp")
result.parent.mkdir(parents=True, exist_ok=True)
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
os.replace(temporary, result)
(state / "PARENT_REJECTED").write_text("valid nonpromotion\n")
PY
else
  llama_conflicts=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^llama-(server|cli|bench|perplexi)/ {print}')
  if [[ -n "$llama_conflicts" ]]; then
    echo "another llama.cpp workload is active" >&2
    printf '%s\n' "$llama_conflicts" >&2
    exit 2
  fi
  training_conflicts=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^(python|python3|torchrun)$/ && $0 ~ /(train_mlp|qwen.*qat|activation.*cache)/ {print}')
  if [[ -n "$training_conflicts" ]]; then
    echo "another QAT or activation-cache process is active" >&2
    printf '%s\n' "$training_conflicts" >&2
    exit 2
  fi
  if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((10 * 1024 * 1024)) ]]; then
    echo "less than 10 GiB is available for activation gain-hash fitting" >&2
    exit 72
  fi
  if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
    echo "swap is active during activation gain-hash admission" >&2
    exit 74
  fi
  if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
    echo "less than 10 GiB disk is free for activation gain-hash fitting" >&2
    exit 75
  fi
  free -b > "$state/memory-before.txt"
  df -B1 /workspace > "$state/disk-before.txt"
  "$python_bin" "$runtime/collect_qwen38_27b_layer0_mlp_activations_v1.py" \
    > "$state/activation-cache.stdout" 2> "$state/activation-cache.stderr"
  printf 'complete\n' > "$state/ACTIVATION_CACHE_COMPLETE"
  set +e
  PYTHONPATH="$repo" "$python_bin" \
    "$runtime/run_qwen38_27b_activation_gain_hash_residual_v1.py" \
    --threads 4 \
    --out "$result" \
    > "$state/experiment.stdout" 2> "$state/experiment.stderr"
  experiment_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "activation gain-hash runner exited without a terminal receipt" >&2
    exit 70
  fi
  if (( experiment_status != 0 && experiment_status != 2 )); then
    echo "activation gain-hash runner had unexpected exit $experiment_status" >&2
    exit "$experiment_status"
  fi
  "$python_bin" - "$result" "$experiment_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("activation gain-hash receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("activation gain-hash process status and gate disagree")
PY
  if (( experiment_status == 0 )); then
    printf 'pass\n' > "$state/ACTIVATION_GAIN_HASH_PASS"
  else
    printf 'valid rejection\n' > "$state/ACTIVATION_GAIN_HASH_REJECTED"
  fi
  free -b > "$state/memory-after.txt"
  df -B1 /workspace > "$state/disk-after.txt"
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
    "schema": "qwen38_27b_activation_gain_hash_residual_watch_receipt_v1",
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
