#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-activation-binary-mlp-v1}
bundle=${BUNDLE:?BUNDLE must name the frozen activation-binary bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-binary-low-rank-residual-v1.service}
parent_receipt=${PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-binary-low-rank-v1.json}
cache_receipt=${CACHE_RECEIPT:-/workspace/quant-state/qwen38-27b-layer0-mlp-activation-cache-v1.json}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-activation-binary-mlp-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-activation-binary-mlp-v1.json}
artifact=${FACTOR_ARTIFACT:-/workspace/quant-state/qwen38-27b-activation-binary-mlp-v1.pt}
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
  echo "authenticated activation-binary MLP receipt already exists"
  exit 0
fi

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen activation-binary bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty activation-binary repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed activation-binary commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_materiality_priority_amendment_v2.json" \
  "$runtime/qwen38_27b_activation_binary_mlp_preregistration_v1.json" \
  "$runtime/run_qwen38_27b_activation_binary_mlp_v1.py" \
  "$runtime/collect_qwen38_27b_layer0_mlp_activations_v1.py" \
  clark_quantize/qwen38_activation_binary_mlp.py \
  clark_quantize/test_qwen38_activation_binary_mlp.py \
  clark_quantize/test_qwen38_activation_binary_preregistration.py \
  > "$state/source.sha256"

"$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_activation_binary_mlp.py \
  clark_quantize/test_qwen38_activation_binary_preregistration.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"

parent_passed=$(
  "$python_bin" - "$parent_receipt" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if (
    payload.get("schema") != "qwen38_27b_binary_low_rank_residual_receipt_v1"
    or payload.get("complete") is not True
    or payload.get("terminal") is not True
):
    raise SystemExit("binary-low-rank parent receipt is invalid")
print("true" if payload.get("gate", {}).get("passed") is True else "false")
PY
)
if [[ "$parent_passed" != true ]]; then
  "$python_bin" - "$parent_receipt" "$result" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
parent, result = map(Path, sys.argv[1:])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_activation_binary_mlp_parent_rejection_skip_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "complete": True,
    "terminal": True,
    "valid_nonpromotion": True,
    "activation_cache_launched": False,
    "activation_fit_launched": False,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "reason": "the frozen rank-40 full gate-up weight-space gate rejected",
    "gate": {"passed": False},
    "claim_boundary": "terminal dependency skip; not activation evidence",
}
temporary = result.with_suffix(result.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, result)
PY
  printf 'valid parent rejection\n' > "$state/ACTIVATION_BINARY_SKIPPED"
else
  while :; do
    conflicts=$(ps -eo pid=,comm=,args= | awk \
      '$2 ~ /^llama-(server|cli|bench|perplexi|quantize)/ ||
       ($2 ~ /^(python|python3|torchrun)$/ &&
        $0 ~ /((run|train|collect|convert).*qwen38|qwen38.*(run|train|collect|convert))/) {print}')
    available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [[ -z "$conflicts" ]] && (( available_kib >= 16 * 1024 * 1024 )); then
      break
    fi
    sleep 30
  done
  if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
    echo "swap is active during activation-binary admission" >&2
    exit 74
  fi
  if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
    echo "less than 10 GiB disk is free for activation-binary fitting" >&2
    exit 75
  fi
  free -b > "$state/memory-before.txt"
  df -B1 /workspace > "$state/disk-before.txt"

  PYTHONPATH="$repo" taskset -c 0-7 "$python_bin" \
    "$runtime/collect_qwen38_27b_layer0_mlp_activations_v1.py" \
    > "$state/activation-cache.stdout" 2> "$state/activation-cache.stderr"
  if [[ ! -s "$cache_receipt" ]]; then
    echo "activation-cache collector exited without a receipt" >&2
    exit 70
  fi
  printf 'complete\n' > "$state/ACTIVATION_CACHE_COMPLETE"

  set +e
  PYTHONPATH="$repo" taskset -c 0-7 "$python_bin" \
    "$runtime/run_qwen38_27b_activation_binary_mlp_v1.py" \
    --parent-receipt "$parent_receipt" \
    --activation-cache-receipt "$cache_receipt" \
    --out "$result" \
    --artifact "$artifact" \
    --threads 8 \
    > "$state/run.stdout" 2> "$state/run.stderr"
  run_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "activation-binary runner exited without a terminal receipt" >&2
    exit 70
  fi
  if (( run_status != 0 && run_status != 2 )); then
    echo "activation-binary runner had unexpected exit $run_status" >&2
    exit "$run_status"
  fi
  "$python_bin" - "$result" "$run_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("activation-binary receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("activation-binary process status and gate disagree")
PY
  if (( run_status == 0 )); then
    printf 'pass\n' > "$state/ACTIVATION_BINARY_PASS"
  else
    printf 'valid rejection\n' > "$state/ACTIVATION_BINARY_REJECTED"
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
    "schema": "qwen38_27b_activation_binary_mlp_watch_receipt_v1",
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

