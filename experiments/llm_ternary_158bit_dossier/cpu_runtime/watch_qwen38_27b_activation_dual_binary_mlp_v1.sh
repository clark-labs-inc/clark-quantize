#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-activation-dual-v1}
bundle=${BUNDLE:?BUNDLE must name the frozen activation-dual bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-activation-binary-mlp-v1.service}
parent_watch_receipt=${PARENT_WATCH_RECEIPT:-/workspace/quant-state/qwen38-27b-activation-binary-mlp-watch-v1/receipt.json}
weight_parent_receipt=${WEIGHT_PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-binary-low-rank-v1.json}
frozen_right_receipt=${FROZEN_RIGHT_RECEIPT:-/workspace/quant-state/qwen38-27b-activation-binary-mlp-v1.json}
cache_receipt=${CACHE_RECEIPT:-/workspace/quant-state/qwen38-27b-layer0-mlp-activation-cache-v1.json}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-activation-dual-binary-mlp-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-activation-dual-binary-mlp-v1.json}
artifact=${FACTOR_ARTIFACT:-/workspace/quant-state/qwen38-27b-activation-dual-binary-mlp-v1.pt}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

# shellcheck source=wait_for_qwen38_terminal_receipt.sh
source "$(dirname "$0")/wait_for_qwen38_terminal_receipt.sh"
qwen38_wait_for_terminal_receipt "$parent_unit" "$parent_watch_receipt" 30

if [[ -s "$result" ]] && "$python_bin" - "$result" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("terminal") is True else 1)
PY
then
  echo "authenticated activation-dual MLP receipt already exists"
  exit 0
fi

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen activation-dual bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty activation-dual repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed activation-dual commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_activation_dual_binary_method_preregistration_v1.json" \
  "$runtime/qwen38_27b_activation_dual_binary_execution_amendment_v1.json" \
  "$runtime/run_qwen38_27b_activation_dual_binary_mlp_v1.py" \
  "$runtime/watch_qwen38_27b_activation_dual_binary_mlp_v1.sh" \
  "$runtime/quant-qwen38-27b-activation-dual-binary-mlp-v1.service" \
  clark_quantize/qwen38_activation_dual_binary.py \
  clark_quantize/test_qwen38_activation_dual_binary.py \
  clark_quantize/test_qwen38_activation_dual_preregistration.py \
  clark_quantize/test_qwen38_activation_dual_execution.py \
  > "$state/source.sha256"

"$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_activation_dual_binary.py \
  clark_quantize/test_qwen38_activation_dual_preregistration.py \
  clark_quantize/test_qwen38_activation_dual_execution.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"

decision=$(
  "$python_bin" - \
    "$parent_watch_receipt" \
    "$weight_parent_receipt" \
    "$frozen_right_receipt" <<'PY'
import hashlib, json, sys
from pathlib import Path

watch_path, weight_path, frozen_path = map(Path, sys.argv[1:])
watch_content = watch_path.read_bytes()
weight_content = weight_path.read_bytes()
frozen_content = frozen_path.read_bytes()
watch = json.loads(watch_content)
weight = json.loads(weight_content)
frozen = json.loads(frozen_content)
if (
    watch.get("schema") != "qwen38_27b_activation_binary_mlp_watch_receipt_v1"
    or watch.get("terminal") is not True
    or watch.get("result_receipt_sha256")
    != hashlib.sha256(frozen_content).hexdigest()
):
    raise SystemExit("activation-binary parent watch receipt is invalid")
if (
    weight.get("schema") != "qwen38_27b_binary_low_rank_residual_receipt_v1"
    or weight.get("complete") is not True
    or weight.get("terminal") is not True
):
    raise SystemExit("weight-binary parent receipt is invalid")
if weight.get("gate", {}).get("passed") is not True:
    print("skip")
elif (
    frozen.get("schema") == "qwen38_27b_activation_binary_mlp_receipt_v1"
    and frozen.get("complete") is True
    and frozen.get("terminal") is True
):
    print("run")
elif frozen.get("complete") is True and frozen.get("terminal") is True:
    print("skip")
else:
    raise SystemExit("activation-binary scientific receipt is nonterminal")
PY
)

if [[ "$decision" == skip ]]; then
  "$python_bin" - \
    "$parent_watch_receipt" \
    "$weight_parent_receipt" \
    "$frozen_right_receipt" \
    "$result" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

watch, weight, frozen, result = map(Path, sys.argv[1:])
receipt = {
    "schema": "qwen38_27b_activation_dual_binary_dependency_skip_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "complete": True,
    "terminal": True,
    "valid_nonpromotion": True,
    "activation_dual_fit_launched": False,
    "parents": {
        "activation_watch_sha256": hashlib.sha256(watch.read_bytes()).hexdigest(),
        "weight_binary_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        "frozen_right_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
    },
    "reason": "a required frozen binary parent did not produce an executable passing dependency",
    "gate": {"passed": False},
    "claim_boundary": "terminal dependency skip; not activation-dual evidence",
}
temporary = result.with_suffix(result.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, result)
PY
  printf 'valid dependency rejection\n' > "$state/ACTIVATION_DUAL_SKIPPED"
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
    echo "swap is active during activation-dual admission" >&2
    exit 74
  fi
  if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
    echo "less than 10 GiB disk is free for activation-dual fitting" >&2
    exit 75
  fi
  free -b > "$state/memory-before.txt"
  df -B1 /workspace > "$state/disk-before.txt"

  set +e
  PYTHONPATH="$repo" taskset -c 0-7 "$python_bin" \
    "$runtime/run_qwen38_27b_activation_dual_binary_mlp_v1.py" \
    --weight-parent-receipt "$weight_parent_receipt" \
    --frozen-right-receipt "$frozen_right_receipt" \
    --activation-cache-receipt "$cache_receipt" \
    --out "$result" \
    --artifact "$artifact" \
    --threads 8 \
    > "$state/run.stdout" 2> "$state/run.stderr"
  run_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "activation-dual runner exited without a terminal receipt" >&2
    exit 70
  fi
  if (( run_status != 0 && run_status != 2 )); then
    echo "activation-dual runner had unexpected exit $run_status" >&2
    exit "$run_status"
  fi
  "$python_bin" - "$result" "$run_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("activation-dual receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("activation-dual process status and gate disagree")
PY
  if (( run_status == 0 )); then
    printf 'pass\n' > "$state/ACTIVATION_DUAL_PASS"
  else
    printf 'valid rejection\n' > "$state/ACTIVATION_DUAL_REJECTED"
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
    "schema": "qwen38_27b_activation_dual_binary_mlp_watch_receipt_v1",
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
