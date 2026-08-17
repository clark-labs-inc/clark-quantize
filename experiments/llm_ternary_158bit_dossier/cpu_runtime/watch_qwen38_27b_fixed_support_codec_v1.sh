#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen fixed-support codec bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-fixed-support-codec-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-fixed-support-codec-v1.json}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-hash-residual-screen-v1.service}
parent_receipt=${PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-hash-residual-screen-v1.json}
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
  echo "authenticated fixed-support receipt already exists"
  exit 0
fi

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen Qwen3.8-27B fixed-support bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B fixed-support repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B fixed-support commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_fixed_support_codec_preregistration_v1.json" \
  "$runtime/export_qwen38_27b_fixed_support_codec_v1.py" \
  clark_quantize/qwen38_fixed_support_codec.py \
  > "$state/source.sha256"

"$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_fixed_support_codec.py \
  clark_quantize/test_qwen38_fixed_support_preregistration.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"

if ! "$python_bin" - "$parent_receipt" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("gate", {}).get("passed") is True else 1)
PY
then
  "$python_bin" - "$parent_receipt" "$result" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
parent, result = map(Path, sys.argv[1:])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_fixed_support_codec_skip_receipt_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_hash_receipt": str(parent),
    "parent_hash_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "complete": True,
    "terminal": True,
    "valid_nonpromotion": True,
    "reason": "the frozen data-free Clark-Hash residual gate rejected",
    "codes_recomputed": False,
    "artifacts_materialized": False,
    "gate": {"passed": False},
    "claim_boundary": "valid parent rejection only; no fixed-support codec result",
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
  training_conflicts=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^(python|python3|torchrun)$/ && $0 ~ /(train_mlp|qwen.*qat|activation.*cache)/ {print}')
  if [[ -n "$training_conflicts" ]]; then
    echo "another QAT or activation-cache process is active" >&2
    printf '%s\n' "$training_conflicts" >&2
    exit 2
  fi
  if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((6 * 1024 * 1024)) ]]; then
    echo "less than 6 GiB is available for fixed-support export" >&2
    exit 72
  fi
  if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
    echo "swap is active during fixed-support admission" >&2
    exit 74
  fi
  if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
    echo "less than 10 GiB disk is free for fixed-support export" >&2
    exit 75
  fi
  free -b > "$state/memory-before.txt"
  set +e
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 0-3 env PYTHONPATH="$repo" \
    "$python_bin" "$runtime/export_qwen38_27b_fixed_support_codec_v1.py" \
    --parent-receipt "$parent_receipt" \
    --out "$result" \
    --threads 4 \
    > "$state/export.stdout" 2> "$state/export.stderr"
  export_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "fixed-support exporter exited without a terminal receipt" >&2
    exit 70
  fi
  if (( export_status != 0 && export_status != 2 )); then
    echo "fixed-support exporter had unexpected exit $export_status" >&2
    exit "$export_status"
  fi
  "$python_bin" - "$result" "$export_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("fixed-support receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("fixed-support process status and gate disagree")
PY
  if (( export_status == 0 )); then
    printf 'pass\n' > "$state/FIXED_SUPPORT_PASS"
  else
    printf 'valid rejection\n' > "$state/FIXED_SUPPORT_REJECTED"
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
    "schema": "qwen38_27b_fixed_support_codec_watch_receipt_v1",
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
