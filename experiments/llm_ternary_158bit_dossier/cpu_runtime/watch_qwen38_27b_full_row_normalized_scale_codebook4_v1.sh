#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen full row-scale source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-row-normalized-scale-codebook4-watch-v1.service}
parent_state=${PARENT_STATE:-/workspace/quant-state/qwen38-27b-row-normalized-scale-codebook4-watch-v1}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-full-row-normalized-scale-codebook4-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-full-row-normalized-scale-codebook4-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

while systemctl is-active --quiet "$parent_unit"; do
  sleep 30
done
if [[ ! -s "$parent_state/WATCH_TERMINAL" || ! -s "$parent_state/receipt.json" ]]; then
  echo "parent Qwen3.8-27B row-scale watch lacks a terminal receipt" >&2
  exit 3
fi

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B full row-scale repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B full row-scale commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_full_row_normalized_scale_codebook4_preregistration_v1.json" \
  "$runtime/run_qwen38_27b_full_row_normalized_scale_codebook4_v1.py" \
  clark_quantize/qwen38_row_scale_codebook.py \
  clark_quantize/qwen38_row_scale_storage.py \
  > "$state/source.sha256"

if "$python_bin" - "$parent_state/receipt.json" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
raise SystemExit(0 if receipt.get("terminal") is True and receipt.get("static_reconstruction_pass") is True else 1)
PY
then
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
  free -b > "$state/memory-before.txt"
  df -B1 /workspace > "$state/disk-before.txt"
  set +e
  PYTHONPATH="$repo" "$python_bin" \
    "$runtime/run_qwen38_27b_full_row_normalized_scale_codebook4_v1.py" \
    > "$state/experiment.stdout" 2> "$state/experiment.stderr"
  experiment_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "full Qwen3.8-27B row-scale fit exited without a terminal receipt" >&2
    exit 70
  fi
  if (( experiment_status != 0 && experiment_status != 2 )); then
    echo "full Qwen3.8-27B row-scale fit had unexpected exit $experiment_status" >&2
    exit "$experiment_status"
  fi
  "$python_bin" - "$result" "$experiment_status" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if receipt.get("complete") is not True:
    raise SystemExit("full row-scale receipt is not complete")
passed = receipt.get("gate", {}).get("passed") is True
if passed != (status == 0):
    raise SystemExit("full row-scale process status and frozen gate disagree")
PY
  if (( experiment_status == 0 )); then
    printf 'pass\n' > "$state/FULL_RECONSTRUCTION_AND_STORAGE_PASS"
  else
    printf 'valid rejection\n' > "$state/FULL_RECONSTRUCTION_AND_STORAGE_REJECTED"
  fi
  printf 'terminal\n' > "$state/EXPERIMENT_TERMINAL"
  free -b > "$state/memory-after.txt"
  df -B1 /workspace > "$state/disk-after.txt"
else
  "$python_bin" - "$parent_state/receipt.json" "$state" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
parent = Path(sys.argv[1])
state = Path(sys.argv[2])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_full_row_scale_parent_rejection_skip_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "valid_nonpromotion": True,
    "full_fit_launched": False,
    "reason": "mapped 402-tensor row-normalized reconstruction gate rejected",
}
(state / "full-fit-skip.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
  printf 'parent rejected\n' > "$state/FULL_FIT_SKIPPED"
fi

"$python_bin" - "$state" "$result" "$parent_state/receipt.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, result, parent = map(Path, sys.argv[1:])
parent_content = parent.read_bytes()
result_content = result.read_bytes() if result.is_file() else None
result_receipt = json.loads(result_content) if result_content is not None else None
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
receipt = {
    "schema": "qwen38_27b_full_row_normalized_scale_codebook4_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "full_fit_launched": (state / "EXPERIMENT_TERMINAL").is_file(),
    "full_gate_pass": result_receipt is not None and result_receipt.get("gate", {}).get("passed") is True,
    "valid_nonpromotion": not (result_receipt is not None and result_receipt.get("gate", {}).get("passed") is True),
    "result_receipt": str(result) if result_content is not None else None,
    "result_receipt_sha256": hashlib.sha256(result_content).hexdigest() if result_content is not None else None,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(parent_content).hexdigest(),
    "artifacts": files,
    "claim_boundary": "complete static reconstruction and storage orchestration only; no model-quality or breakthrough claim",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
