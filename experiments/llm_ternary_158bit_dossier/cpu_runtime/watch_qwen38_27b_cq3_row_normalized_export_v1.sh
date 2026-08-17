#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen row-normalized export source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
ordering_unit=${ORDERING_UNIT:-quant-qwen38-27b-block203-row-scale4-chunk16-watch-v1.service}
parent_state=${PARENT_STATE:-/workspace/quant-state/qwen38-27b-full-row-normalized-scale-codebook4-watch-v1}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-cq3-row-normalized-export-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-cq3-row-normalized-export-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
out_dir=/workspace/quant-runs/qwen38-cq3-row-normalized-container-v1
output=$out_dir/Qwen3.8-27B-row-normalized.cq3
library=$repo/clark_quantize/_ternary_cpu.so

while systemctl is-active --quiet "$ordering_unit"; do
  sleep 30
done
if [[ ! -s "$parent_state/WATCH_TERMINAL" || ! -s "$parent_state/receipt.json" ]]; then
  echo "complete Qwen3.8-27B row-scale watch lacks a terminal receipt" >&2
  exit 3
fi

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B row-normalized export repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B row-normalized export commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_cq3_row_normalized_export_preregistration_v1.json" \
  "$runtime/run_qwen38_27b_cq3_export_v1.py" \
  "$runtime/test_qwen38_27b_cq3_row_normalized_export.py" \
  clark_quantize/qwen38_cq3_container.py \
  clark_quantize/qwen38_cq3_export.py \
  clark_quantize/ternary_cpu.c \
  clark_quantize/ternary_cpu.py \
  > "$state/source.sha256"

if "$python_bin" - "$parent_state/receipt.json" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
raise SystemExit(0 if receipt.get("terminal") is True and receipt.get("full_gate_pass") is True else 1)
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
  scripts/cpu/build_ternary_cpu.sh \
    > "$state/build-native.stdout" 2> "$state/build-native.stderr"
  sha256sum "$library" > "$state/native-library.sha256"
  CLARK_TERNARY_CPU_LIB="$library" "$python_bin" -m pytest -q \
    clark_quantize/test_qwen38_cq3_container.py \
    clark_quantize/test_qwen38_cq3_export.py \
    "$runtime/test_qwen38_27b_cq3_row_normalized_export.py" \
    > "$state/tests.stdout" 2> "$state/tests.stderr"
  printf 'pass\n' > "$state/EXPORT_TESTS_PASS"
  free -b > "$state/memory-before.txt"
  df -B1 /workspace > "$state/disk-before.txt"
  set +e
  CLARK_TERNARY_CPU_LIB="$library" PYTHONPATH="$repo" \
    "$python_bin" "$runtime/run_qwen38_27b_cq3_export_v1.py" \
    --row-normalized \
    --mapped-codebooks /workspace/quant-runs/qwen38-row-normalized-scale-codebook4-v1 \
    --full-codebooks /workspace/quant-runs/qwen38-full-row-normalized-scale-codebook4-v1 \
    --out-dir "$out_dir" \
    --output "$output" \
    --receipt "$result" \
    > "$state/export.stdout" 2> "$state/export.stderr"
  export_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "Qwen3.8-27B row-normalized export lacks a terminal receipt" >&2
    exit 70
  fi
  if (( export_status != 0 && export_status != 2 )); then
    echo "Qwen3.8-27B row-normalized export had unexpected exit $export_status" >&2
    exit "$export_status"
  fi
  "$python_bin" - "$result" "$export_status" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if receipt.get("schema") != "qwen38_27b_cq3_row_normalized_export_receipt_v1":
    raise SystemExit("row-normalized export receipt schema changed")
passed = receipt.get("gate", {}).get("passed") is True
if passed != (status == 0):
    raise SystemExit("row-normalized export process status and frozen gate disagree")
PY
  if (( export_status == 0 )); then
    printf 'pass\n' > "$state/CONTAINER_STORAGE_GATE_PASS"
  else
    printf 'valid rejection\n' > "$state/CONTAINER_STORAGE_GATE_REJECTED"
  fi
  printf 'terminal\n' > "$state/EXPORT_TERMINAL"
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
    "schema": "qwen38_27b_cq3_row_normalized_export_parent_rejection_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "valid_nonpromotion": True,
    "container_export_launched": False,
    "reason": "complete 666-tensor row-normalized reconstruction or storage gate rejected",
}
(state / "export-skip.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
  printf 'parent rejected\n' > "$state/CONTAINER_EXPORT_SKIPPED"
fi

"$python_bin" - "$state" "$result" "$parent_state/receipt.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, result, parent = map(Path, sys.argv[1:])
parent_content = parent.read_bytes()
launched = (state / "EXPORT_TERMINAL").is_file()
# A parent rejection must never inherit a stale result receipt left at the
# canonical result path by an earlier manual run.
result_content = result.read_bytes() if launched and result.is_file() else None
result_receipt = json.loads(result_content) if result_content is not None else None
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
passed = result_receipt is not None and result_receipt.get("gate", {}).get("passed") is True
receipt = {
    "schema": "qwen38_27b_cq3_row_normalized_export_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "export_launched": launched,
    "container_storage_gate_pass": passed,
    "valid_nonpromotion": not passed,
    "result_receipt": str(result) if result_content is not None else None,
    "result_receipt_sha256": hashlib.sha256(result_content).hexdigest() if result_content is not None else None,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(parent_content).hexdigest(),
    "artifacts": files,
    "claim_boundary": "complete authenticated Qwen3.8-27B row-normalized storage container only; no runnable model, quality, throughput, or breakthrough claim",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
