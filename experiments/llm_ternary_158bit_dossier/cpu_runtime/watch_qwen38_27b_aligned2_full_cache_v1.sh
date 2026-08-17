#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen full-cache source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
ordering_unit=${ORDERING_UNIT:-quant-qwen38-27b-aligned2-cache-watch-v1.service}
parent_watch=${PARENT_WATCH_STATE:-/workspace/quant-state/qwen38-27b-aligned2-cache-watch-v1}
parent_result=${PARENT_RESULT:-/workspace/quant-state/qwen38-27b-aligned2-cache-v1.json}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-aligned2-full-cache-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-aligned2-full-cache-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
library=$repo/clark_quantize/_ternary_cpu.so

while systemctl is-active --quiet "$ordering_unit"; do
  sleep 30
done
if [[ ! -s "$parent_watch/WATCH_TERMINAL" || ! -s "$parent_watch/receipt.json" ]]; then
  echo "Qwen3.8-27B aligned2 parent watch lacks a terminal receipt" >&2
  exit 3
fi

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B full-cache repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B full-cache commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_aligned2_full_cache_preregistration_v1.json" \
  "$runtime/build_qwen38_27b_aligned2_cache_v1.py" \
  clark_quantize/qwen38_aligned2_cache.py \
  clark_quantize/test_qwen38_aligned2_cache.py \
  clark_quantize/qwen38_cq3_runtime.py \
  clark_quantize/ternary_cpu.c \
  clark_quantize/ternary_cpu.py \
  > "$state/source.sha256"

if "$python_bin" - "$parent_result" <<'PY'
import json, sys
try:
    parent = json.load(open(sys.argv[1]))
except FileNotFoundError:
    raise SystemExit(1)
passed = (
    parent.get("schema") == "qwen38_27b_aligned2_cache_receipt_v1"
    and parent.get("gate", {}).get("passed") is True
)
raise SystemExit(0 if passed else 1)
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
  lscpu > "$state/lscpu.txt"
  CLARK_TERNARY_CPU_LIB="$library" "$python_bin" -m pytest -q \
    clark_quantize/test_qwen38_aligned2_cache.py \
    clark_quantize/test_qwen38_cq3_runtime.py \
    clark_quantize/test_ternary_cpu.py \
    > "$state/tests.stdout" 2> "$state/tests.stderr"
  printf 'pass\n' > "$state/FULL_CACHE_TESTS_PASS"
  free -b > "$state/memory-before.txt"
  df -B1 /workspace > "$state/disk-before.txt"
  set +e
  CLARK_TERNARY_CPU_LIB="$library" PYTHONPATH="$repo" \
    "$python_bin" "$runtime/build_qwen38_27b_aligned2_cache_v1.py" \
    --parent-receipt "$parent_result" \
    --receipt "$result" \
    > "$state/build-cache.stdout" 2> "$state/build-cache.stderr"
  build_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "Qwen3.8-27B full-cache build lacks a terminal receipt" >&2
    exit 70
  fi
  if (( build_status != 0 && build_status != 2 )); then
    echo "Qwen3.8-27B full-cache build had unexpected exit $build_status" >&2
    exit "$build_status"
  fi
  "$python_bin" - "$result" "$build_status" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if receipt.get("schema") != "qwen38_27b_aligned2_full_cache_receipt_v1":
    raise SystemExit("full-cache receipt schema changed")
passed = receipt.get("gate", {}).get("passed") is True
if passed != (status == 0):
    raise SystemExit("full-cache process status and frozen gate disagree")
PY
  if (( build_status == 0 )); then
    printf 'pass\n' > "$state/FULL_CACHE_GATE_PASS"
  else
    printf 'valid rejection\n' > "$state/FULL_CACHE_GATE_REJECTED"
  fi
  printf 'terminal\n' > "$state/BUILD_TERMINAL"
  free -b > "$state/memory-after.txt"
  df -B1 /workspace > "$state/disk-after.txt"
else
  "$python_bin" - "$parent_watch/receipt.json" "$state" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
parent, state = map(Path, sys.argv[1:])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_aligned2_full_cache_parent_rejection_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "terminal": True,
    "valid_nonpromotion": True,
    "cache_allocated": False,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "reason": "the frozen actual-tensor aligned2 speed or integrity gate rejected",
}
(state / "build-skip.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY
  printf 'parent rejected\n' > "$state/FULL_CACHE_SKIPPED"
fi

"$python_bin" - "$state" "$result" "$parent_watch/receipt.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, result, parent = map(Path, sys.argv[1:])
launched = (state / "BUILD_TERMINAL").is_file()
result_content = result.read_bytes() if launched and result.is_file() else None
result_receipt = json.loads(result_content) if result_content is not None else None
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
parent_content = parent.read_bytes()
passed = result_receipt is not None and result_receipt.get("gate", {}).get("passed") is True
receipt = {
    "schema": "qwen38_27b_aligned2_full_cache_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "build_launched": launched,
    "full_cache_gate_pass": passed,
    "valid_nonpromotion": not passed,
    "result_receipt": str(result) if result_content is not None else None,
    "result_receipt_sha256": hashlib.sha256(result_content).hexdigest() if result_content is not None else None,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(parent_content).hexdigest(),
    "artifacts": files,
    "claim_boundary": "complete derived Qwen3.8-27B weights only; no runnable whole-model graph, capability, end-to-end tokens per second, or breakthrough claim",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
