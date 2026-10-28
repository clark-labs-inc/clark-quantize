#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen aligned2 cache source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
ordering_unit=${ORDERING_UNIT:-quant-qwen38-27b-cq3-layer0-mlp-watch-v1.service}
mlp_watch=${MLP_WATCH_STATE:-/workspace/quant-state/qwen38-27b-cq3-layer0-mlp-watch-v1}
mlp_result=${MLP_RESULT:-/workspace/quant-state/qwen38-27b-cq3-layer0-mlp-v1.json}
mmap_result=${MMAP_RESULT:-/workspace/quant-state/qwen38-27b-cq3-runtime-mmap-v1.json}
export_result=${EXPORT_RESULT:-/workspace/quant-state/qwen38-27b-cq3-row-normalized-export-v1.json}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-aligned2-cache-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-aligned2-cache-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
library=$repo/clark_quantize/_ternary_cpu.so

while systemctl is-active --quiet "$ordering_unit"; do
  sleep 30
done
if [[ ! -s "$mlp_watch/WATCH_TERMINAL" || ! -s "$mlp_watch/receipt.json" ]]; then
  echo "Qwen3.8-27B direct MLP watch lacks a terminal receipt" >&2
  exit 3
fi

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B aligned2 cache repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B aligned2 cache commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_aligned2_cache_preregistration_v1.json" \
  "$runtime/benchmark_qwen38_27b_aligned2_cache_v1.py" \
  clark_quantize/qwen38_cq3_runtime.py \
  clark_quantize/test_qwen38_cq3_runtime.py \
  clark_quantize/ternary_cpu.c \
  clark_quantize/ternary_cpu.py \
  clark_quantize/test_ternary_cpu.py \
  > "$state/source.sha256"

if "$python_bin" - "$mmap_result" "$export_result" "$mlp_result" <<'PY'
import json, sys
mmap_path, export_path, mlp_path = sys.argv[1:]
try:
    mmap = json.load(open(mmap_path))
    export = json.load(open(export_path))
    mlp = json.load(open(mlp_path))
except FileNotFoundError:
    raise SystemExit(1)
valid = (
    mmap.get("schema") == "qwen38_27b_cq3_runtime_mmap_receipt_v1"
    and mmap.get("gate", {}).get("passed") is True
    and export.get("schema")
    == "qwen38_27b_cq3_row_normalized_export_receipt_v1"
    and export.get("gate", {}).get("passed") is True
    and mlp.get("schema") == "qwen38_27b_cq3_layer0_mlp_receipt_v1"
    and isinstance(mlp.get("gate", {}).get("passed"), bool)
)
raise SystemExit(0 if valid else 1)
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
    clark_quantize/test_qwen38_cq3_runtime.py \
    clark_quantize/test_ternary_cpu.py \
    > "$state/tests.stdout" 2> "$state/tests.stderr"
  printf 'pass\n' > "$state/NATIVE_ALIGNED2_TESTS_PASS"
  free -b > "$state/memory-before.txt"
  set +e
  CLARK_TERNARY_CPU_LIB="$library" PYTHONPATH="$repo" \
    "$python_bin" "$runtime/benchmark_qwen38_27b_aligned2_cache_v1.py" \
    --receipt "$result" \
    > "$state/benchmark.stdout" 2> "$state/benchmark.stderr"
  benchmark_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "Qwen3.8-27B aligned2 cache benchmark lacks a terminal receipt" >&2
    exit 70
  fi
  if (( benchmark_status != 0 && benchmark_status != 2 )); then
    echo "Qwen3.8-27B aligned2 cache benchmark had unexpected exit $benchmark_status" >&2
    exit "$benchmark_status"
  fi
  "$python_bin" - "$result" "$benchmark_status" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if receipt.get("schema") != "qwen38_27b_aligned2_cache_receipt_v1":
    raise SystemExit("aligned2 cache receipt schema changed")
passed = receipt.get("gate", {}).get("passed") is True
if passed != (status == 0):
    raise SystemExit("aligned2 cache process status and frozen gate disagree")
PY
  if (( benchmark_status == 0 )); then
    printf 'pass\n' > "$state/ALIGNED2_CACHE_GATE_PASS"
  else
    printf 'valid rejection\n' > "$state/ALIGNED2_CACHE_GATE_REJECTED"
  fi
  printf 'terminal\n' > "$state/BENCHMARK_TERMINAL"
  free -b > "$state/memory-after.txt"
else
  "$python_bin" - "$mmap_result" "$export_result" "$mlp_watch/receipt.json" "$state" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
mmap, export, mlp_watch, state = map(Path, sys.argv[1:])
parents = {}
for label, path in (("mmap", mmap), ("export", export), ("mlp_watch", mlp_watch)):
    if path.is_file():
        content = path.read_bytes()
        parents[label] = {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
receipt = {
    "schema": "qwen38_27b_aligned2_cache_parent_rejection_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "terminal": True,
    "valid_nonpromotion": True,
    "benchmark_launched": False,
    "parents": parents,
    "reason": "mmap or export gate rejected, or the direct MLP gate lacked a valid terminal result",
}
(state / "benchmark-skip.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY
  printf 'parent rejected\n' > "$state/ALIGNED2_CACHE_SKIPPED"
fi

"$python_bin" - "$state" "$result" "$mmap_result" "$export_result" "$mlp_watch/receipt.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, result, mmap, export, mlp_watch = map(Path, sys.argv[1:])
launched = (state / "BENCHMARK_TERMINAL").is_file()
result_content = result.read_bytes() if launched and result.is_file() else None
result_receipt = json.loads(result_content) if result_content is not None else None
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
parents = {}
for label, path in (("mmap", mmap), ("export", export), ("mlp_watch", mlp_watch)):
    if path.is_file():
        content = path.read_bytes()
        parents[label] = {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
passed = result_receipt is not None and result_receipt.get("gate", {}).get("passed") is True
receipt = {
    "schema": "qwen38_27b_aligned2_cache_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "benchmark_launched": launched,
    "aligned2_cache_gate_pass": passed,
    "valid_nonpromotion": not passed,
    "result_receipt": str(result) if result_content is not None else None,
    "result_receipt_sha256": hashlib.sha256(result_content).hexdigest() if result_content is not None else None,
    "parents": parents,
    "artifacts": files,
    "claim_boundary": "one real Qwen3.8-27B derived tensor cache only; no whole-model cache, model quality, end-to-end tokens per second, or breakthrough claim",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
