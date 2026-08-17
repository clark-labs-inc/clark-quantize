#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen recurrent DeltaNet source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
ordering_unit=${ORDERING_UNIT:-quant-qwen38-27b-aligned2-full-cache-watch-v1.service}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-recurrent-delta-avx2-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-recurrent-delta-avx2-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
library=$repo/clark_quantize/_ternary_cpu.so

while systemctl is-active --quiet "$ordering_unit"; do
  sleep 30
done
if [[ -s "$state/WATCH_TERMINAL" && -s "$state/receipt.json" ]]; then
  exit 0
fi

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
  echo "another Qwen3.8 QAT or activation-cache process is active" >&2
  printf '%s\n' "$training_conflicts" >&2
  exit 2
fi

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B recurrent-kernel repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B recurrent-kernel commit" >&2
  exit 65
fi

git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_recurrent_delta_avx2_preregistration_v1.json" \
  "$runtime/benchmark_qwen38_27b_recurrent_delta_avx2_v1.py" \
  clark_quantize/ternary_cpu.c \
  clark_quantize/ternary_cpu.py \
  clark_quantize/test_ternary_cpu.py \
  clark_quantize/test_qwen_evaluation_preregistration.py \
  > "$state/source.sha256"
lscpu > "$state/lscpu.txt"
cc --version > "$state/compiler.txt"
free -b > "$state/memory-before.txt"
scripts/cpu/build_ternary_cpu.sh \
  > "$state/build-native.stdout" 2> "$state/build-native.stderr"
sha256sum "$library" > "$state/native-library.sha256"
CLARK_TERNARY_CPU_LIB="$library" "$python_bin" -m pytest -q \
  clark_quantize/test_ternary_cpu.py \
  clark_quantize/test_qwen_evaluation_preregistration.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"
printf 'pass\n' > "$state/TESTS_PASS"

set +e
CLARK_TERNARY_CPU_LIB="$library" PYTHONPATH="$repo" \
  "$python_bin" "$runtime/benchmark_qwen38_27b_recurrent_delta_avx2_v1.py" \
  --receipt "$result" \
  > "$state/benchmark.stdout" 2> "$state/benchmark.stderr"
benchmark_status=$?
set -e
if [[ ! -s "$result" ]]; then
  echo "Qwen3.8-27B recurrent DeltaNet benchmark lacks a terminal receipt" >&2
  exit 70
fi
if (( benchmark_status != 0 && benchmark_status != 2 )); then
  echo "Qwen3.8-27B recurrent DeltaNet benchmark had unexpected exit $benchmark_status" >&2
  exit "$benchmark_status"
fi
"$python_bin" - "$result" "$benchmark_status" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if receipt.get("schema") != "qwen38_27b_recurrent_delta_avx2_receipt_v1":
    raise SystemExit("recurrent DeltaNet receipt schema changed")
passed = receipt.get("gate", {}).get("passed") is True
if passed != (status == 0):
    raise SystemExit("recurrent DeltaNet process status and frozen gate disagree")
PY
if (( benchmark_status == 0 )); then
  printf 'pass\n' > "$state/KERNEL_GATE_PASS"
else
  printf 'valid rejection\n' > "$state/KERNEL_GATE_REJECTED"
fi
free -b > "$state/memory-after.txt"
printf 'terminal\n' > "$state/BENCHMARK_TERMINAL"

"$python_bin" - "$state" "$result" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, result = map(Path, sys.argv[1:])
content = result.read_bytes()
candidate = json.loads(content)
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
receipt = {
    "schema": "qwen38_27b_recurrent_delta_avx2_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "kernel_gate_pass": candidate.get("gate", {}).get("passed") is True,
    "valid_nonpromotion": candidate.get("gate", {}).get("passed") is not True,
    "result_receipt": str(result),
    "result_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "artifacts": files,
    "claim_boundary": "isolated recurrent update only; no full-layer parity, end-to-end capability, end-to-end tokens per second, or breakthrough claim",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
