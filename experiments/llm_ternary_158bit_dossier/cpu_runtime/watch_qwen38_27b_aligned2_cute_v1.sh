#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen CUTE source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
ordering_unit=${ORDERING_UNIT:-quant-qwen38-27b-aligned2-full-text-watch-v1.service}
parent_watch=${PARENT_WATCH_STATE:-/workspace/quant-state/qwen38-27b-aligned2-full-text-watch-v1}
parent_result=${PARENT_RESULT:-/workspace/quant-state/qwen38-27b-aligned2-full-text-v1.json}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-aligned2-cute-watch-v1}
progress=${PROGRESS_RECEIPT:-/workspace/quant-state/qwen38-27b-aligned2-cute-v1.progress.json}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-aligned2-cute-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
library=$repo/clark_quantize/_ternary_cpu.so

while systemctl is-active --quiet "$ordering_unit"; do
  sleep 30
done
if [[ -s "$state/WATCH_TERMINAL" && -s "$state/receipt.json" ]]; then
  exit 0
fi
if [[ ! -s "$parent_watch/WATCH_TERMINAL" || ! -s "$parent_watch/receipt.json" ]]; then
  echo "Qwen3.8-27B full-text watch lacks a terminal receipt" >&2
  exit 3
fi

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B CUTE repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B CUTE commit" >&2
  exit 65
fi

git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_aligned2_cute_preregistration_v1.json" \
  "$runtime/eval_qwen38_27b_aligned2_cute_v1.py" \
  "$runtime/eval_llama_cpp_cute_quick.py" \
  "$runtime/cute_quick_v1.json" \
  clark_quantize/qwen38_aligned2_cache.py \
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
    parent.get("schema") == "qwen38_27b_aligned2_full_text_receipt_v1"
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
    echo "another Qwen3.8 QAT or activation-cache process is active" >&2
    printf '%s\n' "$training_conflicts" >&2
    exit 2
  fi
  if [[ "$($python_bin -c 'import transformers; print(transformers.__version__)')" != "5.15.0" ]]; then
    echo "infrastructure-invalid: Transformers version changed" >&2
    exit 4
  fi
  lscpu > "$state/lscpu.txt"
  cc --version > "$state/compiler.txt"
  free -b > "$state/memory-before.txt"
  scripts/cpu/build_ternary_cpu.sh \
    > "$state/build-native.stdout" 2> "$state/build-native.stderr"
  sha256sum "$library" > "$state/native-library.sha256"
  CLARK_TERNARY_CPU_LIB="$library" "$python_bin" -m pytest -q \
    clark_quantize/test_qwen38_cpu_runtime_scripts.py \
    clark_quantize/test_qwen38_aligned2_cache.py \
    clark_quantize/test_qwen38_cq3_runtime.py \
    clark_quantize/test_ternary_cpu.py \
    > "$state/tests.stdout" 2> "$state/tests.stderr"
  printf 'pass\n' > "$state/TESTS_PASS"
  set +e
  CLARK_TERNARY_CPU_LIB="$library" PYTHONPATH="$repo" \
    "$python_bin" "$runtime/eval_qwen38_27b_aligned2_cute_v1.py" \
    --parent-receipt "$parent_result" \
    --progress "$progress" \
    --receipt "$result" \
    > "$state/evaluation.stdout" 2> "$state/evaluation.stderr"
  evaluation_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "Qwen3.8-27B CUTE evaluation lacks a terminal receipt" >&2
    exit 70
  fi
  if (( evaluation_status != 0 && evaluation_status != 2 )); then
    echo "Qwen3.8-27B CUTE evaluation had unexpected exit $evaluation_status" >&2
    exit "$evaluation_status"
  fi
  "$python_bin" - "$result" "$evaluation_status" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if receipt.get("schema") != "qwen38_27b_aligned2_cute_receipt_v1":
    raise SystemExit("CUTE receipt schema changed")
summary = next(
    (value for value in receipt.get("summaries", []) if value.get("task") == "cute_quick_20"),
    None,
)
if not isinstance(summary, dict) or summary.get("n") != 20 or len(receipt.get("records", [])) != 20:
    raise SystemExit("CUTE terminal denominator changed")
passed = receipt.get("gate", {}).get("passed") is True
if passed != (status == 0):
    raise SystemExit("CUTE process status and frozen gate disagree")
PY
  if (( evaluation_status == 0 )); then
    printf 'pass\n' > "$state/CUTE_GATE_PASS"
  else
    printf 'valid rejection\n' > "$state/CUTE_GATE_REJECTED"
  fi
  printf 'terminal\n' > "$state/EVALUATION_TERMINAL"
  free -b > "$state/memory-after.txt"
else
  "$python_bin" - "$parent_watch/receipt.json" "$state" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
parent, state = map(Path, sys.argv[1:])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_aligned2_cute_parent_rejection_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "terminal": True,
    "valid_nonpromotion": True,
    "evaluation_launched": False,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "reason": "the complete full-text graph parent gate rejected",
}
(state / "evaluation-skip.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY
  printf 'parent rejected\n' > "$state/CUTE_SKIPPED"
fi

"$python_bin" - "$state" "$result" "$parent_watch/receipt.json" "$progress" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, result, parent, progress = map(Path, sys.argv[1:])
launched = (state / "EVALUATION_TERMINAL").is_file()
result_content = result.read_bytes() if launched and result.is_file() else None
candidate = json.loads(result_content) if result_content is not None else None
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
if progress.is_file():
    files["external-progress"] = {
        "path": str(progress),
        "bytes": progress.stat().st_size,
        "sha256": hashlib.sha256(progress.read_bytes()).hexdigest(),
    }
parent_content = parent.read_bytes()
passed = candidate is not None and candidate.get("gate", {}).get("passed") is True
receipt = {
    "schema": "qwen38_27b_aligned2_cute_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "evaluation_launched": launched,
    "cute_gate_pass": passed,
    "valid_nonpromotion": not passed,
    "result_receipt": str(result) if result_content is not None else None,
    "result_receipt_sha256": hashlib.sha256(result_content).hexdigest() if result_content is not None else None,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(parent_content).hexdigest(),
    "artifacts": files,
    "claim_boundary": "frozen non-leaderboard CUTE-Quick-20 only; MATH or MMLU replication, broad capability, paired throughput, and breakthrough remain open",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
