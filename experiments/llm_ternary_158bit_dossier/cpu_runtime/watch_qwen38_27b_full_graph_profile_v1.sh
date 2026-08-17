#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen Qwen3.8-27B profile source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
ordering_unit=${ORDERING_UNIT:-quant-qwen38-27b-persistent-affinity-watch-v1.service}
parent_result=${PARENT_RESULT:-/workspace/quant-state/qwen38-27b-aligned2-full-text-v1.json}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-full-graph-profile-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-full-graph-profile-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
library=$repo/clark_quantize/_ternary_cpu.so

while systemctl is-active --quiet "$ordering_unit"; do
  sleep 30
done
if [[ -s "$state/WATCH_TERMINAL" && -s "$state/receipt.json" ]]; then
  exit 0
fi

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B profile repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B profile commit" >&2
  exit 65
fi

git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_full_graph_profile_preregistration_v1.json" \
  "$runtime/profile_qwen38_27b_full_graph_v1.py" \
  "$runtime/benchmark_qwen38_27b_aligned2_full_text_v1.py" \
  "$runtime/benchmark_qwen38_27b_persistent_affinity_v1.py" \
  clark_quantize/qwen38_aligned2_cache.py \
  clark_quantize/qwen38_cq3_runtime.py \
  clark_quantize/ternary_cpu.c \
  clark_quantize/ternary_cpu.py \
  > "$state/source.sha256"

if "$python_bin" - "$parent_result" "$library" <<'PY'
import hashlib, json, sys
from pathlib import Path
parent = json.load(open(sys.argv[1]))
library = Path(sys.argv[2])
digest = hashlib.sha256(library.read_bytes()).hexdigest() if library.is_file() else None
passed = (
    parent.get("schema") == "qwen38_27b_aligned2_full_text_receipt_v1"
    and parent.get("gate", {}).get("passed") is True
    and parent.get("native_library", {}).get("sha256") == digest
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
  lscpu > "$state/lscpu.txt"
  free -b > "$state/memory-before.txt"
  sha256sum "$library" > "$state/native-library.sha256"
  CLARK_TERNARY_CPU_LIB="$library" "$python_bin" -m pytest -q \
    clark_quantize/test_qwen38_cpu_runtime_scripts.py \
    > "$state/tests.stdout" 2> "$state/tests.stderr"
  printf 'pass\n' > "$state/TESTS_PASS"
  set +e
  CLARK_TERNARY_CPU_LIB="$library" PYTHONPATH="$repo" \
    "$python_bin" "$runtime/profile_qwen38_27b_full_graph_v1.py" \
    --parent-receipt "$parent_result" \
    --receipt "$result" \
    > "$state/profile.stdout" 2> "$state/profile.stderr"
  profile_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "Qwen3.8-27B profile lacks a terminal receipt" >&2
    exit 70
  fi
  if (( profile_status != 0 && profile_status != 2 )); then
    echo "Qwen3.8-27B profile had unexpected exit $profile_status" >&2
    exit "$profile_status"
  fi
  "$python_bin" - "$result" "$profile_status" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if receipt.get("schema") != "qwen38_27b_full_graph_profile_receipt_v1":
    raise SystemExit("full-graph profile receipt schema changed")
passed = receipt.get("gate", {}).get("passed") is True
if passed != (status == 0):
    raise SystemExit("full-graph profile process status and gate disagree")
PY
  if (( profile_status == 0 )); then
    printf 'pass\n' > "$state/PROFILE_GATE_PASS"
  else
    printf 'valid rejection\n' > "$state/PROFILE_GATE_REJECTED"
  fi
  printf 'terminal\n' > "$state/PROFILE_TERMINAL"
  free -b > "$state/memory-after.txt"
else
  "$python_bin" - "$parent_result" "$state" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
parent, state = map(Path, sys.argv[1:])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_full_graph_profile_parent_rejection_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "terminal": True,
    "valid_nonpromotion": True,
    "profile_launched": False,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "reason": "the full-text parent or its native library identity rejected",
}
(state / "profile-skip.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
  printf 'parent rejected\n' > "$state/PROFILE_SKIPPED"
fi

"$python_bin" - "$state" "$result" "$parent_result" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, result, parent = map(Path, sys.argv[1:])
launched = (state / "PROFILE_TERMINAL").is_file()
result_content = result.read_bytes() if launched and result.is_file() else None
candidate = json.loads(result_content) if result_content is not None else None
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
parent_content = parent.read_bytes()
passed = candidate is not None and candidate.get("gate", {}).get("passed") is True
receipt = {
    "schema": "qwen38_27b_full_graph_profile_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "profile_launched": launched,
    "profile_gate_pass": passed,
    "valid_nonpromotion": not passed,
    "result_receipt": str(result) if result_content is not None else None,
    "result_receipt_sha256": hashlib.sha256(result_content).hexdigest() if result_content is not None else None,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(parent_content).hexdigest(),
    "artifacts": files,
    "claim_boundary": "one-token Qwen3.8-27B projection attribution only; capability throughput improvement and breakthrough remain open",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
