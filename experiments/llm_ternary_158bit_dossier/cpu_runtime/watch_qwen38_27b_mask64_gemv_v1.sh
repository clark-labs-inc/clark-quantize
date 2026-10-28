#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen mask64 GEMV bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-learned-hash-routing-v1.service}
parent_receipt=${PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-learned-hash-routing-watch-v1/receipt.json}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-mask64-gemv-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-mask64-gemv-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

# shellcheck source=experiments/llm_ternary_158bit_dossier/cpu_runtime/wait_for_qwen38_terminal_receipt.sh
source "$(dirname "$0")/wait_for_qwen38_terminal_receipt.sh"
qwen38_wait_for_terminal_receipt "$parent_unit" "$parent_receipt" 30

if [[ -s "$result" ]] && "$python_bin" - "$result" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("terminal") is True else 1)
PY
then
  echo "authenticated mask64 GEMV receipt already exists"
  exit 0
fi

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen mask64 GEMV bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B mask64 GEMV repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B mask64 GEMV commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_mask64_gemv_preregistration_v1.json" \
  "$runtime/benchmark_qwen38_27b_mask64_gemv_v1.py" \
  clark_quantize/qwen38_fixed_support_mask64_cpu.c \
  clark_quantize/qwen38_fixed_support_mask64_cpu.py \
  clark_quantize/test_qwen38_fixed_support_mask64_cpu.py \
  clark_quantize/test_qwen38_mask64_gemv_preregistration.py \
  clark_quantize/ternary_cpu.c \
  > "$state/source.sha256"

llama_conflicts=$(ps -eo pid=,comm=,args= | awk \
  '$2 ~ /^llama-(server|cli|bench|perplexi)/ {print}')
if [[ -n "$llama_conflicts" ]]; then
  echo "another llama.cpp workload is active" >&2
  printf '%s\n' "$llama_conflicts" >&2
  exit 2
fi
if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((2 * 1024 * 1024)) ]]; then
  echo "less than 2 GiB is available for mask64 GEMV benchmark" >&2
  exit 72
fi
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "swap is active during mask64 GEMV admission" >&2
  exit 74
fi
if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
  echo "less than 10 GiB disk is free for mask64 GEMV benchmark" >&2
  exit 75
fi

mask_library="$state/qwen38-mask64-cpu.so"
ternary_library="$state/ternary-cpu-control.so"
cc -std=c11 -O3 -Wall -Wextra -Werror -shared -fPIC -mtune=native \
  clark_quantize/qwen38_fixed_support_mask64_cpu.c -o "$mask_library"
cc -std=c11 -O3 -Wall -Wextra -Werror -shared -fPIC -mtune=native \
  clark_quantize/ternary_cpu.c -lm -o "$ternary_library"
sha256sum "$mask_library" "$ternary_library" > "$state/library.sha256"
CLARK_MASK64_CPU_LIB="$mask_library" "$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_fixed_support_mask64_cpu.py \
  clark_quantize/test_qwen38_mask64_gemv_preregistration.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"
free -b > "$state/memory-before.txt"
set +e
taskset -c 0 env PYTHONPATH="$repo" CLARK_MASK64_CPU_LIB="$mask_library" \
  "$python_bin" "$runtime/benchmark_qwen38_27b_mask64_gemv_v1.py" \
  --mask-library "$mask_library" \
  --ternary-library "$ternary_library" \
  --out "$result" \
  --warmups 3 \
  --repetitions 11 \
  > "$state/benchmark.stdout" 2> "$state/benchmark.stderr"
benchmark_status=$?
set -e
if [[ ! -s "$result" ]]; then
  echo "mask64 GEMV benchmark exited without a terminal receipt" >&2
  exit 70
fi
if (( benchmark_status != 0 && benchmark_status != 2 )); then
  echo "mask64 GEMV benchmark had unexpected exit $benchmark_status" >&2
  exit "$benchmark_status"
fi
"$python_bin" - "$result" "$benchmark_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("mask64 GEMV receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("mask64 GEMV process status and gate disagree")
PY
if (( benchmark_status == 0 )); then
  printf 'pass\n' > "$state/MASK64_GEMV_PASS"
else
  printf 'valid rejection\n' > "$state/MASK64_GEMV_REJECTED"
fi
free -b > "$state/memory-after.txt"

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
    "schema": "qwen38_27b_mask64_gemv_watch_receipt_v1",
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
