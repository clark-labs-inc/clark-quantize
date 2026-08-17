#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen fused mask64/gain-hash bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
gain_unit=${GAIN_UNIT:-quant-qwen38-27b-gain-hash-routing-v1.service}
gain_receipt=${GAIN_RECEIPT:-/workspace/quant-state/qwen38-27b-gain-hash-routing-v1.json}
sign_unit=${SIGN_UNIT:-quant-qwen38-27b-mask64-hash-fused-v1.service}
sign_receipt=${SIGN_RECEIPT:-/workspace/quant-state/qwen38-27b-mask64-hash-fused-v1.json}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-mask64-gain-hash-fused-watch-v1}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-mask64-gain-hash-fused-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

# shellcheck source=wait_for_qwen38_terminal_receipt.sh
source "$(dirname "$0")/wait_for_qwen38_terminal_receipt.sh"
qwen38_wait_for_terminal_receipt "$gain_unit" "$gain_receipt" 30
qwen38_wait_for_terminal_receipt "$sign_unit" "$sign_receipt" 30

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen fused mask64/gain-hash bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty fused mask64/gain-hash repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed fused mask64/gain-hash commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_mask64_gain_hash_fused_preregistration_v1.json" \
  "$runtime/benchmark_qwen38_27b_mask64_gain_hash_fused_v1.py" \
  clark_quantize/qwen38_fixed_support_mask64_cpu.c \
  clark_quantize/qwen38_mask64_hash_fused_cpu.c \
  clark_quantize/qwen38_mask64_gain_hash_fused_cpu.c \
  clark_quantize/qwen38_mask64_gain_hash_fused_cpu.py \
  clark_quantize/test_qwen38_mask64_gain_hash_fused_cpu.py \
  > "$state/source.sha256"

if ! "$python_bin" - "$gain_receipt" "$sign_receipt" <<'PY'
import json, sys
parents = [json.load(open(path)) for path in sys.argv[1:]]
raise SystemExit(0 if all(p.get("gate", {}).get("passed") is True for p in parents) else 1)
PY
then
  "$python_bin" - "$state" "$gain_receipt" "$sign_receipt" "$result" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
state, gain, sign, result = map(Path, sys.argv[1:])
receipt = {
    "schema": "qwen38_27b_mask64_gain_hash_fused_skip_receipt_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_receipts": {
        "gain": {"path": str(gain), "sha256": hashlib.sha256(gain.read_bytes()).hexdigest()},
        "sign_kernel": {"path": str(sign), "sha256": hashlib.sha256(sign.read_bytes()).hexdigest()},
    },
    "complete": True,
    "terminal": True,
    "valid_nonpromotion": True,
    "reason": "the real-weight gain route or sign-only fused-kernel gate rejected",
    "benchmark_launched": False,
    "gate": {"passed": False},
    "claim_boundary": "valid parent rejection only; no gain-kernel result",
}
temporary = result.with_suffix(result.suffix + ".tmp")
result.parent.mkdir(parents=True, exist_ok=True)
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
os.replace(temporary, result)
(state / "PARENT_REJECTED").write_text("valid nonpromotion\n")
PY
else
  llama_conflicts=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^llama-(server|cli|bench|perplexi)/ {print}')
  if [[ -n "$llama_conflicts" ]]; then
    echo "another llama.cpp workload is active" >&2
    printf '%s\n' "$llama_conflicts" >&2
    exit 2
  fi
  if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((2 * 1024 * 1024)) ]]; then
    echo "less than 2 GiB is available for fused mask64/gain-hash benchmark" >&2
    exit 72
  fi
  if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
    echo "swap is active during fused mask64/gain-hash admission" >&2
    exit 74
  fi
  if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
    echo "less than 10 GiB disk is free for fused mask64/gain-hash benchmark" >&2
    exit 75
  fi

  library="$state/qwen38-mask64-gain-hash-fused.so"
  cc -std=c11 -O3 -Wall -Wextra -Werror -shared -fPIC -mtune=native \
    clark_quantize/qwen38_fixed_support_mask64_cpu.c \
    clark_quantize/qwen38_mask64_hash_fused_cpu.c \
    clark_quantize/qwen38_mask64_gain_hash_fused_cpu.c \
    -o "$library"
  sha256sum "$library" > "$state/library.sha256"
  CLARK_MASK64_GAIN_HASH_FUSED_CPU_LIB="$library" "$python_bin" -m pytest -q \
    clark_quantize/test_qwen38_mask64_gain_hash_fused_cpu.py \
    clark_quantize/test_qwen38_mask64_gain_hash_fused_preregistration.py \
    > "$state/tests.stdout" 2> "$state/tests.stderr"
  free -b > "$state/memory-before.txt"
  set +e
  taskset -c 0 env PYTHONPATH="$repo" \
    CLARK_MASK64_GAIN_HASH_FUSED_CPU_LIB="$library" \
    "$python_bin" "$runtime/benchmark_qwen38_27b_mask64_gain_hash_fused_v1.py" \
    --library "$library" \
    --out "$result" \
    --warmups 3 \
    --repetitions 11 \
    > "$state/benchmark.stdout" 2> "$state/benchmark.stderr"
  benchmark_status=$?
  set -e
  if [[ ! -s "$result" ]]; then
    echo "fused mask64/gain-hash benchmark exited without a terminal receipt" >&2
    exit 70
  fi
  if (( benchmark_status != 0 && benchmark_status != 2 )); then
    echo "fused mask64/gain-hash benchmark had unexpected exit $benchmark_status" >&2
    exit "$benchmark_status"
  fi
  "$python_bin" - "$result" "$benchmark_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("fused mask64/gain-hash receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("fused mask64/gain-hash status and gate disagree")
PY
  if (( benchmark_status == 0 )); then
    printf 'pass\n' > "$state/MASK64_GAIN_HASH_FUSED_PASS"
  else
    printf 'valid rejection\n' > "$state/MASK64_GAIN_HASH_FUSED_REJECTED"
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
    "schema": "qwen38_27b_mask64_gain_hash_fused_watch_receipt_v1",
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
