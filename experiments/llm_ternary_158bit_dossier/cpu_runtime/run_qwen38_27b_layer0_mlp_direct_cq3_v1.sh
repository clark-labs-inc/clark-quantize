#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
baseline=${BASELINE_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
candidate=${CANDIDATE_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M-layer0-MLP-CQ3_0.gguf}
llama_base=${LLAMA_BASE:-/workspace/src/llama.cpp}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp-qwen38-selective-v1}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-qwen38-selective-v1}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-layer0-mlp-direct-cq3-v1}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-v1}
baseline_state=${BASELINE_STATE:-/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_layer0_mlp_direct_cq3_preregistration_v1.json"
cq3_patch="$runtime_dir/llama_cpp_4df29be_cq3.patch"
selective_patch="$runtime_dir/llama_cpp_4df29be_selective_copy_v1.patch"
scope_verifier="$runtime_dir/verify_gguf_selective_conversion.py"
target_regex='^blk\.0\.ffn_(down|gate|up)\.weight$'
expected_baseline_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
expected_baseline_bytes=18973870432
expected_candidate_bytes=18894489952
expected_llama_revision=4df29be4f4c3673f428170fda944a5b19f743bb8

require_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

apply_once() {
  local patch=$1
  local sentinel=$2
  if rg -q "$sentinel" "$llama_source"; then
    return
  fi
  git -C "$llama_source" apply --check "$patch"
  git -C "$llama_source" apply "$patch"
}

mkdir -p "$run_dir" "$state_dir" "$(dirname "$candidate")" "$(dirname "$llama_source")" "$(dirname "$llama_build")"
cd "$repo"
if [[ -e "$state_dir/CANDIDATE_READY" ]]; then
  (cd / && sha256sum -c "$state_dir/artifacts.sha256")
  echo "candidate marker and artifact hashes are valid: $state_dir/CANDIDATE_READY"
  exit 0
fi
if [[ -n "$(git status --short)" ]]; then
  echo "27B selective conversion requires a clean research checkout" >&2
  exit 2
fi
if [[ ! -s "$baseline_state/QUALITY_COMPLETE" || ! -s "$baseline_state/summary.json" ]]; then
  echo "the pinned 27B Q4 quality baseline is not complete; refusing to build the candidate" >&2
  exit 3
fi
if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a frozen Qwen 4B service is active; refusing to share the node" >&2
  exit 2
fi
if pgrep -af 'llama-(server|cli|bench|perplexity)' | grep -v -F "$0" > "$run_dir/conflicting-processes.log"; then
  echo "a llama.cpp model process is active; refusing concurrent conversion" >&2
  exit 2
fi

require_sha256 "$prereg" dfba2e0b7aed4b5654a1c3f5c561a363b6e9496b86920973c2820b162f89740c
require_sha256 "$cq3_patch" b5c584a338018ff7e509bc160c195693102fb6e265b6cbe4935d6e54d7cd67c1
require_sha256 "$selective_patch" 19a40bf56d1ed3d1af6ea12d941891a75252b8b6541229c21db8102b2dd2e7c1
require_sha256 "$scope_verifier" 3a2a864ad8d93b28154c04bddf2410a2f22c6887c373abea08c84c78cd526cce
if [[ "$(stat -c %s "$baseline")" != "$expected_baseline_bytes" ]]; then
  echo "unexpected baseline byte count" >&2
  exit 2
fi
require_sha256 "$baseline" "$expected_baseline_sha"

if [[ ! -e "$llama_source/.git" ]]; then
  git -C "$llama_base" worktree add --detach "$llama_source" "$expected_llama_revision"
fi
if [[ "$(git -C "$llama_source" rev-parse HEAD)" != "$expected_llama_revision" ]]; then
  echo "unexpected llama.cpp candidate worktree revision" >&2
  exit 2
fi
apply_once "$cq3_patch" 'GGML_TYPE_CQ3_0'
apply_once "$selective_patch" 'applying selective COPY override'
git -C "$llama_source" diff --check

cmake -S "$llama_source" -B "$llama_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  > "$run_dir/cmake-configure.log" 2>&1
cmake --build "$llama_build" --parallel 4 \
  --target llama-quantize llama-cli llama-server llama-perplexity llama-bench test-cq3 \
  > "$run_dir/cmake-build.log" 2>&1
LD_LIBRARY_PATH="$llama_build/bin" "$llama_build/bin/test-cq3" \
  > "$run_dir/test-cq3.log" 2>&1

git -C "$repo" rev-parse HEAD > "$state_dir/source.commit"
git -C "$llama_source" rev-parse HEAD > "$state_dir/llama.commit"
git -C "$llama_source" diff --binary > "$state_dir/llama-source.diff"
sha256sum "$prereg" "$cq3_patch" "$selective_patch" "$scope_verifier" \
  "$llama_build/bin/llama-quantize" "$llama_build/bin/llama-cli" \
  "$llama_build/bin/llama-server" "$llama_build/bin/llama-perplexity" \
  "$llama_build/bin/llama-bench" "$llama_build/bin/test-cq3" \
  "$state_dir/llama-source.diff" \
  > "$state_dir/inputs.sha256"

if [[ ! -s "$candidate" ]]; then
  "$python_bin" - "$candidate" <<'PY'
import os
import sys
from pathlib import Path

target = Path(sys.argv[1])
available = os.statvfs(target.parent).f_bavail * os.statvfs(target.parent).f_frsize
required = 22 * 1024**3
if available < required:
    raise SystemExit(f"need at least {required} free bytes before conversion, have {available}")
PY
  partial="${candidate}.partial"
  if [[ -e "$partial" ]]; then
    echo "uncommitted partial candidate requires inspection before retry: $partial" >&2
    exit 2
  fi
  /usr/bin/time -v "$llama_build/bin/llama-quantize" \
    --allow-requantize \
    --tensor-type "${target_regex}=CQ3_0" \
    "$baseline" "$partial" COPY 4 \
    > "$run_dir/conversion.log" 2>&1
  sync "$partial"
  mv "$partial" "$candidate"
fi

if [[ "$(stat -c %s "$candidate")" != "$expected_candidate_bytes" ]]; then
  echo "unexpected candidate byte count" >&2
  exit 2
fi
"$python_bin" - "$run_dir/conversion.log" <<'PY'
import re
import sys
from pathlib import Path

expected = {
    "blk.0.ffn_down.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
}
text = Path(sys.argv[1]).read_text()
actual = set(re.findall(r"(blk\.0\.ffn_(?:down|gate|up)\.weight)\s+- applying selective COPY override", text))
if actual != expected:
    raise SystemExit(f"unexpected selective override set: {sorted(actual)}")
if text.count("applying selective COPY override") != 3:
    raise SystemExit("conversion log does not contain exactly three selective overrides")
PY

PYTHONPATH="$llama_source/gguf-py" "$python_bin" "$scope_verifier" \
  --baseline "$baseline" \
  --candidate "$candidate" \
  --target-regex "$target_regex" \
  --expected-target-count 3 \
  --baseline-target-type 12 \
  --candidate-target-type 43 \
  --out "$state_dir/gguf-scope.json" \
  > "$run_dir/gguf-scope.log" 2>&1

/usr/bin/time -v "$llama_build/bin/llama-cli" \
  -m "$candidate" -t 4 -tb 4 -c 2048 -b 512 -ub 512 \
  --no-warmup --single-turn --simple-io \
  -p 'Reply with only OK.' -n 2 --temp 0 \
  > "$state_dir/load-smoke.log" 2>&1
grep -Eq 'Exit status:[[:space:]]+0$' "$state_dir/load-smoke.log"
grep -Eq 'Swaps:[[:space:]]+0$' "$state_dir/load-smoke.log"

"$python_bin" - "$prereg" "$baseline_state/summary.json" "$state_dir/gguf-scope.json" \
  "$candidate" "$llama_build" "$state_dir" <<'PY'
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

prereg, baseline_summary, scope_path, candidate, build, state = map(Path, sys.argv[1:])
scope = json.loads(scope_path.read_text())
if not scope["valid"]:
    raise SystemExit("scope receipt is invalid")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

receipt = {
    "schema": "qwen38_27b_layer0_mlp_direct_cq3_conversion_receipt_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "preregistration": {"path": str(prereg), "sha256": sha256(prereg)},
    "baseline_summary": {
        "path": str(baseline_summary),
        "sha256": sha256(baseline_summary),
    },
    "candidate": {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    },
    "scope_receipt": {"path": str(scope_path), "sha256": sha256(scope_path)},
    "binaries": {
        name: sha256(build / "bin" / name)
        for name in ("llama-quantize", "llama-cli", "llama-server", "llama-perplexity", "llama-bench", "test-cq3")
    },
    "host": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    },
    "checks": {
        "scope_valid": scope["valid"],
        "target_count": scope["target_count"] == 3,
        "non_target_count": scope["non_target_count"] == 848,
        "candidate_bytes": candidate.stat().st_size == 18894489952,
        "load_smoke_exit_zero": "Exit status: 0" in (state / "load-smoke.log").read_text(),
        "load_smoke_no_swap": "Swaps: 0" in (state / "load-smoke.log").read_text(),
    },
    "claim_boundary": "direct Q4-to-CQ3 selective PTQ of one Qwen3.8-27B MLP triplet; quality is not yet established",
}
receipt["valid"] = all(receipt["checks"].values())
temporary = state / "conversion-receipt.json.tmp"
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(state / "conversion-receipt.json")
if not receipt["valid"]:
    raise SystemExit("conversion receipt is invalid")
PY

sha256sum "$candidate" "$state_dir/gguf-scope.json" \
  "$state_dir/load-smoke.log" "$state_dir/conversion-receipt.json" \
  > "$state_dir/artifacts.sha256"
printf 'complete\n' > "$state_dir/CANDIDATE_READY"
