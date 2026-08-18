#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-row-scale-v3}
llama_seed=${LLAMA_SEED:-/workspace/src/llama.cpp}
control_source=${CONTROL_SOURCE:-/workspace/src/llama.cpp-q4k-ragged-control}
candidate_source=${CANDIDATE_SOURCE:-/workspace/src/llama.cpp-q4k-ragged-candidate}
control_build=${CONTROL_BUILD:-/workspace/build/llama.cpp-q4k-ragged-control}
candidate_build=${CANDIDATE_BUILD:-/workspace/build/llama.cpp-q4k-ragged-candidate}
target=${TARGET:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
draft=${DRAFT:-/workspace/models/controls/mtp-Qwen3.8-27B-Q4_0.gguf}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-q4-k-ragged-kernel-v1}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q4-k-ragged-kernel-v1}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
patch="$runtime_dir/kernels/llama_cpp_q4_k_ragged_rows_v1.patch"
base_revision=4df29be4f4c3673f428170fda944a5b19f743bb8
patch_sha=7ed408aaa908f070f95ba2a567175b196c0e1619fb0b987fc1faded282e211d3
target_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
draft_sha=051a1764cff8c4f3ee6ae8b00593a0364c7539c67fa50ffc58f3f96509fca38e

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

prepare_worktree() {
  local path=$1
  if [[ ! -d "$path/.git" && ! -f "$path/.git" ]]; then
    git -C "$llama_seed" worktree add --detach "$path" "$base_revision"
  fi
  if [[ "$(git -C "$path" rev-parse HEAD)" != "$base_revision" ]]; then
    echo "unexpected llama.cpp revision in $path" >&2
    exit 2
  fi
}

mkdir -p "$run_dir" "$state_dir"
cd "$repo"
if [[ -e "$state_dir/KERNEL_COMPLETE" ]]; then
  /workspace/venvs/quant-cpu/bin/python - "$state_dir/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
print(json.dumps({"event": "terminal_resumed", "passed": summary["passed"]}))
raise SystemExit(0 if summary["passed"] else 2)
PY
fi
if [[ -n "$(git status --short)" ]]; then
  echo "kernel experiment requires a clean Clark source checkout" >&2
  exit 2
fi
if pgrep -af 'llama-(server|cli|bench|perplexity|quantize).*Qwen3[.]8-27B' >/dev/null; then
  echo "another Qwen3.8 workload is active" >&2
  exit 2
fi

require_sha256 "$patch" "$patch_sha"
require_sha256 "$target" "$target_sha"
require_sha256 "$draft" "$draft_sha"
prepare_worktree "$control_source"
prepare_worktree "$candidate_source"
if ! git -C "$control_source" diff --quiet || ! git -C "$control_source" diff --cached --quiet; then
  echo "control llama.cpp worktree is dirty" >&2
  exit 2
fi
if git -C "$candidate_source" apply --reverse --check "$patch" >/dev/null 2>&1; then
  :
else
  if ! git -C "$candidate_source" diff --quiet || ! git -C "$candidate_source" diff --cached --quiet; then
    echo "candidate llama.cpp worktree has an unknown diff" >&2
    exit 2
  fi
  git -C "$candidate_source" apply --check "$patch"
  git -C "$candidate_source" apply "$patch"
fi
git -C "$candidate_source" diff --check

cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_NATIVE=ON
  -DGGML_OPENMP=ON
  -DGGML_BLAS=OFF
  -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TESTS=ON
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_EXAMPLES=OFF
)
cmake -S "$control_source" -B "$control_build" "${cmake_args[@]}"
cmake --build "$control_build" --target test-backend-ops llama-server -j 4
cmake -S "$candidate_source" -B "$candidate_build" "${cmake_args[@]}"
cmake --build "$candidate_build" --target test-backend-ops llama-server -j 4

git -C "$control_source" rev-parse HEAD > "$state_dir/llama.commit"
git rev-parse HEAD > "$state_dir/source.commit"
sha256sum "$patch" "$control_build/bin/test-backend-ops" \
  "$candidate_build/bin/test-backend-ops" "$control_build/bin/llama-server" \
  "$candidate_build/bin/llama-server" > "$state_dir/inputs.sha256"
lscpu --json > "$state_dir/lscpu.json"

perf_filter='q4_K.*m=4096,n=(1|2|3|4|5|8),k=14336'
run_perf() {
  local label=$1
  local binary=$2
  local index=$3
  local output="$state_dir/${label}-perf-${index}.log"
  if [[ ! -s "$output" ]]; then
    OMP_NUM_THREADS=4 taskset -c 0,2,4,6 "$binary" perf -b CPU \
      -o MUL_MAT -p "$perf_filter" > "$output.tmp" 2>&1
    mv "$output.tmp" "$output"
  fi
}

run_perf control "$control_build/bin/test-backend-ops" 1
run_perf candidate "$candidate_build/bin/test-backend-ops" 1
run_perf candidate "$candidate_build/bin/test-backend-ops" 2
run_perf control "$control_build/bin/test-backend-ops" 2
printf 'complete\n' > "$state_dir/MICROBENCH_COMPLETE"

mtp_runner="$runtime_dir/run_qwen38_27b_mtp_parity_v1.py"
if [[ ! -s "$run_dir/control-mtp/summary.json" ]]; then
  /workspace/venvs/quant-cpu/bin/python "$mtp_runner" \
    --server "$control_build/bin/llama-server" --target "$target" --draft "$draft" \
    --output-dir "$run_dir/control-mtp" --widths 0,3 --n-predict 96 \
    --threads 4 --cpu-list 0,2,4,6 --context 512 --batch 128 --ubatch 64 \
    --port-base 18400 > "$run_dir/control-mtp.log" 2>&1
fi
if [[ ! -s "$run_dir/candidate-mtp/summary.json" ]]; then
  /workspace/venvs/quant-cpu/bin/python "$mtp_runner" \
    --server "$candidate_build/bin/llama-server" --target "$target" --draft "$draft" \
    --output-dir "$run_dir/candidate-mtp" --widths 0,3 --n-predict 96 \
    --threads 4 --cpu-list 0,2,4,6 --context 512 --batch 128 --ubatch 64 \
    --port-base 18500 > "$run_dir/candidate-mtp.log" 2>&1
fi

set +e
/workspace/venvs/quant-cpu/bin/python \
  "$runtime_dir/summarize_qwen38_27b_q4_k_ragged_kernel_v1.py" \
  --control-perf "$state_dir/control-perf-1.log" \
  --control-perf "$state_dir/control-perf-2.log" \
  --candidate-perf "$state_dir/candidate-perf-1.log" \
  --candidate-perf "$state_dir/candidate-perf-2.log" \
  --control-mtp "$run_dir/control-mtp" --candidate-mtp "$run_dir/candidate-mtp" \
  --control-server "$control_build/bin/llama-server" \
  --candidate-server "$candidate_build/bin/llama-server" \
  --out "$state_dir/summary.json"
status=$?
set -e
printf 'complete\n' > "$state_dir/KERNEL_COMPLETE"
exit "$status"
