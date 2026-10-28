#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-row-scale-v3}
llama_seed=${LLAMA_SEED:-/workspace/src/llama.cpp}
control_source=${CONTROL_SOURCE:-/workspace/src/llama.cpp-q4k-fused-ragged-avx2-v2}
candidate_source=${CANDIDATE_SOURCE:-/workspace/src/llama.cpp-q4k-fused-four-row-avx2-v3}
control_build=${CONTROL_BUILD:-/workspace/build/llama.cpp-q4k-fused-ragged-avx2-v2}
candidate_build=${CANDIDATE_BUILD:-/workspace/build/llama.cpp-q4k-fused-four-row-avx2-v3}
target=${TARGET:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
draft=${DRAFT:-/workspace/models/controls/mtp-Qwen3.8-27B-Q4_0.gguf}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-q4-k-fused-four-row-avx2-v3}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q4-k-fused-four-row-avx2-v3}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
base_patch="$runtime_dir/kernels/llama_cpp_q4_k_ragged_rows_v1.patch"
fused_patch="$runtime_dir/kernels/llama_cpp_q4_k_fused_ragged_avx2_v2.patch"
four_row_patch="$runtime_dir/kernels/llama_cpp_q4_k_fused_four_row_avx2_v3.patch"
base_revision=4df29be4f4c3673f428170fda944a5b19f743bb8

require_sha256() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

prepare_fresh_worktree() {
  local path=$1
  shift
  if [[ ! -e "$path/.git" ]]; then
    git -C "$llama_seed" worktree add --detach "$path" "$base_revision"
    for patch in "$@"; do
      git -C "$path" apply --check "$patch"
      git -C "$path" apply "$patch"
    done
  fi
  test "$(git -C "$path" rev-parse HEAD)" = "$base_revision"
  git -C "$path" diff --check
}

build_runtime() {
  local source=$1 build=$2
  cmake -S "$source" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
    -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF
  cmake --build "$build" --target test-backend-ops llama-bench llama-server -j 4
}

run_graph() {
  local label=$1 binary=$2 output="$state_dir/$label.json"
  if [[ ! -s "$output" ]]; then
    taskset -c 0,2,4,6 "$binary" -m "$target" -p 4 -n 0 -r 7 \
      -b 128 -ub 64 -t 4 -o json > "$output.tmp" 2> "$state_dir/$label.stderr"
    mv "$output.tmp" "$output"
  fi
}

run_mtp() {
  local label=$1 server=$2 output=$3 port=$4 status=0
  if [[ ! -s "$output/summary.json" ]]; then
    /workspace/venvs/quant-cpu/bin/python \
      "$runtime_dir/run_qwen38_27b_mtp_parity_v1.py" \
      --server "$server" --target "$target" --draft "$draft" \
      --output-dir "$output" --widths 0,3 --n-predict 96 \
      --threads 4 --cpu-list 0,2,4,6 --context 512 --batch 128 --ubatch 64 \
      --port-base "$port" > "$run_dir/$label-harness.log" 2>&1 || status=$?
    if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then
      return "$status"
    fi
  fi
  /workspace/venvs/quant-cpu/bin/python - "$output/summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("terminal") is not True:
    raise SystemExit("MTP summary is not terminal")
generation = payload["generation"]
if generation["prompt_count"] != 4 or generation["tokens_per_prompt"] != 96:
    raise SystemExit("MTP denominator is not 4 x 96")
PY
}

mkdir -p "$run_dir" "$state_dir"
cd "$repo"
if [[ -s "$state_dir/terminal-decision.json" && -e "$state_dir/EXPERIMENT_COMPLETE" ]]; then
  jq '{event:"terminal_resumed",passed:.passed}' "$state_dir/terminal-decision.json"
  jq -e '.passed == true' "$state_dir/terminal-decision.json" >/dev/null
  exit $?
fi
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi

require_sha256 "$base_patch" 7ed408aaa908f070f95ba2a567175b196c0e1619fb0b987fc1faded282e211d3
require_sha256 "$fused_patch" 4fab46f823b5980f7292c101d3051719a633b9dda2db304c10c8836d4b156b2e
require_sha256 "$four_row_patch" 41fac22043304854d4b21562d3b5fb6c5bbbc880b339a4af9c7aae8bb849c5ab
require_sha256 "$target" 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
require_sha256 "$draft" 051a1764cff8c4f3ee6ae8b00593a0364c7539c67fa50ffc58f3f96509fca38e

prepare_fresh_worktree "$control_source" "$base_patch" "$fused_patch"
prepare_fresh_worktree "$candidate_source" "$base_patch" "$fused_patch" "$four_row_patch"
build_runtime "$control_source" "$control_build"
build_runtime "$candidate_source" "$candidate_build"

OMP_NUM_THREADS=4 taskset -c 0,2,4,6 "$candidate_build/bin/test-backend-ops" \
  test -b CPU -o MUL_MAT \
  -p 'type_a=q4_K,type_b=f32,m=16,n=4,k=256' \
  > "$state_dir/backend-q4k-n4.log" 2>&1
test "$(grep -c 'MUL_MAT.*OK' "$state_dir/backend-q4k-n4.log")" -eq 3
grep -q '3/3 tests passed' "$state_dir/backend-q4k-n4.log"

run_graph control-1 "$control_build/bin/llama-bench"
run_graph candidate-1 "$candidate_build/bin/llama-bench"
run_graph candidate-2 "$candidate_build/bin/llama-bench"
run_graph control-2 "$control_build/bin/llama-bench"

run_mtp control "$control_build/bin/llama-server" "$run_dir/control" 19000
run_mtp candidate "$candidate_build/bin/llama-server" "$run_dir/candidate" 19100

status=0
/workspace/venvs/quant-cpu/bin/python \
  "$runtime_dir/summarize_qwen38_27b_q4_k_fused_four_row_avx2_v3.py" \
  --control-bench "$state_dir/control-1.json" \
  --control-bench "$state_dir/control-2.json" \
  --candidate-bench "$state_dir/candidate-1.json" \
  --candidate-bench "$state_dir/candidate-2.json" \
  --backend-log "$state_dir/backend-q4k-n4.log" \
  --control-mtp "$run_dir/control" --candidate-mtp "$run_dir/candidate" \
  --control-server "$control_build/bin/llama-server" \
  --candidate-server "$candidate_build/bin/llama-server" \
  --base-patch "$base_patch" --fused-patch "$fused_patch" \
  --four-row-patch "$four_row_patch" \
  --out "$state_dir/terminal-decision.json" || status=$?
printf 'complete\n' > "$state_dir/EXPERIMENT_COMPLETE"
exit "$status"
