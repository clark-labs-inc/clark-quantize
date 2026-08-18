#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-row-scale-v3}
control_build=${CONTROL_BUILD:-/workspace/build/llama.cpp-q4k-fused-ragged-avx2-v2}
parent_build=${PARENT_BUILD:-/workspace/build/llama.cpp-q4k-native-x4-deinterleave-v21}
candidate_source=${CANDIDATE_SOURCE:-/workspace/src/llama.cpp-q4k-dual-layout-producer-v24}
candidate_build=${CANDIDATE_BUILD:-/workspace/build/llama.cpp-q4k-dual-layout-producer-v24}
target=${TARGET:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q4-k-dual-layout-producer-v24}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-q4-k-dual-layout-producer-v24}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_q4_k_dual_layout_q8_producer_avx2_v24.json"
execution_amendment="$runtime_dir/qwen38_27b_q4_k_dual_layout_q8_producer_avx2_v24_execution_amendment.json"
candidate_patch="$runtime_dir/kernels/llama_cpp_q4_k_dual_layout_q8_producer_avx2_v24.patch"
property_test="$runtime_dir/tests/test_q8_k_dual_layout_producer_v24.cpp"
microbenchmark="$runtime_dir/tests/benchmark_q8_k_dual_layout_producer_v24.cpp"
summarizer="$runtime_dir/summarize_qwen38_27b_q4_k_dual_layout_q8_producer_avx2_v24.py"

require_sha256() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

run_graph() {
  local label=$1 binary=$2 output=$3
  if [[ ! -s "$output" ]]; then
    OMP_NUM_THREADS=4 taskset -c 0,2,4,6 "$binary" \
      -m "$target" -p 4 -n 0 -r 7 -b 128 -ub 64 -t 4 -o json \
      > "$output.tmp" 2> "$state_dir/${label}.stderr"
    mv "$output.tmp" "$output"
  fi
}

mkdir -p "$state_dir" "$run_dir"

if [[ -s "$state_dir/EXPERIMENT_COMPLETE" && -s "$state_dir/terminal-decision.json" ]]; then
  jq '{event:"terminal_resumed",passed:.passed}' "$state_dir/terminal-decision.json"
  jq -e '.passed == true' "$state_dir/terminal-decision.json" >/dev/null
  exit $?
fi
if [[ -s "$state_dir/GRAPH_COMPLETE" && -s "$state_dir/graph-decision.json" ]]; then
  jq '{event:"graph_resumed",passed:.passed}' "$state_dir/graph-decision.json"
  jq -e '.passed == true' "$state_dir/graph-decision.json" >/dev/null
  exit $?
fi
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null || pgrep -x llama-perplexity >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi

require_sha256 "$prereg" d5bb4a17b7af476c0ec284bdf32f09e59fc9397b4957db41adf1d7a793c9fbd2
require_sha256 "$execution_amendment" 9cb82f44ff82d8676de9b34b163d2ccd506a29bf7c231092b78b0b205cf63e5e
require_sha256 "$candidate_patch" 0c0a8242adc3b1a4a912d8211ad16d42ae28c8004047a4f89733ae0e72a14704
require_sha256 "$property_test" 17698f7333ff9486274f22c9bb9bebbc3f0d7a8fccd4d6c1e719b0c24e0575e7
require_sha256 "$microbenchmark" 98e243c56ad79855faea4dd8d2c27577ee9bd4c5becf2017e9574ec1826c178b
require_sha256 "$summarizer" 3bb345d22bcf125cbd28b438159588e0e600f3da127eab3f58a12b0f8ea92ee8
require_sha256 "$target" 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
test "$(git -C "$candidate_source" rev-parse HEAD)" = 4df29be4f4c3673f428170fda944a5b19f743bb8
git -C "$candidate_source" diff --check
git -C "$candidate_source" apply --reverse --check "$candidate_patch"

for binary in \
  "$control_build/bin/llama-bench" \
  "$parent_build/bin/llama-bench" \
  "$candidate_build/bin/llama-bench" \
  "$candidate_build/bin/test-backend-ops"; do
  test -x "$binary"
done

for source_and_output in \
  "$property_test:$state_dir/test-q8-k-dual-layout-producer-v24" \
  "$microbenchmark:$state_dir/benchmark-q8-k-dual-layout-producer-v24"; do
  source=${source_and_output%%:*}
  output=${source_and_output#*:}
  g++ -std=c++17 -O3 -march=native \
    -I"$candidate_source/ggml/src" \
    -I"$candidate_source/ggml/include" \
    -I"$candidate_source/ggml/src/ggml-cpu" \
    "$source" \
    -L"$candidate_build/bin" -Wl,-rpath,"$candidate_build/bin" \
    -lggml-cpu -lggml-base \
    -o "$output"
done

taskset -c 0 "$state_dir/test-q8-k-dual-layout-producer-v24" \
  > "$state_dir/property.log" 2>&1
taskset -c 0 "$state_dir/benchmark-q8-k-dual-layout-producer-v24" \
  > "$state_dir/microbenchmark.json" 2> "$state_dir/microbenchmark.stderr"

OMP_NUM_THREADS=4 taskset -c 0,2,4,6 "$candidate_build/bin/test-backend-ops" \
  test -b CPU -o MUL_MAT \
  -p 'type_a=q4_K,type_b=f32,m=16,n=4,k=256' \
  > "$state_dir/backend-q4k-n4.log" 2>&1

: > "$state_dir/frequency.log"
(
  while true; do
    awk '
      /cpu MHz/ { sum += $4; count += 1 }
      END { if (count > 0) printf "%s %.6f\n", systime(), sum / count }
    ' /proc/cpuinfo >> "$state_dir/frequency.log"
    sleep 1
  done
) &
frequency_pid=$!
cleanup() {
  kill "$frequency_pid" 2>/dev/null || true
  wait "$frequency_pid" 2>/dev/null || true
}
trap cleanup EXIT

for pair in 1 2 3 4 5 6; do
  if (( pair % 2 == 1 )); then
    parent_order=(v21 v24)
    control_order=(control v24)
  else
    parent_order=(v24 v21)
    control_order=(v24 control)
  fi

  for label in "${parent_order[@]}"; do
    if [[ "$label" == v21 ]]; then
      binary="$parent_build/bin/llama-bench"
    else
      binary="$candidate_build/bin/llama-bench"
    fi
    run_graph "parent-pair-${pair}-${label}" "$binary" \
      "$state_dir/parent-pair-${pair}-${label}.json"
  done

  for label in "${control_order[@]}"; do
    if [[ "$label" == control ]]; then
      binary="$control_build/bin/llama-bench"
    else
      binary="$candidate_build/bin/llama-bench"
    fi
    run_graph "control-pair-${pair}-${label}" "$binary" \
      "$state_dir/control-pair-${pair}-${label}.json"
  done
done

cleanup
trap - EXIT

status=0
/workspace/venvs/quant-cpu/bin/python "$summarizer" \
  --state "$state_dir" \
  --property-log "$state_dir/property.log" \
  --micro-log "$state_dir/microbenchmark.json" \
  --backend-log "$state_dir/backend-q4k-n4.log" \
  --frequency-log "$state_dir/frequency.log" \
  --preregistration "$prereg" \
  --execution-amendment "$execution_amendment" \
  --candidate-patch "$candidate_patch" \
  --property-test "$property_test" \
  --microbenchmark "$microbenchmark" \
  --control-binary "$control_build/bin/llama-bench" \
  --parent-binary "$parent_build/bin/llama-bench" \
  --candidate-binary "$candidate_build/bin/llama-bench" \
  --out "$state_dir/graph-decision.json" || status=$?

if [[ "$status" -ne 0 ]]; then
  cp "$state_dir/graph-decision.json" "$state_dir/terminal-decision.json"
  printf 'complete\n' > "$state_dir/EXPERIMENT_COMPLETE"
  exit "$status"
fi

printf 'complete\n' > "$state_dir/GRAPH_COMPLETE"
jq '{event:"graph_gate_complete",checks,passed,next_stage}' \
  "$state_dir/graph-decision.json"
