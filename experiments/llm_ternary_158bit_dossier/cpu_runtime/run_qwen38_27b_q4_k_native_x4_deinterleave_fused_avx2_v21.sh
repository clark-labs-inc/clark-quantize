#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-row-scale-v3}
control_build=${CONTROL_BUILD:-/workspace/build/llama.cpp-q4k-fused-ragged-avx2-v2}
candidate_source=${CANDIDATE_SOURCE:-/workspace/src/llama.cpp-q4k-native-x4-deinterleave-v21}
candidate_build=${CANDIDATE_BUILD:-/workspace/build/llama.cpp-q4k-native-x4-deinterleave-v21}
target=${TARGET:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q4-k-native-x4-deinterleave-v21}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-q4-k-native-x4-deinterleave-v21}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_q4_k_native_x4_deinterleave_fused_avx2_v21.json"
amendment="$runtime_dir/qwen38_27b_q4_k_native_x4_deinterleave_fused_avx2_v21_layout_amendment.json"
candidate_patch="$runtime_dir/kernels/llama_cpp_q4_k_native_x4_deinterleave_fused_avx2_v21.patch"
layout_test="$runtime_dir/tests/test_q8_k_x4_deinterleave_v21.cpp"
summarizer="$runtime_dir/summarize_qwen38_27b_q4_k_native_x4_deinterleave_fused_avx2_v21.py"

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

require_sha256 "$prereg" 22920fe4bc2dbb9c4ee109b207c93e8c7f00a62c3f141221cc53bf8904cb8c30
require_sha256 "$amendment" 853a4fbe87b3494a7e5a9af7fe67f3935ce621a06bfb9f2cdb835e7f4a062df0
require_sha256 "$candidate_patch" feedcb6bcf6b0ded4181a4cc234d8cef08731234ed494f31b43b107884531b69
require_sha256 "$layout_test" a9f9695e27aadf3f2a8dc4362e7cc55f12d5c20e72efcd8d6c912116a9c91ced
require_sha256 "$target" 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
test "$(git -C "$candidate_source" rev-parse HEAD)" = 4df29be4f4c3673f428170fda944a5b19f743bb8
git -C "$candidate_source" diff --check

for binary in \
  "$control_build/bin/llama-bench" \
  "$candidate_build/bin/llama-bench" \
  "$candidate_build/bin/test-backend-ops"; do
  test -x "$binary"
done

g++ -std=c++17 -O2 -march=native \
  -I"$candidate_source/ggml/src" \
  -I"$candidate_source/ggml/include" \
  -I"$candidate_source/ggml/src/ggml-cpu" \
  "$layout_test" \
  -L"$candidate_build/bin" -Wl,-rpath,"$candidate_build/bin" \
  -lggml-cpu -lggml-base \
  -o "$state_dir/test-q8-k-x4-deinterleave-v21"

taskset -c 0 "$state_dir/test-q8-k-x4-deinterleave-v21" \
  > "$state_dir/layout-property.log" 2>&1

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
    order=(control candidate)
  else
    order=(candidate control)
  fi
  for label in "${order[@]}"; do
    if [[ "$label" == control ]]; then
      binary="$control_build/bin/llama-bench"
    else
      binary="$candidate_build/bin/llama-bench"
    fi
    run_graph "pair-${pair}-${label}" "$binary" "$state_dir/pair-${pair}-${label}.json"
  done
done

cleanup
trap - EXIT

status=0
/workspace/venvs/quant-cpu/bin/python "$summarizer" \
  --stage graph --state "$state_dir" \
  --property-log "$state_dir/layout-property.log" \
  --backend-log "$state_dir/backend-q4k-n4.log" \
  --frequency-log "$state_dir/frequency.log" \
  --preregistration "$prereg" --layout-amendment "$amendment" \
  --candidate-patch "$candidate_patch" --layout-test "$layout_test" \
  --control-binary "$control_build/bin/llama-bench" \
  --candidate-binary "$candidate_build/bin/llama-bench" \
  --out "$state_dir/graph-decision.json" || status=$?

if [[ "$status" -ne 0 ]]; then
  cp "$state_dir/graph-decision.json" "$state_dir/terminal-decision.json"
  printf 'complete\n' > "$state_dir/EXPERIMENT_COMPLETE"
  exit "$status"
fi

printf 'complete\n' > "$state_dir/GRAPH_COMPLETE"
jq '{event:"graph_gate_complete",paired_ratio_median,candidate_process_median_mad_fraction,checks}' \
  "$state_dir/graph-decision.json"
