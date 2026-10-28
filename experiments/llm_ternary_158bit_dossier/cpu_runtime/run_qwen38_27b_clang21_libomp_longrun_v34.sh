#!/usr/bin/env bash
set -euo pipefail

state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-clang21-libomp-longrun-v34}
parent=${PARENT_BIN:-/workspace/build/llama.cpp-q4k-native-x4-deinterleave-v21/bin/llama-bench}
candidate=${CANDIDATE_BIN:-/workspace/build/llama.cpp-q4k-clang21-pair-local-scaled-v30/bin/llama-bench}
model=${MODEL:-/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-imatrix-Q4_K-v10.gguf}

expected_parent=90a08eb2f1a70d540c117b9db563c28fa9bb0b03eedca48d061cc147aab79a28
expected_candidate=7a01aa1a0a866a2904e6ca09b42d501372a408503948f1880acc19fcfc356b70
expected_model=6176472b7d039a6d911b34eea78dd4507256b4a22b3003d40b8587b711014f2a

for path in "$parent" "$candidate" "$model"; do
  test -s "$path"
done
test "$(sha256sum "$parent" | awk '{print $1}')" = "$expected_parent"
test "$(sha256sum "$candidate" | awk '{print $1}')" = "$expected_candidate"
test "$(sha256sum "$model" | awk '{print $1}')" = "$expected_model"
if pgrep -f '/llama-(server|bench|perplexity)( |$)' >/dev/null || pgrep -x perf >/dev/null; then
  echo "another benchmark or perf process is active" >&2
  exit 2
fi

mkdir -p "$state_dir"
test ! -e "$state_dir/LONGRUN_COMPLETE"
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

run_one() {
  local label=$1
  local binary=$2
  local policy=$3
  local output="$state_dir/$label.json"
  local stderr="$state_dir/$label.stderr"
  test ! -e "$output"
  if [[ "$policy" == active ]]; then
    OMP_NUM_THREADS=4 KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active KMP_LIBRARY=turnaround \
      taskset -c 0,2,4,6 "$binary" \
        -m "$model" -p 4 -n 0 -r 32 -b 128 -ub 64 -t 4 -o json \
        > "$output.tmp" 2> "$stderr"
  else
    OMP_NUM_THREADS=4 \
      taskset -c 0,2,4,6 "$binary" \
        -m "$model" -p 4 -n 0 -r 32 -b 128 -ub 64 -t 4 -o json \
        > "$output.tmp" 2> "$stderr"
  fi
  jq -e 'length == 1 and (.[0].samples_ns | length) == 32' "$output.tmp" >/dev/null
  mv "$output.tmp" "$output"
}

run_one parent_a "$parent" default
run_one clang_default_a "$candidate" default
run_one clang_active_a "$candidate" active
run_one clang_active_b "$candidate" active
run_one clang_default_b "$candidate" default
run_one parent_b "$parent" default

cleanup
trap - EXIT
sha256sum "$state_dir"/*.json "$state_dir"/*.stderr "$state_dir/frequency.log" \
  > "$state_dir/sealed-inputs.sha256"
printf 'complete\n' > "$state_dir/LONGRUN_COMPLETE"
