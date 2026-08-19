#!/usr/bin/env bash
set -euo pipefail

state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q4-k-clang21-holdout-replication-v31}
parent=${PARENT_BIN:-/workspace/build/llama.cpp-q4k-native-x4-deinterleave-v21/bin/llama-bench}
candidate=${CANDIDATE_BIN:-/workspace/build/llama.cpp-q4k-clang21-pair-local-scaled-v30/bin/llama-bench}
model=${MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}

for path in "$parent" "$candidate" "$model"; do
  test -s "$path"
done
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null || \
   pgrep -x llama-perplexity >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi

mkdir -p "$state_dir/graph"
run_one() {
  local phase=$1 pair=$2 label=$3 binary output
  if [[ "$label" == parent ]]; then
    binary=$parent
  else
    binary=$candidate
  fi
  output="$state_dir/graph/${phase}-${pair}-${label}.json"
  if [[ -s "$output" ]]; then
    jq -e 'length == 1 and (.[0].samples_ns | length) == 7' "$output" >/dev/null
    return
  fi
  OMP_NUM_THREADS=4 taskset -c 0,2,4,6 "$binary" \
    -m "$model" -p 4 -n 0 -r 7 -b 128 -ub 64 -t 4 -o json \
    > "$output.tmp" 2> "$state_dir/graph/${phase}-${pair}-${label}.stderr"
  jq -e 'length == 1 and (.[0].samples_ns | length) == 7' "$output.tmp" >/dev/null
  mv "$output.tmp" "$output"
}

run_one warmup 1 parent
run_one warmup 1 candidate

: > "$state_dir/graph/frequency.log"
(
  while true; do
    awk '
      /cpu MHz/ { sum += $4; count += 1 }
      END { if (count > 0) printf "%s %.6f\n", systime(), sum / count }
    ' /proc/cpuinfo >> "$state_dir/graph/frequency.log"
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
    order=(parent candidate)
  else
    order=(candidate parent)
  fi
  for label in "${order[@]}"; do
    run_one measured "$pair" "$label"
  done
done

cleanup
trap - EXIT
printf 'complete\n' > "$state_dir/GRAPH_RAW_COMPLETE"
