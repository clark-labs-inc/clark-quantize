#!/usr/bin/env bash
set -euo pipefail

state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q4-k-out-of-line-row-pair2-v29}
parent=${PARENT_BIN:-$state_dir/verify-parent}
candidate=${CANDIDATE_BIN:-$state_dir/verify-candidate}
eviction_bytes=${EVICTION_BYTES:-268435456}
warmup_pairs=${WARMUP_PAIRS:-2}
measured_pairs=${MEASURED_PAIRS:-31}

shapes=(
  "10240:5120"
  "6144:5120"
  "5120:6144"
  "12288:5120"
  "1024:5120"
)

for binary in "$parent" "$candidate"; do
  test -x "$binary"
done
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null || \
   pgrep -x llama-perplexity >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi

mkdir -p "$state_dir/micro"
: > "$state_dir/micro/frequency.log"
(
  while true; do
    awk '
      /cpu MHz/ { sum += $4; count += 1 }
      END { if (count > 0) printf "%s %.6f\n", systime(), sum / count }
    ' /proc/cpuinfo >> "$state_dir/micro/frequency.log"
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
  local shape_id=$1 m=$2 k=$3 phase=$4 pair=$5 label=$6 binary output
  if [[ "$label" == parent ]]; then
    binary=$parent
  else
    binary=$candidate
  fi
  output="$state_dir/micro/${shape_id}-${phase}-${pair}-${label}.json"
  if [[ -s "$output" ]]; then
    jq -e '.mode == "benchmark" and (.samples_us | length) == 1' "$output" >/dev/null
    return
  fi
  OMP_NUM_THREADS=1 taskset -c 0 "$binary" \
    --mode benchmark --m "$m" --k "$k" --warmups 0 --repetitions 1 \
    --eviction-bytes "$eviction_bytes" > "$output.tmp"
  jq -e '.mode == "benchmark" and (.samples_us | length) == 1' "$output.tmp" >/dev/null
  mv "$output.tmp" "$output"
}

for shape in "${shapes[@]}"; do
  IFS=: read -r m k <<< "$shape"
  shape_id="m${m}-n4-k${k}"
  for pair in $(seq 1 "$warmup_pairs"); do
    if (( pair % 2 == 1 )); then
      order=(parent candidate)
    else
      order=(candidate parent)
    fi
    for label in "${order[@]}"; do
      run_one "$shape_id" "$m" "$k" warmup "$pair" "$label"
    done
  done
  for pair in $(seq 1 "$measured_pairs"); do
    if (( pair % 2 == 1 )); then
      order=(parent candidate)
    else
      order=(candidate parent)
    fi
    for label in "${order[@]}"; do
      run_one "$shape_id" "$m" "$k" measured "$pair" "$label"
    done
  done
done

cleanup
trap - EXIT
printf 'complete\n' > "$state_dir/MICRO_RAW_COMPLETE"
