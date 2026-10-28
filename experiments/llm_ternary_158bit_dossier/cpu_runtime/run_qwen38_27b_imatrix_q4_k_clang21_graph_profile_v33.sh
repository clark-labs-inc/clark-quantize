#!/usr/bin/env bash
set -euo pipefail

state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-imatrix-q4-k-clang21-graph-profile-v33}
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
if pgrep -f '/llama-(server|bench|perplexity)( |$)' >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi

mkdir -p "$state_dir"
test ! -e "$state_dir/PROFILE_RAW_COMPLETE"
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

for label in parent candidate; do
  if [[ "$label" == parent ]]; then
    binary=$parent
  else
    binary=$candidate
  fi
  data="$state_dir/$label.perf.data"
  output="$state_dir/$label.bench.json"
  stderr="$state_dir/$label.perf.stderr"
  report="$state_dir/$label.perf-report.csv"
  hash="$state_dir/$label.perf-data.sha256"
  test ! -e "$output" && test ! -e "$report" && test ! -e "$hash"
  OMP_NUM_THREADS=4 perf record \
    -e cycles:P -F 999 --call-graph fp -o "$data" -- \
    taskset -c 0,2,4,6 "$binary" \
      -m "$model" -p 4 -n 0 -r 32 -b 128 -ub 64 -t 4 -o json \
      > "$output.tmp" 2> "$stderr"
  jq -e 'length == 1 and (.[0].samples_ns | length) == 32' "$output.tmp" >/dev/null
  mv "$output.tmp" "$output"
  perf report -i "$data" --stdio --stdio-color never --no-children \
    --percent-limit 0.01 -t ';' -F overhead,sample,dso,symbol > "$report.tmp"
  mv "$report.tmp" "$report"
  sha256sum "$data" > "$hash"
  rm -f "$data"
done

cleanup
trap - EXIT
sha256sum \
  "$model" "$parent" "$candidate" \
  "$state_dir/frequency.log" \
  "$state_dir/parent.bench.json" "$state_dir/candidate.bench.json" \
  "$state_dir/parent.perf-report.csv" "$state_dir/candidate.perf-report.csv" \
  > "$state_dir/sealed-inputs.sha256"
printf 'complete\n' > "$state_dir/PROFILE_RAW_COMPLETE"
