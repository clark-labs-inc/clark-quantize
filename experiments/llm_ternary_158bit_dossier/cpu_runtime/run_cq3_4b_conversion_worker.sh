#!/usr/bin/env bash
set -euo pipefail

input=${1:?input BF16 GGUF is required}
output=${2:?output CQ3 GGUF is required}
run_dir=${3:?run directory is required}
quantize_bin=${QUANTIZE_BIN:-/workspace/build/llama.cpp/bin/llama-quantize}
threads=${QUANTIZE_THREADS:-4}
partial="${output}.partial"

mkdir -p "$run_dir"

if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 1
fi

# The partial path is exact and dedicated to this output. Removing it makes a
# failed conversion restartable without risking any completed artifact.
rm -f "$partial"

/usr/bin/time -v "$quantize_bin" --pure \
    --tensor-type 'ssm_(alpha|beta)\.weight$=BF16' \
    --tensor-type 'nextn\..*weight$=BF16' \
    "$input" "$partial" CQ3_0 "$threads" \
    > "$run_dir/quantize.log" 2>&1

mv "$partial" "$output"
sha256sum "$output" > "$run_dir/model.sha256"
stat --printf='%n %s bytes\n' "$output" > "$run_dir/model.stat"
