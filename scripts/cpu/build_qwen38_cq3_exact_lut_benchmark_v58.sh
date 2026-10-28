#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT" >&2
  exit 2
fi

output=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$script_dir" rev-parse --show-toplevel)
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT

cp "$repo/clark_quantize/qwen38_cq3_exact_lut_cpu.c" "$stage/"
cp "$repo/clark_quantize/qwen38_cq3_exact_lut_cpu.h" "$stage/"
cp "$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime/kernels/benchmark_qwen38_cq3_exact_lut_native_v58.c" "$stage/benchmark.c"

mkdir -p "$(dirname -- "$output")"
cc \
  -std=c11 -O3 -DNDEBUG -mavx2 -mfma -mf16c \
  -Wall -Wextra -Werror \
  -I "$stage" \
  "$stage/qwen38_cq3_exact_lut_cpu.c" \
  "$stage/benchmark.c" \
  -lm \
  -o "$output"
