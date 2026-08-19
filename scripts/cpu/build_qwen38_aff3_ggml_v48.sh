#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 LLAMA_SOURCE LLAMA_BUILD OUTPUT" >&2
  exit 2
fi

llama_source=$1
llama_build=$2
output=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$script_dir" rev-parse --show-toplevel)
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
source_file="$runtime/qwen38_aff3_ggml_v48.cpp"

mkdir -p "$(dirname "$output")"
g++ \
  -std=c++17 \
  -O3 \
  -DNDEBUG \
  -mavx2 \
  -mfma \
  -mf16c \
  -mssse3 \
  -Wall \
  -Wextra \
  -Werror \
  -I "$llama_source/ggml/include" \
  -I "$runtime" \
  "$source_file" \
  -L "$llama_build/bin" \
  -Wl,-rpath,"$llama_build/bin" \
  -lggml-cpu \
  -lggml-base \
  -lggml \
  -pthread \
  -o "$output"
