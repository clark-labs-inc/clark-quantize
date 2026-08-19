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
model_source="$llama_source/src/models"
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT

test -f "$runtime/llama_cpp/qwen38-aff3-runtime-smoke.cpp"
test -f "$model_source/qwen38-aff3-runtime.cpp"
test -f "$model_source/qwen38-aff3-runtime.h"
test -d "$model_source/qwen38-aff3-kernels"

cp "$runtime/llama_cpp/qwen38-aff3-runtime-smoke.cpp" "$stage/"
cp "$model_source/qwen38-aff3-runtime.cpp" "$stage/"
cp "$model_source/qwen38-aff3-runtime.h" "$stage/"
cp -R "$model_source/qwen38-aff3-kernels" "$stage/"

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
  -Wno-subobject-linkage \
  -I "$llama_source/ggml/include" \
  -I "$stage" \
  "$stage/qwen38-aff3-runtime-smoke.cpp" \
  -L "$llama_build/bin" \
  -Wl,-rpath,"$llama_build/bin" \
  -lggml-cpu \
  -lggml-base \
  -lggml \
  -pthread \
  -o "$output"
