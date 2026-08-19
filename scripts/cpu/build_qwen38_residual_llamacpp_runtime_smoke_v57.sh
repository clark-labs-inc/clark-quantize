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

test -f "$runtime/llama_cpp/qwen38-residual-runtime-smoke.cpp"
test -f "$model_source/qwen38-residual-runtime.cpp"
test -f "$model_source/qwen38-residual-runtime.h"
test -d "$model_source/qwen38-residual-kernels"

cp "$runtime/llama_cpp/qwen38-residual-runtime-smoke.cpp" "$stage/"
cp "$model_source/qwen38-residual-runtime.cpp" "$stage/"
cp "$model_source/qwen38-residual-runtime.h" "$stage/"
cp -R "$model_source/qwen38-residual-kernels" "$stage/"

mkdir -p "$(dirname "$output")"
cc \
  -std=c11 -O3 -DNDEBUG -mavx2 -mfma -mf16c \
  -Wall -Wextra -Werror -fPIC \
  -c "$stage/qwen38-residual-kernels/qwen38_calibration_residual_cpu.c" \
  -o "$stage/qwen38-residual-native.o"

g++ \
  -std=c++17 -O3 -DNDEBUG -mavx2 -mfma -mf16c -mssse3 \
  -Wall -Wextra -Werror -Wno-subobject-linkage \
  -I "$llama_source/ggml/include" \
  -I "$llama_source/ggml/src" \
  -I "$stage" \
  "$stage/qwen38-residual-runtime-smoke.cpp" \
  "$stage/qwen38-residual-native.o" \
  -L "$llama_build/bin" \
  -Wl,-rpath,"$llama_build/bin" \
  -lggml-cpu -lggml-base -lggml \
  -lm -pthread \
  -o "$output"
