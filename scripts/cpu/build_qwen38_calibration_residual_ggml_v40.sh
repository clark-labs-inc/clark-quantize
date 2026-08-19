#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 LLAMA_CPP_SOURCE LLAMA_CPP_BUILD OUTPUT" >&2
  exit 64
fi

repo=$(cd "$(dirname "$0")/../.." && pwd)
llama_source=$1
llama_build=$2
output=$3
object="${output}.residual.o"

cc \
  -std=c11 -O3 -Wall -Wextra -Werror -fPIC \
  -c "$repo/clark_quantize/qwen38_calibration_residual_cpu.c" \
  -o "$object"

c++ \
  -std=c++17 -O3 -Wall -Wextra -Werror \
  -I"$repo/clark_quantize" \
  -I"$llama_source/ggml/include" \
  -I"$llama_source/ggml/src" \
  "$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime/qwen38_calibration_residual_ggml_v40.cpp" \
  "$object" \
  -L"$llama_build/bin" \
  -Wl,-rpath,"$llama_build/bin" \
  -lggml-cpu -lggml-base -lggml \
  -lm -pthread \
  -o "$output"

rm -f "$object"
