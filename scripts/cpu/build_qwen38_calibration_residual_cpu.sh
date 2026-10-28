#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_file="$repo/clark_quantize/qwen38_calibration_residual_cpu.c"
case "$(uname -s)" in
  Darwin) output="$repo/clark_quantize/_qwen38_calibration_residual_cpu.dylib"; flags=(-dynamiclib) ;;
  Linux) output="$repo/clark_quantize/_qwen38_calibration_residual_cpu.so"; flags=(-shared -fPIC) ;;
  *) echo "unsupported operating system" >&2; exit 2 ;;
esac

cc_bin=${CC:-cc}
"$cc_bin" -std=c11 -O3 -Wall -Wextra -Werror "${flags[@]}" \
  "$source_file" -lm -o "$output"
printf '%s\n' "$output"
