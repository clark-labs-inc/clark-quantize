#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${REPO}/clark_quantize/qwen38_hash_residual_cpu.c"
case "$(uname -s)" in
  Darwin) OUTPUT="${REPO}/clark_quantize/_qwen38_hash_residual_cpu.dylib"; FLAGS=(-dynamiclib) ;;
  Linux) OUTPUT="${REPO}/clark_quantize/_qwen38_hash_residual_cpu.so"; FLAGS=(-shared -fPIC) ;;
  *) echo "unsupported operating system" >&2; exit 2 ;;
esac

CC_BIN="${CC:-cc}"
"${CC_BIN}" -std=c11 -O3 -Wall -Wextra -Werror "${FLAGS[@]}" \
  "${SOURCE}" -lm -o "${OUTPUT}"
printf '%s\n' "${OUTPUT}"
