#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MASK_SOURCE="${REPO}/clark_quantize/qwen38_fixed_support_mask64_cpu.c"
FUSED_SOURCE="${REPO}/clark_quantize/qwen38_mask64_hash_fused_cpu.c"
case "$(uname -s)" in
  Darwin) OUTPUT="${REPO}/clark_quantize/_qwen38_mask64_hash_fused_cpu.dylib"; FLAGS=(-dynamiclib) ;;
  Linux) OUTPUT="${REPO}/clark_quantize/_qwen38_mask64_hash_fused_cpu.so"; FLAGS=(-shared -fPIC) ;;
  *) echo "unsupported operating system" >&2; exit 2 ;;
esac

CC_BIN="${CC:-cc}"
"${CC_BIN}" -std=c11 -O3 -Wall -Wextra -Werror -mtune=native "${FLAGS[@]}" \
  "${MASK_SOURCE}" "${FUSED_SOURCE}" -o "${OUTPUT}"
printf '%s\n' "${OUTPUT}"
