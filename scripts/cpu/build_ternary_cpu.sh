#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${REPO}/clark_quantize/ternary_cpu.c"
case "$(uname -s)" in
  Darwin) OUTPUT="${REPO}/clark_quantize/_ternary_cpu.dylib"; FLAGS=(-dynamiclib) ;;
  Linux) OUTPUT="${REPO}/clark_quantize/_ternary_cpu.so"; FLAGS=(-shared -fPIC) ;;
  *) echo "unsupported operating system" >&2; exit 2 ;;
esac

CC_BIN="${CC:-cc}"
case "$(uname -m)" in
  x86_64|amd64)
    "${CC_BIN}" -std=c11 -O3 -Wall -Wextra -Werror -mtune=native "${FLAGS[@]}" "${SOURCE}" -lm -o "${OUTPUT}"
    ;;
  *)
    "${CC_BIN}" -std=c11 -O3 -Wall -Wextra -Werror "${FLAGS[@]}" "${SOURCE}" -lm -o "${OUTPUT}"
    ;;
esac
printf '%s\n' "${OUTPUT}"
