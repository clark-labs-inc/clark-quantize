#!/usr/bin/env bash
set -euo pipefail

REPO="${QUANT_REPO:-/workspace/clark-quantize}"
SOURCE="${REPO}/clark_quantize/base3_native.c"
case "$(uname -s)" in
  Darwin) OUTPUT="${REPO}/clark_quantize/_base3_native.dylib" ; FLAGS=(-dynamiclib) ;;
  Linux) OUTPUT="${REPO}/clark_quantize/_base3_native.so" ; FLAGS=(-shared -fPIC) ;;
  *) echo "unsupported native base-3 platform: $(uname -s)" >&2; exit 2 ;;
esac

test -f "${SOURCE}"
temporary="${OUTPUT}.tmp.$$"
trap 'rm -f "${temporary}"' EXIT
"${CC:-cc}" -O3 -std=c11 "${FLAGS[@]}" "${SOURCE}" -o "${temporary}"
chmod 0755 "${temporary}"
mv -f "${temporary}" "${OUTPUT}"
trap - EXIT
echo "${OUTPUT}"
