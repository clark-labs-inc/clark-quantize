#!/usr/bin/env bash
set -euo pipefail

readonly expected_commit="${CLARK_EXPECTED_COMMIT:?CLARK_EXPECTED_COMMIT is required}"
readonly bundle="/workspace/tmp/clark-quantize-qwen38-entropy.bundle"
readonly checkout="/workspace/tmp/clark-quantize-qwen38-entropy"
readonly native_decoder="/workspace/clark-quantize/clark_quantize/_base3_native.so"
readonly native_decoder_sha256="153c3c375d81c15320998e73cd9e864710b863af42021de1a090580aa14eec02"
readonly receipt="/workspace/quant-state/qwen38-27b-ternary-type-class-rate-v1.json"
readonly log="/workspace/quant-runs/qwen38-27b-ternary-type-class-rate-v1.stdout"

while systemctl is-active --quiet quant-qwen38-27b-geometric-ternary-ablation-v2.service; do
  sleep 15
done
if [[ ! -f "${bundle}" ]]; then
  echo "infrastructure-invalid: frozen entropy source bundle is absent" >&2
  exit 70
fi
if [[ ! -f "${native_decoder}" ]] || \
   [[ $(sha256sum "${native_decoder}" | awk '{print $1}') != "${native_decoder_sha256}" ]]; then
  echo "infrastructure-invalid: authenticated native source decoder is absent" >&2
  exit 71
fi
if pgrep -f '/workspace/build/.*/llama-(server|cli)' >/dev/null; then
  echo "infrastructure-invalid: refusing to compete with a Qwen3.8 llama process" >&2
  exit 72
fi
if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((8 * 1024 * 1024)) ]]; then
  echo "infrastructure-invalid: less than 8 GiB memory is available" >&2
  exit 73
fi
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "infrastructure-invalid: swap is active" >&2
  exit 74
fi
if [[ -e "${checkout}" ]]; then
  echo "infrastructure-invalid: entropy checkout path already exists" >&2
  exit 75
fi

git clone --quiet --no-checkout "${bundle}" "${checkout}"
git -C "${checkout}" checkout --quiet --detach "${expected_commit}"
if [[ $(git -C "${checkout}" rev-parse HEAD) != "${expected_commit}" ]]; then
  echo "infrastructure-invalid: entropy checkout commit changed" >&2
  exit 76
fi

cd "${checkout}"
/workspace/venvs/quant-cpu/bin/python -m pytest -q \
  clark_quantize/test_qwen38_ternary_entropy.py
mkdir -p "$(dirname "${receipt}")" "$(dirname "${log}")"
export CLARK_BASE3_NATIVE_LIB="${native_decoder}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
taskset -c 0-3 /workspace/venvs/quant-cpu/bin/python \
  experiments/llm_ternary_158bit_dossier/cpu_runtime/analyze_qwen38_27b_ternary_type_class_rate_v1.py \
  --workers 4 \
  --batch-chunks 32 \
  --out "${receipt}" | tee "${log}"
