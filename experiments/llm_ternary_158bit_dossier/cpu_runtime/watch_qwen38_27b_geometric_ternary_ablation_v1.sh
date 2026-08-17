#!/usr/bin/env bash
set -euo pipefail

readonly expected_commit="${CLARK_EXPECTED_COMMIT:?CLARK_EXPECTED_COMMIT is required}"
readonly bundle="/workspace/tmp/clark-quantize-qwen38-geometric.bundle"
readonly checkout="/workspace/tmp/clark-quantize-qwen38-geometric"
readonly receipt="/workspace/quant-state/qwen38-27b-geometric-ternary-ablation-v1.json"
readonly log="/workspace/quant-runs/qwen38-27b-geometric-ternary-ablation-v1.stdout"

if [[ ! -f "${bundle}" ]]; then
  echo "infrastructure-invalid: frozen geometric source bundle is absent" >&2
  exit 70
fi
if pgrep -f '/workspace/build/.*/llama-(server|cli)' >/dev/null; then
  echo "infrastructure-invalid: refusing to compete with a Qwen3.8 llama process" >&2
  exit 71
fi
if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((8 * 1024 * 1024)) ]]; then
  echo "infrastructure-invalid: less than 8 GiB memory is available" >&2
  exit 72
fi
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "infrastructure-invalid: swap is active" >&2
  exit 73
fi

if [[ -e "${checkout}" ]]; then
  echo "infrastructure-invalid: geometric checkout path already exists" >&2
  exit 75
fi
git clone --quiet --no-checkout "${bundle}" "${checkout}"
git -C "${checkout}" checkout --quiet --detach "${expected_commit}"
if [[ $(git -C "${checkout}" rev-parse HEAD) != "${expected_commit}" ]]; then
  echo "infrastructure-invalid: geometric checkout commit changed" >&2
  exit 74
fi

cd "${checkout}"
/workspace/venvs/quant-cpu/bin/python -m pytest -q \
  clark_quantize/test_qwen38_affine_ternary.py \
  clark_quantize/test_qwen38_geometric_transform.py
mkdir -p "$(dirname "${receipt}")" "$(dirname "${log}")"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
taskset -c 0-3 /workspace/venvs/quant-cpu/bin/python \
  experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_geometric_ternary_ablation_v1.py \
  --threads 4 \
  --out "${receipt}" | tee "${log}"
