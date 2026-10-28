#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/clark-quantize
PYTHON=/workspace/venvs/quant/bin/python
MODEL=/workspace/models/Qwen3.8-27B
CACHE=/workspace/quant-runs/qwen38-global-pilot-v1/hidden_cache/teacher_cache.pt
PATCH=/workspace/quant-runs/qwen38-global-pilot-v1/checkpoints/qwen38-hard-ternary-step224.safetensors
MATRIX=${1:-family}
if [[ "$MATRIX" == "target" ]]; then
  OUT=/workspace/quant-state/qwen38-global-pilot-v1/forensics/target-matrix-v1
elif [[ "$MATRIX" == "projection" ]]; then
  OUT=/workspace/quant-state/qwen38-global-pilot-v1/forensics/projection-matrix-v1
elif [[ "$MATRIX" == "composition" ]]; then
  OUT=/workspace/quant-state/qwen38-global-pilot-v1/forensics/composition-matrix-v1
elif [[ "$MATRIX" == "family" ]]; then
  OUT=/workspace/quant-state/qwen38-global-pilot-v1/forensics/matrix-v1
else
  echo "matrix must be family, target, projection, or composition" >&2
  exit 5
fi

restore_power() {
  /usr/bin/nvidia-smi -pl 300 >/dev/null 2>&1 || true
}
trap restore_power EXIT

if [[ ! -f "$CACHE" || ! -f "$PATCH" ]]; then
  echo "required retained cache or patch is missing" >&2
  exit 2
fi

free_kib=$(df --output=avail /workspace | tail -n 1 | tr -d ' ')
if (( free_kib < 314572800 )); then
  echo "less than 300 GiB free on /workspace" >&2
  exit 3
fi

compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$compute_pids" ]]; then
  echo "GPU is already occupied by compute PIDs: $compute_pids" >&2
  exit 4
fi

/usr/bin/nvidia-smi -pl 250 >/dev/null
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH=experiments/qwen_agentworld:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" \
  experiments/llm_ternary_158bit_dossier/qwen38_27b_global_pilot/run_forensic_ablation_matrix.py \
  --model-dir "$MODEL" \
  --cache "$CACHE" \
  --patch "$PATCH" \
  --out-dir "$OUT" \
  --max-rows-per-task 2 \
  --max-tokens 256 \
  --max-new-tokens 48 \
  --matrix "$MATRIX"
