#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-residual-v37}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
model_dir=${MODEL_DIR:-/workspace/models/Qwen3.8-27B}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-sensitivity-gated-residual-v37}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-sensitivity-gated-residual-v37}
training_receipt=${TRAINING_RECEIPT:-/workspace/quant-state/qwen38-calibration-ranked-mixed-training-v1.json}
v36_receipt=${V36_RECEIPT:-/workspace/quant-state/qwen38-calibration-span-residual-v36/calibration.json}
training_cache=${TRAINING_CACHE:-/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/block-cache.pt}
parent_artifact_root=${PARENT_ARTIFACT_ROOT:-/workspace/quant-runs/qwen38-affine-butterfly-blocks-v1/artifact}
trained_artifact_root=${TRAINED_ARTIFACT_ROOT:-/workspace/quant-runs/qwen38-calibration-ranked-mixed-v1/trained}
expected_commit=${EXPECTED_REPOSITORY_COMMIT:?EXPECTED_REPOSITORY_COMMIT is required}

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OMP_PROC_BIND=true

prereg="$runtime_dir/qwen38_27b_sensitivity_gated_residual_v37.json"
runner="$runtime_dir/run_qwen38_27b_sensitivity_gated_residual_v37.py"
collector="$runtime_dir/collect_qwen38_27b_mixed_confirmation_cache_v1.py"
artifact_root="$run_dir/artifacts"
calibration_receipt="$state_dir/calibration.json"
confirmation_cache="$run_dir/fresh-confirmation-cache.pt"
confirmation_cache_receipt="$state_dir/fresh-confirmation-cache.json"
confirmation_receipt="$state_dir/confirmation.json"

finish() {
  local status=$1
  find "$state_dir" "$run_dir" -type f \
    ! -name EXPERIMENT_COMPLETE \
    ! -name artifacts.sha256 ! -name artifacts.sha256.tmp -print0 \
    | sort -z | xargs -0 sha256sum > "$state_dir/artifacts.sha256.tmp"
  mv "$state_dir/artifacts.sha256.tmp" "$state_dir/artifacts.sha256"
  printf 'complete\n' > "$state_dir/EXPERIMENT_COMPLETE.tmp"
  mv "$state_dir/EXPERIMENT_COMPLETE.tmp" "$state_dir/EXPERIMENT_COMPLETE"
  echo "{\"event\":\"v37_terminal\",\"exit_code\":${status}}"
  exit "$status"
}

mkdir -p "$state_dir" "$run_dir" "$artifact_root"
if [[ -s "$state_dir/EXPERIMENT_COMPLETE" ]]; then
  if [[ -s "$confirmation_receipt" ]]; then
    jq '{event:"v37_terminal_resumed",stage:"confirmation",gate:.gate}' \
      "$confirmation_receipt"
    jq -e '.complete == true and .terminal == true and .gate.passed == true' \
      "$confirmation_receipt" >/dev/null
    exit $?
  fi
  if [[ -s "$calibration_receipt" ]]; then
    jq '{event:"v37_terminal_resumed",stage:"calibration",gate:.gate}' \
      "$calibration_receipt"
    exit 2
  fi
  echo "v37 completion marker has no terminal receipt" >&2
  exit 1
fi

actual_commit=$(git -C "$repo" rev-parse HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "v37 repository commit changed: $actual_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "v37 repository checkout is dirty" >&2
  exit 2
fi
if pgrep -f '/bin/llama-(server|bench|perplexity)( |$)' >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi
if pgrep -f '^/workspace/venvs/quant-cpu/bin/python .*[q]wen' >/dev/null; then
  echo "another Qwen Python workload is active" >&2
  exit 2
fi
if [[ -e /workspace/quant-runs/qwen38-calibration-span-residual-v36/fresh-confirmation-cache.pt \
   || -e /workspace/quant-state/qwen38-calibration-span-residual-v36/fresh-confirmation-cache.json ]]; then
  echo "v36 holdout was unexpectedly materialized" >&2
  exit 2
fi
for required in \
  "$prereg" "$runner" "$collector" "$training_receipt" "$v36_receipt" \
  "$training_cache" "$model_dir/model.safetensors.index.json" \
  "$parent_artifact_root" "$trained_artifact_root"; do
  test -e "$required"
done

sha256sum "$prereg" "$runner" "$collector" "$training_receipt" \
  "$v36_receipt" "$training_cache" "$model_dir/model.safetensors.index.json" \
  > "$state_dir/sealed-inputs.sha256"
printf '%s\n' "$actual_commit" > "$state_dir/repository-commit.txt"

common=(
  --model-dir "$model_dir"
  --preregistration "$prereg"
  --training-receipt "$training_receipt"
  --v36-receipt "$v36_receipt"
  --training-cache "$training_cache"
  --parent-artifact-root "$parent_artifact_root"
  --trained-artifact-root "$trained_artifact_root"
  --artifact-root "$artifact_root"
  --threads 4
)

calibration_status=0
taskset -c 0,2,4,6 "$python_bin" "$runner" \
  --stage calibration "${common[@]}" --out "$calibration_receipt" \
  > "$run_dir/calibration.log" 2>&1 || calibration_status=$?
if [[ "$calibration_status" -ne 0 ]]; then
  if [[ "$calibration_status" -eq 2 && -s "$calibration_receipt" ]]; then
    finish 2
  fi
  exit "$calibration_status"
fi

taskset -c 0,2,4,6 "$python_bin" "$collector" \
  --model-dir "$model_dir" --preregistration "$prereg" \
  --training-cache "$training_cache" --cache "$confirmation_cache" \
  --receipt "$confirmation_cache_receipt" --threads 4 \
  > "$run_dir/fresh-confirmation-cache.log" 2>&1

confirmation_status=0
taskset -c 0,2,4,6 "$python_bin" "$runner" \
  --stage confirmation "${common[@]}" \
  --calibration-receipt "$calibration_receipt" \
  --confirmation-cache "$confirmation_cache" \
  --confirmation-cache-receipt "$confirmation_cache_receipt" \
  --out "$confirmation_receipt" \
  > "$run_dir/confirmation.log" 2>&1 || confirmation_status=$?
if [[ "$confirmation_status" -eq 0 || "$confirmation_status" -eq 2 ]]; then
  finish "$confirmation_status"
fi
exit "$confirmation_status"
