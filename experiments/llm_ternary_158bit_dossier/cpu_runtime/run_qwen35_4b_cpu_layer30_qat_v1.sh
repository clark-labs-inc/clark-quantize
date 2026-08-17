#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
model_dir=${MODEL_DIR:-/workspace/models/Qwen3.5-4B}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v1}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
agent_dir="$repo/experiments/qwen_agentworld"
source_jsonl="$runtime_dir/qwen35_4b_cpu_layer30_train_v1.jsonl"
prereg="$runtime_dir/qwen35_4b_cpu_layer30_qat_preregistration_v1.json"
inventory="$run_dir/model-inventory.json"
teacher_cache="$run_dir/teacher-completions.pt"
activation_cache="$run_dir/layer30-activations.pt"
local_patch="$run_dir/layer30-local-qat.safetensors"
final_patch="$run_dir/layer30-survival-qat.safetensors"

mkdir -p "$run_dir/checkpoints"
cd "$repo"

sha256sum "$prereg" "$source_jsonl" "$model_dir/model.safetensors.index.json" > "$run_dir/inputs.sha256"
git rev-parse HEAD > "$run_dir/source.commit"
git status --short > "$run_dir/source.status"

if [[ ! -s "$inventory" ]]; then
  /usr/bin/time -v "$python_bin" "$agent_dir/model_inventory.py" \
    --model-dir "$model_dir" \
    --out "$inventory" \
    > "$run_dir/inventory.log" 2>&1
fi

if [[ ! -s "$teacher_cache" ]]; then
  /usr/bin/time -v "$python_bin" "$agent_dir/collect_teacher_rollout_cache.py" \
    --model-dir "$model_dir" \
    --input "$source_jsonl" \
    --out "$teacher_cache" \
    --max-rows 12 \
    --max-model-len 256 \
    --generate-tokens 24 \
    --top-k 128 \
    --hidden-layers 30 \
    --projection-dim 64 \
    --device cpu \
    --torch-threads 4 \
    > "$run_dir/teacher-cache.log" 2>&1
fi

if [[ ! -s "$activation_cache" ]]; then
  /usr/bin/time -v "$python_bin" "$agent_dir/cache_mlp_activations.py" \
    --model-dir "$model_dir" \
    --cache "$teacher_cache" \
    --out "$activation_cache" \
    --layers 30 \
    --positions 24 \
    --device cpu \
    --torch-threads 4 \
    > "$run_dir/activation-cache.log" 2>&1
fi

if [[ ! -s "$local_patch" ]]; then
  resume_args=()
  if [[ -s "$run_dir/checkpoints/layer-30-latest.pt" ]]; then
    resume_args+=(--resume)
  fi
  /usr/bin/time -v "$python_bin" "$agent_dir/train_mlp_triplet_qat.py" \
    --model-dir "$model_dir" \
    --inventory "$inventory" \
    --activation-cache "$activation_cache" \
    --out "$local_patch" \
    --report "$run_dir/layer30-local-qat.json" \
    --layers 30 \
    --mode ternary \
    --group-size 128 \
    --artifact-exact-ternary \
    --steps-per-layer 8 \
    --batch-size 32 \
    --lr 2e-5 \
    --code-weight 0.001 \
    --grad-clip 1 \
    --log-every 1 \
    --seed 20260817 \
    --device cpu \
    --torch-threads 4 \
    --checkpoint-dir "$run_dir/checkpoints" \
    --checkpoint-every 2 \
    "${resume_args[@]}" \
    > "$run_dir/local-qat.log" 2>&1
fi

if [[ ! -s "$final_patch" ]]; then
  /usr/bin/time -v "$python_bin" "$agent_dir/train_block_survival_qat.py" \
    --model-dir "$model_dir" \
    --cache "$teacher_cache" \
    --base-patch "$local_patch" \
    --out-patch "$final_patch" \
    --out-report "$run_dir/layer30-survival-qat.json" \
    --layers 30 \
    --max-train-tokens 256 \
    --max-train-rows 2 \
    --mode ternary \
    --group-size 128 \
    --artifact-exact-ternary \
    --steps-per-layer 2 \
    --lr 5e-6 \
    --hidden-weight 0.05 \
    --anti-weight 0.1 \
    --code-weight 0.002 \
    --no-train-projections \
    --no-train-norms \
    --device cpu \
    --torch-threads 4 \
    > "$run_dir/survival-qat.log" 2>&1
fi

sha256sum "$teacher_cache" "$activation_cache" "$inventory" "$local_patch" "$final_patch" \
  "$run_dir/checkpoints/layer-30-latest.pt" > "$run_dir/artifacts.sha256"
printf 'complete\n' > "$run_dir/TRAINING_COMPLETE"
