#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
model_dir=${MODEL_DIR:-/workspace/models/Qwen3.5-4B}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen35-4b-cpu-layer30-qat-v3}
hf_out=${HF_OUT:-/workspace/models/Qwen3.5-4B-layer30-CPUQAT-v3}
bf16_out=${BF16_OUT:-/workspace/models/controls/Qwen3.5-4B-layer30-CPUQAT-v3-BF16-full.gguf}
cq3_out=${CQ3_OUT:-/workspace/models/controls/Qwen3.5-4B-layer30-CPUQAT-v3-CQ3.gguf}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
patch="$run_dir/layer30-heldout-selected.safetensors"
materialize_receipt="$run_dir/materialize.json"
scope_receipt="$run_dir/gguf-tensor-scope.json"
baseline=/workspace/models/controls/Qwen3.5-4B-BF16.gguf
direct_ptq=/workspace/models/controls/Qwen3.5-4B-layer30-DirectPTQ-CQ3.gguf

cd "$repo"
if [[ ! -s "$run_dir/TRAINING_COMPLETE" || ! -s "$patch" ]]; then
  echo "held-out-selected training is not complete: $run_dir" >&2
  exit 2
fi
if [[ -e "$run_dir/EXPORT_COMPLETE" ]]; then
  echo "completed marker already exists: $run_dir/EXPORT_COMPLETE" >&2
  exit 2
fi
if [[ -s "$run_dir/source.status" ]]; then
  echo "training source receipt was dirty" >&2
  exit 2
fi
if [[ -n "$(git status --short)" ]]; then
  echo "export requires a clean source checkout" >&2
  exit 2
fi

git rev-parse HEAD > "$run_dir/export-source.commit"
git -C "$llama_source" rev-parse HEAD > "$run_dir/llama-source.commit"
sha256sum "$python_bin" "$llama_build/bin/llama-quantize" \
  "$llama_build/bin/llama-server" > "$run_dir/export-tools.sha256"

if [[ ! -s "$materialize_receipt" ]]; then
  "$python_bin" "$runtime_dir/materialize_hf_weight_patch.py" \
    --model-dir "$model_dir" \
    --patch "$patch" \
    --out-dir "$hf_out" \
    --receipt "$materialize_receipt" \
    > "$run_dir/materialize.log" 2>&1
fi

if [[ ! -s "$bf16_out" ]]; then
  bf16_partial="${bf16_out}.partial"
  if [[ -e "$bf16_partial" ]]; then
    echo "partial BF16 GGUF requires inspection: $bf16_partial" >&2
    exit 2
  fi
  "$python_bin" "$llama_source/convert_hf_to_gguf.py" "$hf_out" \
    --outfile "$bf16_partial" \
    --outtype bf16 \
    > "$run_dir/convert-bf16.log" 2>&1
  mv "$bf16_partial" "$bf16_out"
fi

if [[ ! -s "$cq3_out" ]]; then
  cq3_partial="${cq3_out}.partial"
  if [[ -e "$cq3_partial" ]]; then
    echo "partial CQ3 GGUF requires inspection: $cq3_partial" >&2
    exit 2
  fi
  "$llama_build/bin/llama-quantize" \
    --pure \
    --tensor-type '^blk[.]30[.]ffn_(gate|up|down)[.]weight$=CQ3_0' \
    --tensor-type '.*=BF16' \
    "$bf16_out" "$cq3_partial" CQ3_0 4 \
    > "$run_dir/quantize-cq3.log" 2>&1
  mv "$cq3_partial" "$cq3_out"
fi

PYTHONPATH="$llama_source/gguf-py" "$python_bin" \
  "$runtime_dir/verify_gguf_tensor_scope.py" \
  --baseline "$baseline" \
  --candidate "$cq3_out" \
  --control "$direct_ptq" \
  --target-regex 'blk[.]30[.]ffn_(gate|up|down)[.]weight' \
  --baseline-target-type 30 \
  --packed-target-type 43 \
  --out "$scope_receipt" \
  > "$run_dir/verify-gguf-scope.log" 2>&1

sha256sum "$patch" "$bf16_out" "$cq3_out" "$scope_receipt" \
  > "$run_dir/export-artifacts.sha256"
printf 'complete\n' > "$run_dir/EXPORT_COMPLETE"
