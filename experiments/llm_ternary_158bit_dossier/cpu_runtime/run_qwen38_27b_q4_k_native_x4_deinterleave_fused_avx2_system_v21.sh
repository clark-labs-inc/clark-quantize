#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-row-scale-v3}
control_build=${CONTROL_BUILD:-/workspace/build/llama.cpp-q4k-fused-ragged-avx2-v2}
candidate_build=${CANDIDATE_BUILD:-/workspace/build/llama.cpp-q4k-native-x4-deinterleave-v21}
target=${TARGET:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
draft=${DRAFT:-/workspace/models/controls/mtp-Qwen3.8-27B-Q4_0.gguf}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q4-k-native-x4-deinterleave-v21}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-q4-k-native-x4-deinterleave-v21}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
summarizer="$runtime_dir/summarize_qwen38_27b_q4_k_native_x4_deinterleave_fused_avx2_v21.py"

require_sha256() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

run_mtp() {
  local label=$1 server=$2 output=$3 port=$4 status=0
  if [[ ! -s "$output/summary.json" ]]; then
    /workspace/venvs/quant-cpu/bin/python \
      "$runtime_dir/run_qwen38_27b_mtp_parity_v1.py" \
      --server "$server" --target "$target" --draft "$draft" \
      --output-dir "$output" --widths 0,3 --n-predict 96 \
      --threads 4 --cpu-list 0,2,4,6 --context 512 --batch 128 --ubatch 64 \
      --port-base "$port" > "$run_dir/$label-harness.log" 2>&1 || status=$?
    if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then
      return "$status"
    fi
  fi
  /workspace/venvs/quant-cpu/bin/python - "$output/summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
assert payload.get("terminal") is True
assert payload["generation"]["prompt_count"] == 4
assert payload["generation"]["tokens_per_prompt"] == 96
assert {row["configuration"] for row in payload["configurations"]} == {"target", "mtp3"}
PY
}

mkdir -p "$state_dir" "$run_dir"

if [[ -s "$state_dir/EXPERIMENT_COMPLETE" && -s "$state_dir/terminal-decision.json" ]]; then
  jq '{event:"terminal_resumed",passed:.passed}' "$state_dir/terminal-decision.json"
  jq -e '.passed == true' "$state_dir/terminal-decision.json" >/dev/null
  exit $?
fi
if [[ -s "$state_dir/SYSTEM_COMPLETE" && -s "$state_dir/system-decision.json" ]]; then
  jq '{event:"system_resumed",passed:.passed}' "$state_dir/system-decision.json"
  jq -e '.passed == true' "$state_dir/system-decision.json" >/dev/null
  exit $?
fi
if pgrep -f '/llama-(server|bench|perplexity)( |$)' >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi

jq -e '.passed == true and .terminal == false and .denominator.samples_per_runtime == 42' \
  "$state_dir/graph-decision.json" >/dev/null
require_sha256 "$target" 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
require_sha256 "$draft" 051a1764cff8c4f3ee6ae8b00593a0364c7539c67fa50ffc58f3f96509fca38e

for binary in "$control_build/bin/llama-server" "$candidate_build/bin/llama-server"; do
  test -x "$binary"
done

run_mtp control "$control_build/bin/llama-server" "$run_dir/control" 19400
run_mtp candidate "$candidate_build/bin/llama-server" "$run_dir/candidate" 19500

status=0
/workspace/venvs/quant-cpu/bin/python "$summarizer" \
  --stage system \
  --graph-decision "$state_dir/graph-decision.json" \
  --control-mtp "$run_dir/control" --candidate-mtp "$run_dir/candidate" \
  --control-server "$control_build/bin/llama-server" \
  --candidate-server "$candidate_build/bin/llama-server" \
  --out "$state_dir/system-decision.json" || status=$?

if [[ "$status" -ne 0 ]]; then
  cp "$state_dir/system-decision.json" "$state_dir/terminal-decision.json"
  printf 'complete\n' > "$state_dir/EXPERIMENT_COMPLETE"
  exit "$status"
fi

printf 'complete\n' > "$state_dir/SYSTEM_COMPLETE"
jq '{event:"system_gate_complete",target_only,mtp3,checks}' "$state_dir/system-decision.json"
