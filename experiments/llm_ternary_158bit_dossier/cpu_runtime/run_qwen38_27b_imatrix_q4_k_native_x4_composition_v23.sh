#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-row-scale-v3}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
control_build=${CONTROL_BUILD:-/workspace/build/llama.cpp-q4k-fused-ragged-avx2-v2}
candidate_build=${CANDIDATE_BUILD:-/workspace/build/llama.cpp-q4k-native-x4-deinterleave-v21}
quantizer=${QUANTIZER:-/workspace/build/llama.cpp-qwen38-selective-v1/bin/llama-quantize}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp-qwen38-selective-v1}
gguf_python=${GGUF_PYTHON:-$llama_source/gguf-py}
stock_model=${STOCK_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
candidate_model=${CANDIDATE_MODEL:-/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-imatrix-Q4_K-v10.gguf}
draft=${DRAFT:-/workspace/models/controls/mtp-Qwen3.8-27B-Q4_0.gguf}
imatrix=${IMATRIX:-/workspace/quant-state/qwen38-27b-imatrix-q4-k-rowmajor-v10/imatrix.gguf}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-imatrix-q4-k-native-x4-v23}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-imatrix-q4-k-native-x4-v23}
wiki_corpus=${WIKI_CORPUS:-/workspace/quant-runs/cq3-4b-ptq-scope-20260817/wikitext-2-raw/wiki.test.raw}
quality_port=${QUALITY_PORT:-8103}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_imatrix_q4_k_native_x4_composition_v23.json"
amendment="$runtime_dir/qwen38_27b_imatrix_q4_k_native_x4_composition_v23_execution_amendment.json"
summarizer="$runtime_dir/summarize_qwen38_27b_imatrix_q4_k_native_x4_composition_v23.py"
verifier="$runtime_dir/verify_qwen38_27b_selective_gguf_v1.py"
mtp_harness="$runtime_dir/run_qwen38_27b_mtp_parity_v1.py"
v10_receipt="$runtime_dir/receipts/qwen38_27b_imatrix_q4_k_rowmajor_v10/terminal-decision.json"
v21_graph_receipt="$runtime_dir/receipts/qwen38_27b_q4_k_native_x4_deinterleave_fused_avx2_v21/graph-decision.json"
v21_system_receipt="$runtime_dir/receipts/qwen38_27b_q4_k_native_x4_deinterleave_fused_avx2_v21/system-decision.json"
v22_quality_receipt="$runtime_dir/receipts/qwen38_27b_q4_k_native_x4_quality_adjudication_v22/decision.json"
baseline="$runtime_dir/receipts/qwen38_27b_cpu_q4_baseline_v1/summary.json"
cute_manifest="$runtime_dir/cute_quick_v1.json"
math_code_manifest="$runtime_dir/math_code_canary_v1.json"
mmlu_manifest="$runtime_dir/qwen35_4b_cpu_layer30_mmlu_v1.json"
quality_label=qwen38-27b-imatrix-q4-k-native-x4-v23
quality_unit="llama-cpu-eval-${quality_label}.service"
expected_stock_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
expected_candidate_sha=6176472b7d039a6d911b34eea78dd4507256b4a22b3003d40b8587b711014f2a
expected_draft_sha=051a1764cff8c4f3ee6ae8b00593a0364c7539c67fa50ffc58f3f96509fca38e
expected_imatrix_sha=ce6d09c077b22189846861735fd65bb9738a27e947576e23baf887d6c1da4209
expected_candidate_bytes=15606631072

require_sha256() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

cleanup_quality_server() {
  systemctl stop "$quality_unit" >/dev/null 2>&1 || true
}

start_quality_server() {
  cleanup_quality_server
  systemctl reset-failed "$quality_unit" >/dev/null 2>&1 || true
  systemd-run --unit="${quality_unit%.service}" --collect \
    --property=MemoryMax=28G --property=CPUQuota=800% --property=Restart=no \
    /usr/bin/taskset -c 0,2,4,6 \
    "$candidate_build/bin/llama-server" \
    -m "$candidate_model" -t 4 -tb 4 -c 8192 -np 1 -b 512 -ub 512 \
    --host 127.0.0.1 --port "$quality_port" --no-warmup \
    --log-file "$run_dir/server-${quality_label}.log"
  for _ in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:${quality_port}/health" >/dev/null; then
      echo '{"event":"quality_server_ready"}'
      return 0
    fi
    if ! systemctl is-active --quiet "$quality_unit"; then
      systemctl status "$quality_unit" --no-pager >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "$quality_unit did not become healthy within 180 seconds" >&2
  return 1
}

run_mtp() {
  local label=$1 server=$2 target=$3 output=$4 widths=$5 port=$6 status=0
  if [[ ! -s "$output/summary.json" ]]; then
    "$python_bin" "$mtp_harness" \
      --server "$server" --target "$target" --draft "$draft" \
      --output-dir "$output" --widths "$widths" --n-predict 96 \
      --threads 4 --cpu-list 0,2,4,6 --context 512 --batch 128 --ubatch 64 \
      --port-base "$port" > "$run_dir/${label}-harness.log" 2>&1 || status=$?
    if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then
      return "$status"
    fi
  fi
  "$python_bin" - "$output/summary.json" "$widths" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
expected = {"target" if width == 0 else f"mtp{width}" for width in map(int, sys.argv[2].split(","))}
assert payload.get("terminal") is True
assert payload["generation"]["prompt_count"] == 4
assert payload["generation"]["tokens_per_prompt"] == 96
assert {row["configuration"] for row in payload["configurations"]} == expected
PY
}

run_graph_process() {
  local label=$1 arm=$2 output=$3 binary model
  case "$arm" in
    A) binary="$control_build/bin/llama-bench"; model=$stock_model ;;
    B) binary="$candidate_build/bin/llama-bench"; model=$stock_model ;;
    C) binary="$candidate_build/bin/llama-bench"; model=$candidate_model ;;
    *) echo "unknown graph arm: $arm" >&2; return 2 ;;
  esac
  if [[ ! -s "$output" ]]; then
    OMP_NUM_THREADS=4 taskset -c 0,2,4,6 "$binary" \
      -m "$model" -p 4 -n 0 -r 7 -b 128 -ub 64 -t 4 -o json \
      > "$output.tmp" 2> "$state_dir/${label}.stderr"
    mv "$output.tmp" "$output"
  fi
}

finish() {
  local rc=$1
  cleanup_quality_server
  find "$state_dir" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$state_dir/artifacts.sha256"
  printf 'complete\n' > "$state_dir/EXPERIMENT_COMPLETE"
  echo "{\"event\":\"v23_terminal\",\"exit_code\":${rc}}"
  exit "$rc"
}

trap cleanup_quality_server EXIT
mkdir -p "$state_dir" "$run_dir"
cd "$repo"

if [[ -s "$state_dir/EXPERIMENT_COMPLETE" && -s "$state_dir/terminal-decision.json" ]]; then
  (cd / && sha256sum -c "$state_dir/artifacts.sha256")
  jq '{event:"terminal_resumed",passed,decision}' "$state_dir/terminal-decision.json"
  jq -e '.passed == true' "$state_dir/terminal-decision.json" >/dev/null
  exit $?
fi
if [[ -n "$(git status --short)" ]]; then
  echo "v23 requires a clean research checkout" >&2
  exit 2
fi
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null \
  || pgrep -f '/bin/llama-perplexity( |$)' >/dev/null \
  || pgrep -x llama-quantize >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi
if systemctl list-units --type=service --state=running --no-legend | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a Qwen 4B-era service is active; refusing to share the node" >&2
  exit 2
fi

for binary in \
  "$control_build/bin/llama-bench" "$control_build/bin/llama-server" \
  "$candidate_build/bin/llama-bench" "$candidate_build/bin/llama-server" \
  "$candidate_build/bin/llama-perplexity" "$quantizer"; do
  test -x "$binary"
done
test "$(git -C "$llama_source" rev-parse HEAD)" = 4df29be4f4c3673f428170fda944a5b19f743bb8
test "$(stat -c %s "$candidate_model")" = "$expected_candidate_bytes"
require_sha256 "$prereg" 2957f4315c757b6d16036dce33eabab7a585809d83da9bf2fb14114fb55920a0
require_sha256 "$stock_model" "$expected_stock_sha"
require_sha256 "$candidate_model" "$expected_candidate_sha"
require_sha256 "$draft" "$expected_draft_sha"
require_sha256 "$imatrix" "$expected_imatrix_sha"
require_sha256 "$quantizer" 30c8e7234e1e3486077f238901df5f83accd49248e928b80586f93e84bb87b7d
require_sha256 "$gguf_python/gguf/__init__.py" 3ccfc0104cd7ea88c6028743b7bf3f2c89b5f474425de03a217a6072320d7c2f
require_sha256 "$gguf_python/gguf/gguf_reader.py" f1784b59de0f6ef5454091acd0997214a0c3151c0f33ee1f6394eca8441a0002
require_sha256 "$v22_quality_receipt" be8b4e77f5c516bb24c695f90207511e754a724784caa12d9005b79818bb1474
require_sha256 "$baseline" 1557f1aac13f6c557e89006427aa829e2707f07ad73aa29611763e754c409a64
require_sha256 "$cute_manifest" 19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
require_sha256 "$math_code_manifest" 1ec0b41ce3ca3597d65013ac0c8a48c208c4e03b05f0109be61319252c09dab0
require_sha256 "$mmlu_manifest" e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba
require_sha256 "$wiki_corpus" 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08

sha256sum "$prereg" "$amendment" "$summarizer" "$verifier" "$mtp_harness" \
  "$v10_receipt" "$v21_graph_receipt" "$v21_system_receipt" "$v22_quality_receipt" \
  "$baseline" "$cute_manifest" "$math_code_manifest" "$mmlu_manifest" "$wiki_corpus" \
  "$runtime_dir/eval_llama_cpp_cute_quick.py" \
  "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
  "$runtime_dir/eval_llama_cpp_mmlu.py" "$runtime_dir/eval_llama_cpp_wikitext_quick.py" \
  "$control_build/bin/llama-bench" "$control_build/bin/llama-server" \
  "$candidate_build/bin/llama-bench" "$candidate_build/bin/llama-server" \
  "$candidate_build/bin/llama-perplexity" "$quantizer" "$stock_model" \
  "$gguf_python/gguf/__init__.py" "$gguf_python/gguf/gguf_reader.py" \
  "$candidate_model" "$draft" "$imatrix" > "$state_dir/inputs.sha256"
git rev-parse HEAD > "$state_dir/source.commit"

if [[ ! -s "$state_dir/artifact-verification.json" ]]; then
  PYTHONPATH="$gguf_python:$repo" "$python_bin" "$verifier" \
    --control "$stock_model" --candidate "$candidate_model" \
    --selection-profile all_q8_family --expected-target-type Q4_K \
    --expected-bytes-saved 3367239360 \
    --expected-selected-payload-bytes-saved 3367239680 \
    --out "$state_dir/artifact-verification.json"
fi
run_mtp free-generation "$candidate_build/bin/llama-server" "$candidate_model" \
  "$run_dir/free-generation" 0 19500

artifact_status=0
"$python_bin" "$summarizer" --stage artifact --preregistration "$prereg" \
  --amendment "$amendment" --artifact-verification "$state_dir/artifact-verification.json" \
  --v10-receipt "$v10_receipt" --v21-graph-receipt "$v21_graph_receipt" \
  --v21-system-receipt "$v21_system_receipt" --v22-quality-receipt "$v22_quality_receipt" \
  --free-generation "$run_dir/free-generation" --candidate-model "$candidate_model" \
  --stock-model "$stock_model" --imatrix "$imatrix" \
  --candidate-server "$candidate_build/bin/llama-server" --quantizer "$quantizer" \
  --out "$state_dir/artifact-decision.json" || artifact_status=$?
if (( artifact_status != 0 )); then
  cp "$state_dir/artifact-decision.json" "$state_dir/terminal-decision.json"
  finish "$artifact_status"
fi

for arm in A B C; do
  run_graph_process "graph-warmup-${arm}" "$arm" "$state_dir/graph-warmup-${arm}.json"
done

: > "$state_dir/frequency.log"
(
  while true; do
    awk '/cpu MHz/ { sum += $4; count += 1 } END { if (count > 0) printf "%s %.6f\n", systime(), sum / count }' \
      /proc/cpuinfo >> "$state_dir/frequency.log"
    sleep 1
  done
) &
frequency_pid=$!
stop_frequency() {
  kill "$frequency_pid" 2>/dev/null || true
  wait "$frequency_pid" 2>/dev/null || true
}
trap 'stop_frequency; cleanup_quality_server' EXIT

orders=("A B C" "B C A" "C A B" "A C B" "C B A" "B A C")
for index in "${!orders[@]}"; do
  rotation=$((index + 1))
  read -r -a order <<< "${orders[$index]}"
  for arm in "${order[@]}"; do
    run_graph_process "graph-rotation-$(printf '%02d' "$rotation")-${arm}" "$arm" \
      "$state_dir/graph-rotation-$(printf '%02d' "$rotation")-${arm}.json"
  done
done
stop_frequency
trap cleanup_quality_server EXIT

graph_status=0
"$python_bin" "$summarizer" --stage graph --preregistration "$prereg" \
  --state "$state_dir" --artifact-decision "$state_dir/artifact-decision.json" \
  --frequency-log "$state_dir/frequency.log" --out "$state_dir/graph-decision.json" \
  || graph_status=$?

system_args=()
if (( graph_status == 0 )); then
  run_mtp control "$control_build/bin/llama-server" "$stock_model" \
    "$run_dir/control" 0,3 19600
  run_mtp candidate "$candidate_build/bin/llama-server" "$candidate_model" \
    "$run_dir/candidate" 0,3 19700
  system_status=0
  "$python_bin" "$summarizer" --stage system --preregistration "$prereg" \
    --graph-decision "$state_dir/graph-decision.json" \
    --control-mtp "$run_dir/control" --candidate-mtp "$run_dir/candidate" \
    --control-server "$control_build/bin/llama-server" \
    --candidate-server "$candidate_build/bin/llama-server" \
    --out "$state_dir/system-decision.json" || system_status=$?
  system_args=(--system-decision "$state_dir/system-decision.json")
else
  echo '{"event":"serving_skipped","reason":"graph_gate_failed"}'
fi

echo '{"event":"always_run_quick_quality_start","denominators":{"cute":20,"math500":8,"code_trace":6,"wikitext_chunks":4}}'
start_quality_server
if [[ ! -s "$state_dir/cute.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_cute_quick.py" \
    --manifest "$cute_manifest" --server "http://127.0.0.1:${quality_port}" \
    --model-label "$quality_label" --model-sha256 "$expected_candidate_sha" \
    --out "$state_dir/cute.json" > "$run_dir/cute.log" 2>&1
fi
if [[ ! -s "$state_dir/math-code.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
    --manifest "$math_code_manifest" --server "http://127.0.0.1:${quality_port}" \
    --model-label "$quality_label" --model-sha256 "$expected_candidate_sha" \
    --out "$state_dir/math-code.json" > "$run_dir/math-code.log" 2>&1
fi
cleanup_quality_server
if [[ ! -s "$state_dir/wikitext.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_wikitext_quick.py" \
    --model "$candidate_model" --model-label "$quality_label" --corpus "$wiki_corpus" \
    --llama-perplexity "$candidate_build/bin/llama-perplexity" \
    --chunks 4 --context 512 --threads 4 --out "$state_dir/wikitext.json" \
    > "$run_dir/wikitext.log" 2>&1
fi
quick_status=0
"$python_bin" "$summarizer" --stage quick-quality --preregistration "$prereg" \
  --baseline "$baseline" --cute "$state_dir/cute.json" \
  --math-code "$state_dir/math-code.json" --wikitext "$state_dir/wikitext.json" \
  --out "$state_dir/quick-quality-decision.json" || quick_status=$?

final_status=0
"$python_bin" "$summarizer" --stage final --preregistration "$prereg" \
  --artifact-decision "$state_dir/artifact-decision.json" \
  --graph-decision "$state_dir/graph-decision.json" \
  --quick-quality-decision "$state_dir/quick-quality-decision.json" \
  --baseline "$baseline" "${system_args[@]}" \
  --out "$state_dir/terminal-decision.json" || final_status=$?
if jq -e '.terminal == true' "$state_dir/terminal-decision.json" >/dev/null; then
  finish "$final_status"
fi

echo '{"event":"conditional_mmlu_start","n":64}'
start_quality_server
if [[ ! -s "$state_dir/mmlu.json" ]]; then
  resume_args=()
  if [[ -s "$state_dir/mmlu.json.partial" ]]; then
    resume_args+=(--resume)
  fi
  PYTHONPATH="$repo" "$python_bin" "$runtime_dir/eval_llama_cpp_mmlu.py" \
    --manifest "$mmlu_manifest" --model "$candidate_model" --model-label "$quality_label" \
    --model-sha256 "$expected_candidate_sha" --server "http://127.0.0.1:${quality_port}" \
    --workers 1 --timeout 900 --checkpoint-every 4 --out "$state_dir/mmlu.json" \
    "${resume_args[@]}" > "$run_dir/mmlu.log" 2>&1
fi
cleanup_quality_server
final_status=0
"$python_bin" "$summarizer" --stage final --preregistration "$prereg" \
  --artifact-decision "$state_dir/artifact-decision.json" \
  --graph-decision "$state_dir/graph-decision.json" \
  --quick-quality-decision "$state_dir/quick-quality-decision.json" \
  --baseline "$baseline" "${system_args[@]}" --mmlu "$state_dir/mmlu.json" \
  --out "$state_dir/terminal-decision.json" || final_status=$?
finish "$final_status"
