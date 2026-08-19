#!/usr/bin/env bash
set -euo pipefail

runtime_dir=${RUNTIME_DIR:-/workspace/quant-state/qwen38-27b-clang21-serving-transfer-v35/inputs}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-clang21-serving-transfer-v35}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-clang21-serving-transfer-v35}
parent_build=${PARENT_BUILD:-/workspace/build/llama.cpp-q4k-native-x4-deinterleave-v21}
candidate_build=${CANDIDATE_BUILD:-/workspace/build/llama.cpp-q4k-clang21-pair-local-scaled-v30}
target_model=${TARGET_MODEL:-/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-imatrix-Q4_K-v10.gguf}
draft_model=${DRAFT_MODEL:-/workspace/models/controls/mtp-Qwen3.8-27B-Q4_0.gguf}
wiki_corpus=${WIKI_CORPUS:-/workspace/quant-runs/cq3-4b-ptq-scope-20260817/wikitext-2-raw/wiki.test.raw}
prereg="$runtime_dir/qwen38_27b_clang21_serving_transfer_v35.json"
summarizer="$runtime_dir/summarize_qwen38_27b_clang21_serving_transfer_v35.py"
mtp_harness="$runtime_dir/run_qwen38_27b_mtp_parity_v1.py"
baseline="$runtime_dir/baseline.json"
cute_manifest="$runtime_dir/cute_quick_v1.json"
math_code_manifest="$runtime_dir/math_code_canary_v1.json"
mmlu_manifest="$runtime_dir/qwen35_4b_cpu_layer30_mmlu_v1.json"
quality_port=${QUALITY_PORT:-20800}
quality_pid=
frequency_pid=

expected_target_sha=6176472b7d039a6d911b34eea78dd4507256b4a22b3003d40b8587b711014f2a
expected_draft_sha=051a1764cff8c4f3ee6ae8b00593a0364c7539c67fa50ffc58f3f96509fca38e
expected_parent_server_sha=2c9db28068c8cf92f92fa631e1667e5d1cb39c83e21ae3245bb655d69ec358fc
expected_parent_cpu_sha=f6a1b46830bda9e322ff77d8d203eeb835b48cda0986cb547ea6141d08f95029
expected_candidate_server_sha=e0efaa0b8c8412d6d85f1796442bff3ad8d8788fdc0542de7004a97f51d3bf9a
expected_candidate_perplexity_sha=680daa54d7c27ca69abdd31f9234ab97a2791cc0b71620f78909e0f181dfea87
expected_candidate_cpu_sha=02b27cf8dec0c43a62b680d81640a97b4caef45c28f8377524dd2f083b480a81

require_sha256() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

stop_frequency() {
  if [[ -n "${frequency_pid:-}" ]]; then
    kill "$frequency_pid" 2>/dev/null || true
    wait "$frequency_pid" 2>/dev/null || true
    frequency_pid=
  fi
}

stop_quality_server() {
  if [[ -n "${quality_pid:-}" ]]; then
    kill -TERM -- "-$quality_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$quality_pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL -- "-$quality_pid" 2>/dev/null || true
    wait "$quality_pid" 2>/dev/null || true
    quality_pid=
  fi
}

finish() {
  local rc=$1
  stop_frequency
  stop_quality_server
  find "$state_dir" "$run_dir" -type f ! -name artifacts.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$state_dir/artifacts.sha256"
  printf 'complete\n' > "$state_dir/EXPERIMENT_COMPLETE"
  echo "{\"event\":\"v35_terminal\",\"exit_code\":${rc}}"
  exit "$rc"
}

start_frequency() {
  : > "$state_dir/frequency.log"
  (
    while true; do
      awk '/cpu MHz/ { sum += $4; count += 1 } END { if (count > 0) printf "%s %.6f\n", systime(), sum / count }' \
        /proc/cpuinfo >> "$state_dir/frequency.log"
      sleep 1
    done
  ) &
  frequency_pid=$!
}

run_mtp() {
  local label=$1 arm=$2 port=$3 server output status=0
  output="$run_dir/$label"
  if [[ "$arm" == parent ]]; then
    server="$parent_build/bin/llama-server"
  else
    server="$candidate_build/bin/llama-server"
  fi
  if [[ ! -s "$output/summary.json" ]]; then
    if [[ "$arm" == parent ]]; then
      env -u KMP_BLOCKTIME -u KMP_LIBRARY -u OMP_WAIT_POLICY OMP_NUM_THREADS=4 \
        "$python_bin" "$mtp_harness" \
        --server "$server" --target "$target_model" --draft "$draft_model" \
        --output-dir "$output" --widths 0,3 --n-predict 96 \
        --threads 4 --cpu-list 0,2,4,6 --context 512 --batch 128 --ubatch 64 \
        --port-base "$port" > "$run_dir/${label}-harness.log" 2>&1 || status=$?
    else
      env OMP_NUM_THREADS=4 KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
        KMP_LIBRARY=turnaround "$python_bin" "$mtp_harness" \
        --server "$server" --target "$target_model" --draft "$draft_model" \
        --output-dir "$output" --widths 0,3 --n-predict 96 \
        --threads 4 --cpu-list 0,2,4,6 --context 512 --batch 128 --ubatch 64 \
        --port-base "$port" > "$run_dir/${label}-harness.log" 2>&1 || status=$?
    fi
    if [[ "$status" -ne 0 ]]; then
      return "$status"
    fi
  fi
  "$python_bin" - "$output/summary.json" "$label" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
assert payload.get("terminal") is True
assert payload.get("passed") is True
assert payload["generation"]["prompt_count"] == 4
assert payload["generation"]["tokens_per_prompt"] == 96
assert {row["configuration"] for row in payload["configurations"]} == {"target", "mtp3"}
print(json.dumps({"event": "run_valid", "run": sys.argv[2]}))
PY
}

start_quality_server() {
  stop_quality_server
  setsid env OMP_NUM_THREADS=4 KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
    KMP_LIBRARY=turnaround taskset -c 0,2,4,6 \
    "$candidate_build/bin/llama-server" \
    -m "$target_model" -t 4 -tb 4 -c 8192 -np 1 -b 512 -ub 512 \
    --host 127.0.0.1 --port "$quality_port" --no-warmup --no-webui \
    > "$run_dir/quality-server.log" 2>&1 &
  quality_pid=$!
  for _ in $(seq 1 240); do
    if curl --silent --fail "http://127.0.0.1:${quality_port}/health" >/dev/null; then
      echo '{"event":"quality_server_ready"}'
      return 0
    fi
    if ! kill -0 "$quality_pid" 2>/dev/null; then
      tail -100 "$run_dir/quality-server.log" >&2 || true
      return 1
    fi
    sleep 0.5
  done
  echo "quality server was not ready within 120 seconds" >&2
  return 1
}

mkdir -p "$state_dir" "$run_dir"
trap 'stop_frequency; stop_quality_server' EXIT

if [[ -s "$state_dir/EXPERIMENT_COMPLETE" && -s "$state_dir/terminal-decision.json" ]]; then
  jq '{event:"terminal_resumed",passed,decision}' "$state_dir/terminal-decision.json"
  jq -e '.passed == true' "$state_dir/terminal-decision.json" >/dev/null
  exit $?
fi
if pgrep -f '/bin/llama-(server|bench|perplexity)( |$)' >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi
if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a Qwen 4B-era service is active; refusing to share the node" >&2
  exit 2
fi

for required in \
  "$prereg" "$summarizer" "$mtp_harness" "$baseline" \
  "$runtime_dir/eval_llama_cpp_cute_quick.py" \
  "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
  "$runtime_dir/eval_llama_cpp_wikitext_quick.py" \
  "$runtime_dir/eval_llama_cpp_mmlu.py" \
  "$runtime_dir/gate_b_protocol.py" \
  "$cute_manifest" "$math_code_manifest" "$mmlu_manifest" "$wiki_corpus" \
  "$parent_build/bin/llama-server" "$parent_build/bin/libggml-cpu.so.0.20.0" \
  "$candidate_build/bin/llama-server" "$candidate_build/bin/llama-perplexity" \
  "$candidate_build/bin/libggml-cpu.so.0.20.0" "$target_model" "$draft_model"; do
  test -f "$required"
done
test "$(stat -c %s "$target_model")" = 15606631072
test "$(stat -c %s "$draft_model")" = 1680271648
require_sha256 "$target_model" "$expected_target_sha"
require_sha256 "$draft_model" "$expected_draft_sha"
require_sha256 "$parent_build/bin/llama-server" "$expected_parent_server_sha"
require_sha256 "$parent_build/bin/libggml-cpu.so.0.20.0" "$expected_parent_cpu_sha"
require_sha256 "$candidate_build/bin/llama-server" "$expected_candidate_server_sha"
require_sha256 "$candidate_build/bin/llama-perplexity" "$expected_candidate_perplexity_sha"
require_sha256 "$candidate_build/bin/libggml-cpu.so.0.20.0" "$expected_candidate_cpu_sha"
require_sha256 "$prereg" "${EXPECTED_PREREG_SHA256:?EXPECTED_PREREG_SHA256 is required}"
require_sha256 "$summarizer" "${EXPECTED_SUMMARIZER_SHA256:?EXPECTED_SUMMARIZER_SHA256 is required}"
require_sha256 "$runtime_dir/run_qwen38_27b_clang21_serving_transfer_v35.sh" \
  "${EXPECTED_RUNNER_SHA256:?EXPECTED_RUNNER_SHA256 is required}"
require_sha256 "$baseline" 1557f1aac13f6c557e89006427aa829e2707f07ad73aa29611763e754c409a64
require_sha256 "$cute_manifest" 19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
require_sha256 "$math_code_manifest" 1ec0b41ce3ca3597d65013ac0c8a48c208c4e03b05f0109be61319252c09dab0
require_sha256 "$mmlu_manifest" e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba
require_sha256 "$wiki_corpus" 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08

sha256sum \
  "$prereg" "$summarizer" "$runtime_dir/run_qwen38_27b_clang21_serving_transfer_v35.sh" \
  "$mtp_harness" "$baseline" "$cute_manifest" "$math_code_manifest" "$mmlu_manifest" \
  "$runtime_dir/eval_llama_cpp_cute_quick.py" \
  "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
  "$runtime_dir/eval_llama_cpp_wikitext_quick.py" \
  "$runtime_dir/eval_llama_cpp_mmlu.py" "$runtime_dir/gate_b_protocol.py" "$wiki_corpus" \
  "$parent_build/bin/llama-server" "$parent_build/bin/libggml-cpu.so.0.20.0" \
  "$candidate_build/bin/llama-server" "$candidate_build/bin/llama-perplexity" \
  "$candidate_build/bin/libggml-cpu.so.0.20.0" "$target_model" "$draft_model" \
  > "$state_dir/sealed-inputs.sha256"

if [[ ! -s "$state_dir/frequency.log" ]] \
  || [[ ! -s "$run_dir/parent_a/summary.json" ]] \
  || [[ ! -s "$run_dir/candidate_a/summary.json" ]] \
  || [[ ! -s "$run_dir/candidate_b/summary.json" ]] \
  || [[ ! -s "$run_dir/parent_b/summary.json" ]]; then
  start_frequency
  run_mtp parent_a parent 20900
  run_mtp candidate_a candidate 21000
  run_mtp candidate_b candidate 21100
  run_mtp parent_b parent 21200
  stop_frequency
fi

echo '{"event":"always_run_quick_quality_start","denominators":{"cute":20,"math500":8,"code_trace":6,"wikitext_chunks":4}}'
start_quality_server
if [[ ! -s "$state_dir/cute.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_cute_quick.py" \
    --manifest "$cute_manifest" --server "http://127.0.0.1:${quality_port}" \
    --model-label qwen38-27b-clang21-active-v35 --model-sha256 "$expected_target_sha" \
    --out "$state_dir/cute.json" > "$run_dir/cute.log" 2>&1
fi
if [[ ! -s "$state_dir/math-code.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
    --manifest "$math_code_manifest" --server "http://127.0.0.1:${quality_port}" \
    --model-label qwen38-27b-clang21-active-v35 --model-sha256 "$expected_target_sha" \
    --out "$state_dir/math-code.json" > "$run_dir/math-code.log" 2>&1
fi
stop_quality_server
if [[ ! -s "$state_dir/wikitext.json" ]]; then
  env OMP_NUM_THREADS=4 KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active \
    KMP_LIBRARY=turnaround "$python_bin" "$runtime_dir/eval_llama_cpp_wikitext_quick.py" \
    --model "$target_model" --model-label qwen38-27b-clang21-active-v35 \
    --corpus "$wiki_corpus" --llama-perplexity "$candidate_build/bin/llama-perplexity" \
    --chunks 4 --context 512 --threads 4 --out "$state_dir/wikitext.json" \
    > "$run_dir/wikitext.log" 2>&1
fi

pre_status=0
"$python_bin" "$summarizer" --preregistration "$prereg" --state "$state_dir" \
  --run-dir "$run_dir" --baseline "$baseline" \
  --out "$state_dir/terminal-decision.json" || pre_status=$?
if jq -e '.terminal == true' "$state_dir/terminal-decision.json" >/dev/null; then
  finish "$pre_status"
fi

echo '{"event":"conditional_mmlu_start","n":64}'
start_quality_server
if [[ ! -s "$state_dir/mmlu.json" ]]; then
  resume_args=()
  if [[ -s "$state_dir/mmlu.json.partial" ]]; then
    resume_args+=(--resume)
  fi
  PYTHONPATH="$runtime_dir" "$python_bin" "$runtime_dir/eval_llama_cpp_mmlu.py" \
    --manifest "$mmlu_manifest" --model "$target_model" \
    --model-label qwen38-27b-clang21-active-v35 --model-sha256 "$expected_target_sha" \
    --server "http://127.0.0.1:${quality_port}" --workers 1 --timeout 900 \
    --checkpoint-every 4 --out "$state_dir/mmlu.json" "${resume_args[@]}" \
    > "$run_dir/mmlu.log" 2>&1
fi
stop_quality_server
final_status=0
"$python_bin" "$summarizer" --preregistration "$prereg" --state "$state_dir" \
  --run-dir "$run_dir" --baseline "$baseline" --mmlu "$state_dir/mmlu.json" \
  --out "$state_dir/terminal-decision.json" || final_status=$?
finish "$final_status"
