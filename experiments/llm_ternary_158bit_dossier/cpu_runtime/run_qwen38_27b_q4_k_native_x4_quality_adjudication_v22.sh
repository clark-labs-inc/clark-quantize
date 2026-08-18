#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-row-scale-v3}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-q4k-native-x4-deinterleave-v21}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp-q4k-native-x4-deinterleave-v21}
model=${MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
baseline_state=${BASELINE_STATE:-/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-q4-k-native-x4-quality-v22}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q4-k-native-x4-quality-v22}
wiki_corpus=${WIKI_CORPUS:-/workspace/quant-runs/cq3-4b-ptq-scope-20260817/wikitext-2-raw/wiki.test.raw}
port=${PORT:-8098}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_q4_k_native_x4_quality_adjudication_v22.json"
manifest_amendment="$runtime_dir/qwen38_27b_q4_k_native_x4_quality_adjudication_v22_manifest_resolution_amendment.json"
predecessor="$runtime_dir/receipts/qwen38_27b_q4_k_native_x4_deinterleave_fused_avx2_v21/system-decision.json"
patch="$runtime_dir/kernels/llama_cpp_q4_k_native_x4_deinterleave_fused_avx2_v21.patch"
cute_manifest="$runtime_dir/cute_quick_v1.json"
math_code_manifest="$runtime_dir/math_code_canary_v1.json"
mmlu_manifest="$runtime_dir/qwen35_4b_cpu_layer30_mmlu_v1.json"
summarizer="$runtime_dir/summarize_qwen38_27b_q4_k_native_x4_quality_adjudication_v22.py"
label=qwen38-27b-q4-native-x4-v22
server_unit="llama-cpu-eval-${label}.service"
expected_model_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
expected_model_bytes=18973870432

require_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

cleanup_server() {
  systemctl stop "$server_unit" >/dev/null 2>&1 || true
}

start_server() {
  cleanup_server
  systemctl reset-failed "$server_unit" >/dev/null 2>&1 || true
  systemd-run --unit="${server_unit%.service}" --collect \
    --property=MemoryMax=28G \
    --property=CPUQuota=800% \
    --property=Restart=no \
    /usr/bin/taskset -c 0,2,4,6 \
    "$llama_build/bin/llama-server" \
    -m "$model" -t 4 -tb 4 -c 8192 -np 1 -b 512 -ub 512 \
    --host 127.0.0.1 --port "$port" --no-warmup \
    --log-file "$run_dir/server-${label}.log"
  for _ in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null; then
      echo '{"event":"candidate_server_ready"}'
      return 0
    fi
    if ! systemctl is-active --quiet "$server_unit"; then
      echo "$server_unit exited before becoming healthy" >&2
      systemctl status "$server_unit" --no-pager >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "$server_unit did not become healthy within 180 seconds" >&2
  return 1
}

finish() {
  local rc=$1
  cleanup_server
  sha256sum "$state_dir/cute.json" "$state_dir/math-code.json" \
    "$state_dir/wikitext.json" "$state_dir/decision.json" \
    > "$state_dir/artifacts.sha256"
  if [[ -s "$state_dir/mmlu.json" ]]; then
    sha256sum "$state_dir/mmlu.json" >> "$state_dir/artifacts.sha256"
  fi
  printf 'complete\n' > "$state_dir/TERMINAL_COMPLETE"
  echo "{\"event\":\"quality_harness_terminal\",\"exit_code\":${rc}}"
  exit "$rc"
}

trap cleanup_server EXIT
mkdir -p "$run_dir" "$state_dir"
cd "$repo"

if [[ -e "$state_dir/TERMINAL_COMPLETE" ]]; then
  (cd / && sha256sum -c "$state_dir/artifacts.sha256")
  rc=$("$python_bin" - "$state_dir/decision.json" <<'PY'
import json
import sys
print(0 if json.load(open(sys.argv[1]))["passed"] else 2)
PY
)
  echo "terminal quality receipt already exists"
  exit "$rc"
fi
if [[ -n "$(git status --short)" ]]; then
  echo "quality adjudication requires a clean research checkout" >&2
  exit 2
fi
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null || pgrep -x llama-perplexity >/dev/null; then
  echo "another llama.cpp workload is active" >&2
  exit 2
fi
if systemctl list-units --type=service --state=running --no-legend | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a Qwen 4B-era service is active; refusing to share the node" >&2
  exit 2
fi
if [[ ! -s "$baseline_state/QUALITY_COMPLETE" || ! -s "$baseline_state/summary.json" ]]; then
  echo "pinned Qwen3.8-27B Q4 baseline is incomplete" >&2
  exit 3
fi
if [[ "$(git -C "$llama_source" rev-parse HEAD)" != 4df29be4f4c3673f428170fda944a5b19f743bb8 ]]; then
  echo "unexpected llama.cpp source revision" >&2
  exit 2
fi
if [[ "$(stat -c %s "$model")" != "$expected_model_bytes" ]]; then
  echo "unexpected Qwen3.8-27B Q4 model size" >&2
  exit 2
fi

require_sha256 "$prereg" 2c1680cd595d914011d584915a1c51c71afa7ce7350c5d759215c797c42bcdcd
require_sha256 "$manifest_amendment" 5ec58787e7c3307ebb0a61492c9e957d5f98d266c49aca7273c9d94e3ccd465c
require_sha256 "$predecessor" ad73826a56cd903589c88277e8759f92623286d5428f9ad10d004f4be71779a5
require_sha256 "$patch" feedcb6bcf6b0ded4181a4cc234d8cef08731234ed494f31b43b107884531b69
require_sha256 "$baseline_state/summary.json" 1557f1aac13f6c557e89006427aa829e2707f07ad73aa29611763e754c409a64
require_sha256 "$cute_manifest" 19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
require_sha256 "$math_code_manifest" 1ec0b41ce3ca3597d65013ac0c8a48c208c4e03b05f0109be61319252c09dab0
require_sha256 "$mmlu_manifest" e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba
require_sha256 "$wiki_corpus" 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
require_sha256 "$runtime_dir/eval_llama_cpp_cute_quick.py" 9c40a48a0c3f0544156a20ccccdc12f7e90bbae8afd391bf640ead1ff234c8ab
require_sha256 "$runtime_dir/eval_llama_cpp_math_code_canary.py" fa3d0f870b3b0ed1d7d907cc8d146854b164e6d75cfa9d2bd20743979af74677
require_sha256 "$runtime_dir/eval_llama_cpp_mmlu.py" a604ca9a528231fa17de78ab2a7ffed7c1b9a0b9a9ca74459c2d894f1f3612d3
require_sha256 "$runtime_dir/eval_llama_cpp_wikitext_quick.py" 54458dc377a1d07aaf26df90823f65f973b625f55a6ac73895d0c6ec9a666b07
require_sha256 "$llama_build/bin/llama-server" 2c9db28068c8cf92f92fa631e1667e5d1cb39c83e21ae3245bb655d69ec358fc
require_sha256 "$llama_build/bin/llama-perplexity" 1da55f36b046de884b9fcb9a9a5d1f216acb84d884314a9a2f4edd5a5f1ecdd7
require_sha256 "$llama_build/bin/libllama.so" df3d0f87d43b8f90833bfe59fd4bb58a79805063b12d0d385ee74aa43131f994
require_sha256 "$llama_build/bin/libggml.so" 9ce14fedbb5ba89199e24bc6434696cc164f3878ae7c15571f55c1d47b2afc92
require_sha256 "$llama_build/bin/libggml-cpu.so" f6a1b46830bda9e322ff77d8d203eeb835b48cda0986cb547ea6141d08f95029
require_sha256 "$llama_build/bin/libllama-common.so" 8148ac00f190e53aac7768335575676fb1b0cc7efc20ffc4759778f71eb5b215
require_sha256 "$model" "$expected_model_sha"

git rev-parse HEAD > "$state_dir/source.commit"
git -C "$llama_source" diff --binary > "$state_dir/llama-source.diff"
sha256sum "$prereg" "$manifest_amendment" "$predecessor" "$patch" \
  "$baseline_state/summary.json" "$cute_manifest" "$math_code_manifest" \
  "$mmlu_manifest" "$wiki_corpus" "$runtime_dir/eval_llama_cpp_cute_quick.py" \
  "$runtime_dir/eval_llama_cpp_math_code_canary.py" "$runtime_dir/eval_llama_cpp_mmlu.py" \
  "$runtime_dir/eval_llama_cpp_wikitext_quick.py" "$summarizer" \
  "$runtime_dir/run_qwen38_27b_q4_k_native_x4_quality_adjudication_v22.sh" \
  "$llama_build/bin/llama-server" "$llama_build/bin/llama-perplexity" \
  "$llama_build/bin/libllama.so" "$llama_build/bin/libggml.so" \
  "$llama_build/bin/libggml-cpu.so" "$llama_build/bin/libllama-common.so" \
  "$model" > "$state_dir/inputs.sha256"

echo '{"event":"quality_gate_1_start","denominators":{"cute":20,"math500":8,"code_trace":6,"wikitext_chunks":4}}'
start_server

if [[ ! -s "$state_dir/cute.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_cute_quick.py" \
    --manifest "$cute_manifest" --server "http://127.0.0.1:${port}" \
    --model-label "$label" --model-sha256 "$expected_model_sha" \
    --out "$state_dir/cute.json" > "$run_dir/cute.log" 2>&1
fi
echo '{"event":"cute_complete","n":20}'

if [[ ! -s "$state_dir/math-code.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
    --manifest "$math_code_manifest" --server "http://127.0.0.1:${port}" \
    --model-label "$label" --model-sha256 "$expected_model_sha" \
    --out "$state_dir/math-code.json" > "$run_dir/math-code.log" 2>&1
fi
echo '{"event":"math_code_complete","math500_n":8,"code_trace_n":6}'
cleanup_server

if [[ ! -s "$state_dir/wikitext.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_wikitext_quick.py" \
    --model "$model" --model-label "$label" --corpus "$wiki_corpus" \
    --llama-perplexity "$llama_build/bin/llama-perplexity" \
    --chunks 4 --context 512 --threads 4 --out "$state_dir/wikitext.json" \
    > "$run_dir/wikitext.log" 2>&1
fi
echo '{"event":"wikitext_complete","chunks":4}'

stage_1_rc=0
"$python_bin" "$summarizer" \
  --preregistration "$prereg" --manifest-amendment "$manifest_amendment" \
  --baseline "$baseline_state/summary.json" --predecessor "$predecessor" \
  --cute "$state_dir/cute.json" --math-code "$state_dir/math-code.json" \
  --wikitext "$state_dir/wikitext.json" --out "$state_dir/decision.json" \
  || stage_1_rc=$?
if (( stage_1_rc != 0 )); then
  finish "$stage_1_rc"
fi

echo '{"event":"quality_gate_2_start","mmlu_n":64}'
start_server
if [[ ! -s "$state_dir/mmlu.json" ]]; then
  resume_args=()
  if [[ -s "$state_dir/mmlu.json.partial" ]]; then
    resume_args+=(--resume)
  fi
  PYTHONPATH="$repo" "$python_bin" "$runtime_dir/eval_llama_cpp_mmlu.py" \
    --manifest "$mmlu_manifest" --model "$model" --model-label "$label" \
    --model-sha256 "$expected_model_sha" --server "http://127.0.0.1:${port}" \
    --workers 1 --timeout 900 --checkpoint-every 4 --out "$state_dir/mmlu.json" \
    "${resume_args[@]}" > "$run_dir/mmlu.log" 2>&1
fi
echo '{"event":"mmlu_complete","n":64}'
cleanup_server

final_rc=0
"$python_bin" "$summarizer" \
  --preregistration "$prereg" --manifest-amendment "$manifest_amendment" \
  --baseline "$baseline_state/summary.json" --predecessor "$predecessor" \
  --cute "$state_dir/cute.json" --math-code "$state_dir/math-code.json" \
  --wikitext "$state_dir/wikitext.json" --mmlu "$state_dir/mmlu.json" \
  --out "$state_dir/decision.json" || final_rc=$?
finish "$final_rc"
