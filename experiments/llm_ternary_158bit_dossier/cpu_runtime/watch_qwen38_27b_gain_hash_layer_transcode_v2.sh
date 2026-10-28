#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-gain-transcode-v2}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-gain-hash-layer-transcode-watch-v2}
result=${RESULT_RECEIPT:-/workspace/quant-state/qwen38-27b-gain-hash-layer-transcode-v2.json}
expected_commit_file=${EXPECTED_COMMIT_FILE:-$state/expected.commit}
idle_timeout_seconds=${IDLE_TIMEOUT_SECONDS:-43200}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

if [[ -s "$result" ]] && "$python_bin" - "$result" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("complete") is True and payload.get("terminal") is True else 1)
PY
then
  exit 0
fi

if [[ ! -s "$expected_commit_file" ]]; then
  echo "gain-hash transcode v2 expected commit is absent" >&2
  exit 64
fi
expected_commit=$(tr -d '[:space:]' < "$expected_commit_file")
if [[ ! "$expected_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "gain-hash transcode v2 expected commit is invalid" >&2
  exit 65
fi
cd "$repo"
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "gain-hash transcode v2 worktree is not at its deployed commit" >&2
  exit 66
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "gain-hash transcode v2 worktree has tracked changes" >&2
  exit 73
fi

mkdir -p "$state"
date -u +%FT%TZ > "$state/started-at.txt"
git rev-parse HEAD > "$state/repository.commit"
sha256sum \
  "$runtime/qwen38_27b_gain_hash_layer_transcode_preregistration_v1.json" \
  "$runtime/qwen38_27b_gain_hash_layer_transcode_retry_amendment_v2.json" \
  "$runtime/benchmark_qwen38_27b_gain_hash_layer_transcode_v1.py" \
  "$runtime/receipts/qwen38-gain-hash-layer-transcode-v1.json" \
  clark_quantize/qwen38_gain_hash_layer_transcode.c \
  clark_quantize/qwen38_gain_hash_layer_transcode.py \
  clark_quantize/qwen38_fixed_support_cpu.c \
  clark_quantize/qwen38_fixed_support_mask64_cpu.c \
  > "$state/source.sha256"

"$python_bin" - "$runtime/receipts/qwen38-gain-hash-layer-transcode-v1.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("gate", {}).get("passed") is not True:
    raise SystemExit("gain-hash transcode correctness parent did not pass")
PY

deadline=$((SECONDS + idle_timeout_seconds))
while true; do
  llama_conflicts=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^llama-(server|cli|bench|perplexi)/ {print}')
  qwen_python_conflicts=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^python/ && $0 ~ /qwen38/ && $0 !~ /gain_hash_layer_transcode_v2/ {print}')
  if [[ -z "$llama_conflicts" && -z "$qwen_python_conflicts" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for an idle Qwen CPU boundary" >&2
    printf '%s\n' "$llama_conflicts" >&2
    printf '%s\n' "$qwen_python_conflicts" >&2
    exit 76
  fi
  sleep 30
done

if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((2 * 1024 * 1024)) ]]; then
  echo "less than 2 GiB is available for gain-hash transcode v2" >&2
  exit 72
fi
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "swap is active during gain-hash transcode v2 admission" >&2
  exit 74
fi
if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
  echo "less than 10 GiB disk is free for gain-hash transcode v2" >&2
  exit 75
fi

free -b > "$state/memory-before.txt"
set +e
taskset -c 0 env PYTHONPATH="$repo" "$python_bin" \
  "$runtime/benchmark_qwen38_27b_gain_hash_layer_transcode_v1.py" \
  --require-x86 \
  --repeats 5 \
  --out "$result" \
  > "$state/benchmark.stdout" 2> "$state/benchmark.stderr"
benchmark_status=$?
set -e
if [[ ! -s "$result" ]]; then
  echo "gain-hash transcode v2 exited without a receipt" >&2
  exit 70
fi
if (( benchmark_status != 0 && benchmark_status != 2 )); then
  echo "gain-hash transcode v2 had unexpected exit $benchmark_status" >&2
  exit "$benchmark_status"
fi
"$python_bin" - "$result" "$benchmark_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True or payload.get("terminal") is not True:
    raise SystemExit("gain-hash transcode v2 receipt is not terminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("gain-hash transcode v2 status and gate disagree")
PY
if (( benchmark_status == 0 )); then
  printf 'pass\n' > "$state/GAIN_HASH_LAYER_TRANSCODE_PASS"
else
  printf 'valid rejection\n' > "$state/GAIN_HASH_LAYER_TRANSCODE_REJECTED"
fi
free -b > "$state/memory-after.txt"
date -u +%FT%TZ > "$state/completed-at.txt"
sha256sum "$result" > "$state/result.sha256"
printf 'terminal\n' > "$state/WATCH_TERMINAL"
