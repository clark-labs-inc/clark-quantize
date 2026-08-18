#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
hf_bin=${HF_BIN:-/workspace/venvs/quant-cpu/bin/hf}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp}
bf16_model=${BF16_MODEL:-/workspace/models/controls/Qwen3.8-27B-BF16.gguf}
q4_model=${Q4_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
q3_model=${Q3_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q3_K_M.gguf}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-q3-k-m-baseline-v1}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q3-k-m-baseline-v1}
q4_quality_dir=${Q4_QUALITY_DIR:-/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1}
wiki_corpus=${WIKI_CORPUS:-/workspace/quant-runs/cq3-4b-ptq-scope-20260817/wikitext-2-raw/wiki.test.raw}
port=${PORT:-8096}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
pivot="$runtime_dir/qwen38_27b_cpu_throughput_pivot_v1.json"
cute_manifest="$runtime_dir/cute_quick_v1.json"
math_code_manifest="$runtime_dir/math_code_canary_v1.json"
mmlu_manifest="$runtime_dir/qwen35_4b_cpu_layer30_mmlu_v1.json"
server_unit=qwen38-q3-k-m-baseline-v1

bf16_bytes=53808281952
bf16_sha=5a3eedc837bcbd1365cdbf5b71e698df3122e76586ba872f07ce3ed4a9bfa97e
q4_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
llama_revision=4df29be4f4c3673f428170fda944a5b19f743bb8

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

cleanup() {
  systemctl stop "llama-cpu-eval-${server_unit}.service" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$run_dir" "$state_dir" "$(dirname "$q3_model")"
cd "$repo"
if [[ -e "$state_dir/BASELINE_COMPLETE" ]]; then
  echo "Q3_K_M baseline is already complete"
  exit 0
fi
if [[ -n "$(git status --short)" ]]; then
  echo "Q3_K_M baseline requires a clean source checkout" >&2
  exit 2
fi
if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a frozen Qwen 4B service is active" >&2
  exit 2
fi
conflicts=$(ps -eo pid=,comm=,args= | awk \
  '$2 ~ /^(llama-(server|cli|bench|perplexi|quantize)|python|python3|torchrun)$/ &&
   $0 ~ /(Qwen3[.]8-27B|qwen38|qwen.*qat|activation.*cache)/ {print}')
if [[ -n "$conflicts" ]]; then
  echo "another Qwen3.8 workload is active" >&2
  printf '%s\n' "$conflicts" >&2
  exit 2
fi

require_sha256 "$pivot" e56d6ccddb6959b65a75b5ba22592b6381b910887ac8d89c94b1d1d3dd14287f
require_sha256 "$cute_manifest" 19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
require_sha256 "$math_code_manifest" 1ec0b41ce3ca3597d65013ac0c8a48c208c4e03b05f0109be61319252c09dab0
require_sha256 "$mmlu_manifest" e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba
require_sha256 "$wiki_corpus" 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
require_sha256 "$q4_model" "$q4_sha"
require_sha256 "$llama_build/bin/llama-quantize" ca930cb5698665e73f2e133e3c63fe9a5940c15a91ba76e3398404380888e3f2
require_sha256 "$llama_build/bin/llama-server" c1c5af73afc267dfd9256e8f97bd871e7842353dd25518587f68261d731a24c9
require_sha256 "$llama_build/bin/llama-perplexity" 9941787d0c8d35ed67dce640dc3c3a0cc55750f672d31dcb1e6dcd641fd15f78
require_sha256 "$llama_build/bin/llama-bench" 01f85116456ffd6dbe4723801c86c2b75ed1b920a60416aef4f21d376114f7a9
if [[ "$(git -C "$llama_source" rev-parse HEAD)" != "$llama_revision" ]]; then
  echo "unexpected llama.cpp revision" >&2
  exit 2
fi
if [[ ! -s "$q4_quality_dir/QUALITY_COMPLETE" ]]; then
  echo "Q4 quality anchor is incomplete" >&2
  exit 2
fi

git rev-parse HEAD > "$state_dir/source.commit"
git -C "$llama_source" rev-parse HEAD > "$state_dir/llama.commit"
lscpu --json > "$state_dir/lscpu.json"
free -b > "$state_dir/memory-before.txt"

if [[ ! -s "$bf16_model" ]]; then
  available_bytes=$(df --output=avail -B1 /workspace | tail -1 | tr -d ' ')
  if (( available_bytes < 60000000000 )); then
    echo "at least 60 GB free is required before the pinned BF16 download" >&2
    exit 2
  fi
  "$hf_bin" download ggml-org/Qwen3.8-27B-GGUF \
    Qwen3.8-27B-BF16.gguf \
    --revision 0669b98607d47046c7c2b3f801011d54a08cfccf \
    --local-dir "$(dirname "$bf16_model")" \
    > "$run_dir/bf16-download.log" 2>&1
fi
if [[ "$(stat -c %s "$bf16_model")" != "$bf16_bytes" ]]; then
  echo "unexpected pinned BF16 byte count" >&2
  exit 2
fi
require_sha256 "$bf16_model" "$bf16_sha"
printf 'complete\n' > "$state_dir/BF16_INPUT_COMPLETE"

if [[ ! -s "$q3_model" ]]; then
  if [[ -e "$q3_model.partial" ]]; then
    echo "unmarked partial Q3 artifact requires inspection before retry" >&2
    exit 2
  fi
  /usr/bin/time -v "$llama_build/bin/llama-quantize" \
    "$bf16_model" "$q3_model.partial" Q3_K_M 4 \
    > "$run_dir/quantize.log" 2>&1
  mv "$q3_model.partial" "$q3_model"
fi
q3_sha=$(sha256sum "$q3_model" | awk '{print $1}')
q3_bytes=$(stat -c %s "$q3_model")
"$python_bin" - "$state_dir" "$q3_model" "$q3_sha" "$q3_bytes" <<'PY'
import json
import sys
from pathlib import Path

root, model, digest, size = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
payload = {
    "schema": "qwen38_27b_q3_k_m_materialization_v1",
    "model": model,
    "model_sha256": digest,
    "model_bytes": size,
    "source_bf16_sha256": "5a3eedc837bcbd1365cdbf5b71e698df3122e76586ba872f07ce3ed4a9bfa97e",
    "quantization": "Q3_K_M",
    "threads": 4,
    "terminal": True,
}
temporary = root / "materialization.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "materialization.json")
PY
printf 'complete\n' > "$state_dir/Q3_MATERIALIZATION_COMPLETE"

if [[ ! -e "$state_dir/QUICK_QUALITY_COMPLETE" ]]; then
  if [[ ! -s "$state_dir/load-smoke.log" ]]; then
    /usr/bin/time -v "$llama_build/bin/llama-cli" \
      -m "$q3_model" -t 4 -tb 4 -c 2048 -b 512 -ub 512 \
      --no-warmup --single-turn --simple-io \
      -p 'Reply with only OK.' -n 2 --temp 0 \
      > "$state_dir/load-smoke.log" 2>&1
  fi
  CPU_EVAL_RUN_DIR="$run_dir" \
  LLAMA_PARALLEL=1 LLAMA_CONTEXT=8192 LLAMA_THREADS=4 \
    "$runtime_dir/launch_llama_cpu_eval_server.sh" \
      "$q3_model" "$server_unit" "$port"

  if [[ ! -s "$state_dir/cute.json" ]]; then
    "$python_bin" "$runtime_dir/eval_llama_cpp_cute_quick.py" \
      --manifest "$cute_manifest" --server "http://127.0.0.1:${port}" \
      --model-label "$server_unit" --model-sha256 "$q3_sha" \
      --out "$state_dir/cute.json" > "$run_dir/cute.log" 2>&1
  fi
  if [[ ! -s "$state_dir/math-code.json" ]]; then
    "$python_bin" "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
      --manifest "$math_code_manifest" --server "http://127.0.0.1:${port}" \
      --model-label "$server_unit" --model-sha256 "$q3_sha" \
      --out "$state_dir/math-code.json" > "$run_dir/math-code.log" 2>&1
  fi
  cleanup
  if [[ ! -s "$state_dir/wikitext.json" ]]; then
    "$python_bin" "$runtime_dir/eval_llama_cpp_wikitext_quick.py" \
      --model "$q3_model" --model-label "$server_unit" \
      --corpus "$wiki_corpus" \
      --llama-perplexity "$llama_build/bin/llama-perplexity" \
      --chunks 4 --context 512 --threads 4 \
      --out "$state_dir/wikitext.json" > "$run_dir/wikitext.log" 2>&1
  fi
  "$python_bin" - "$state_dir" "$q3_sha" "$q4_quality_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

root, model_sha, q4_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cute = json.loads((root / "cute.json").read_text())
math_code = json.loads((root / "math-code.json").read_text())
wiki = json.loads((root / "wikitext.json").read_text())
q4 = json.loads(q4_path.read_text())
cute_correct = cute["summaries"][-1]["correct"]
scores = {row["task"]: row["correct"] for row in math_code["summaries"]}
ppl = wiki["final_ppl"]
checks = {
    "cute": cute_correct >= 7,
    "math500": scores["math500"] >= 4,
    "code": scores["code_trace"] >= 4,
    "wikitext": ppl / q4["wikitext_ppl"] <= 1.10,
}
payload = {
    "schema": "qwen38_27b_q3_k_m_quick_quality_v1",
    "model_sha256": model_sha,
    "observed": {
        "cute_correct_of_20": cute_correct,
        "math500_correct_of_8": scores["math500"],
        "code_correct_of_6": scores["code_trace"],
        "wikitext_ppl": ppl,
        "wikitext_ppl_ratio_vs_q4": ppl / q4["wikitext_ppl"],
    },
    "checks": checks,
    "pass": all(checks.values()),
    "terminal": True,
}
temporary = root / "quick-quality-summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "quick-quality-summary.json")
PY
  printf 'complete\n' > "$state_dir/QUICK_QUALITY_COMPLETE"
fi

if ! "$python_bin" - "$state_dir/quick-quality-summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["pass"] else 1)
PY
then
  printf 'rejected\n' > "$state_dir/QUICK_QUALITY_REJECTED"
  exit 0
fi

run_screen_row() {
  local label=$1
  local model=$2
  local mode=$3
  local prefill="$state_dir/${label}-prefill-t4.json"
  local decode="$state_dir/${label}-decode-t4.json"
  if [[ "$mode" == prefill && ! -s "$prefill" ]]; then
    taskset -c 0,2,4,6 "$llama_build/bin/llama-bench" \
      -m "$model" -p 512 -n 0 -t 4 -C 0x55 --cpu-strict 1 \
      -b 512 -ub 512 -r 5 --poll 50 -ngl 0 --load-mode mmap -o json \
      > "$prefill.tmp" 2> "$run_dir/${label}-prefill-t4.log"
    mv "$prefill.tmp" "$prefill"
  fi
  if [[ "$mode" == decode && ! -s "$decode" ]]; then
    taskset -c 0,2,4,6 "$llama_build/bin/llama-bench" \
      -m "$model" -p 0 -n 128 -t 4 -C 0x55 --cpu-strict 1 \
      -b 512 -ub 512 -r 5 --poll 50 -ngl 0 --load-mode mmap -o json \
      > "$decode.tmp" 2> "$run_dir/${label}-decode-t4.log"
    mv "$decode.tmp" "$decode"
  fi
}

# Alternate model order across the two product surfaces to reduce order bias.
run_screen_row q4 "$q4_model" prefill
run_screen_row q3 "$q3_model" prefill
run_screen_row q3 "$q3_model" decode
run_screen_row q4 "$q4_model" decode
"$python_bin" - "$state_dir" "$q4_model" "$q3_model" "$q4_sha" "$q3_sha" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = {"q4": str(Path(sys.argv[2]).resolve()), "q3": str(Path(sys.argv[3]).resolve())}
hashes = {"q4": sys.argv[4], "q3": sys.argv[5]}
rows = {}
for label in ("q4", "q3"):
    rows[label] = {}
    for mode in ("prefill", "decode"):
        raw = json.loads((root / f"{label}-{mode}-t4.json").read_text())
        assert isinstance(raw, list) and len(raw) == 1
        item = raw[0]
        assert item["model_filename"] == models[label]
        assert int(item["n_threads"]) == 4 and item["cpu_strict"] is True
        samples = [float(value) for value in item["samples_ts"]]
        assert len(samples) == 5
        rows[label][mode] = {
            "median_tokens_per_second": statistics.median(samples),
            "samples_tokens_per_second": samples,
        }
decode_speedup = rows["q3"]["decode"]["median_tokens_per_second"] / rows["q4"]["decode"]["median_tokens_per_second"]
prefill_fraction = rows["q3"]["prefill"]["median_tokens_per_second"] / rows["q4"]["prefill"]["median_tokens_per_second"]
checks = {
    "decode_at_least_2_25": rows["q3"]["decode"]["median_tokens_per_second"] >= 2.25,
    "decode_speedup_at_least_1_15": decode_speedup >= 1.15,
    "prefill_fraction_at_least_0_90": prefill_fraction >= 0.90,
}
payload = {
    "schema": "qwen38_27b_q3_k_m_speed_screen_v1",
    "models": {key: {"path": models[key], "sha256": hashes[key]} for key in models},
    "rows": rows,
    "decode_speedup_vs_q4": decode_speedup,
    "prefill_fraction_vs_q4": prefill_fraction,
    "checks": checks,
    "pass": all(checks.values()),
    "terminal": True,
}
temporary = root / "speed-screen-summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "speed-screen-summary.json")
PY
printf 'complete\n' > "$state_dir/SPEED_SCREEN_COMPLETE"

if ! "$python_bin" - "$state_dir/speed-screen-summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["pass"] else 1)
PY
then
  printf 'rejected\n' > "$state_dir/SPEED_SCREEN_REJECTED"
  exit 0
fi

if [[ ! -s "$state_dir/mmlu.json" ]]; then
  CPU_EVAL_RUN_DIR="$run_dir" \
  LLAMA_PARALLEL=1 LLAMA_CONTEXT=8192 LLAMA_THREADS=4 \
    "$runtime_dir/launch_llama_cpu_eval_server.sh" \
      "$q3_model" "$server_unit" "$port"
  resume_args=()
  if [[ -s "$state_dir/mmlu.json.partial" ]]; then
    resume_args+=(--resume)
  fi
  PYTHONPATH="$repo" "$python_bin" "$runtime_dir/eval_llama_cpp_mmlu.py" \
    --manifest "$mmlu_manifest" --model "$q3_model" \
    --model-label "$server_unit" --model-sha256 "$q3_sha" \
    --server "http://127.0.0.1:${port}" --workers 1 --timeout 900 \
    --checkpoint-every 4 --out "$state_dir/mmlu.json" \
    "${resume_args[@]}" > "$run_dir/mmlu.log" 2>&1
  cleanup
fi

"$python_bin" - "$state_dir" "$q3_sha" <<'PY'
import json
import sys
from pathlib import Path

root, model_sha = Path(sys.argv[1]), sys.argv[2]
quick = json.loads((root / "quick-quality-summary.json").read_text())
speed = json.loads((root / "speed-screen-summary.json").read_text())
mmlu = json.loads((root / "mmlu.json").read_text())
mmlu_correct = mmlu["summary"]["correct"]
checks = {
    "quick_quality": quick["pass"],
    "speed_screen": speed["pass"],
    "mmlu": mmlu_correct >= 49,
}
payload = {
    "schema": "qwen38_27b_q3_k_m_baseline_summary_v1",
    "model_sha256": model_sha,
    "quick_quality": quick,
    "speed": speed,
    "mmlu_correct_of_64": mmlu_correct,
    "checks": checks,
    "pass": all(checks.values()),
    "terminal": True,
    "claim_boundary": "Q3_K_M practical baseline only; not a Clark research breakthrough",
}
temporary = root / "summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "summary.json")
PY
free -b > "$state_dir/memory-after.txt"
sha256sum "$pivot" "$bf16_model" "$q4_model" "$q3_model" \
  "$state_dir/materialization.json" "$state_dir/quick-quality-summary.json" \
  "$state_dir/speed-screen-summary.json" "$state_dir/mmlu.json" \
  "$state_dir/summary.json" > "$state_dir/artifacts.sha256"
printf 'complete\n' > "$state_dir/BASELINE_COMPLETE"
