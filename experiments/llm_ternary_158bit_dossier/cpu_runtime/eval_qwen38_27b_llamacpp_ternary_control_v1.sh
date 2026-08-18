#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-tq-controls-v1}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-qwen38-selective-v1}
model=${MODEL:?MODEL is required}
format=${FORMAT:?FORMAT is required}
model_sha=${MODEL_SHA256:?MODEL_SHA256 is required}
model_bytes=${MODEL_BYTES:?MODEL_BYTES is required}
port=${PORT:?PORT is required}
run_dir=${RUN_DIR:?RUN_DIR is required}
state_dir=${STATE_DIR:?STATE_DIR is required}
wiki_corpus=${WIKI_CORPUS:-/workspace/quant-runs/cq3-4b-ptq-scope-20260817/wikitext-2-raw/wiki.test.raw}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime/qwen38_27b_llamacpp_ternary_controls_preregistration_v1.json"
cute_manifest="$runtime/cute_quick_v1.json"
math_code_manifest="$runtime/math_code_canary_v1.json"
mmlu_manifest="$runtime/qwen35_4b_cpu_layer30_mmlu_v1.json"
label="qwen38-27b-${format,,}-v1"
server_unit="llama-cpu-eval-${label}.service"

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
  systemctl stop "$server_unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$run_dir" "$state_dir"
cd "$repo"
if [[ -e "$state_dir/QUALITY_COMPLETE" ]]; then
  echo "quality marker already exists: $state_dir/QUALITY_COMPLETE"
  exit 0
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "TQ control evaluation requires a clean source checkout" >&2
  exit 2
fi
if [[ "$(stat -c %s "$model")" != "$model_bytes" ]]; then
  echo "unexpected $format model byte count" >&2
  exit 2
fi
require_sha256 "$model" "$model_sha"
require_sha256 "$cute_manifest" 19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
require_sha256 "$math_code_manifest" 1ec0b41ce3ca3597d65013ac0c8a48c208c4e03b05f0109be61319252c09dab0
require_sha256 "$mmlu_manifest" e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba
require_sha256 "$wiki_corpus" 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
require_sha256 "$llama_build/bin/llama-server" 53d64ef9635604ca1e6f0d9feec7fbe2ed87c30d0c6f3daae22aae02d3555aa3
require_sha256 "$llama_build/bin/llama-perplexity" 1a5fd9403c959fd4e6c1342778ffce747f990532ead5e20227396c34db5693d9
require_sha256 "$llama_build/bin/llama-cli" a80bb7369cd510bf82a7eafe79fb9521c5d78610831fca0f9f6ac8e45a873a40
require_sha256 "$llama_build/bin/llama-bench" 155eb2f52f2b55429885d93bcb5cedae6ab68285d694d2f91ac8c3fd789e51f3

git rev-parse HEAD > "$state_dir/source.commit"
printf '%s\n' "$model_sha" > "$state_dir/model.sha256"
printf '%s\n' "$model_bytes" > "$state_dir/model.bytes"
sha256sum "$prereg" "$cute_manifest" "$math_code_manifest" \
  "$mmlu_manifest" "$wiki_corpus" "$model" \
  "$llama_build/bin/llama-server" "$llama_build/bin/llama-perplexity" \
  "$llama_build/bin/llama-cli" "$llama_build/bin/llama-bench" \
  > "$state_dir/inputs.sha256"

if [[ ! -e "$state_dir/LOAD_SMOKE_COMPLETE" ]]; then
  if [[ -s "$state_dir/load-smoke.log" ]]; then
    echo "unmarked $format load-smoke log requires inspection" >&2
    exit 2
  fi
  /usr/bin/time -v "$llama_build/bin/llama-cli" \
    -m "$model" -t 4 -tb 4 -c 2048 -b 512 -ub 512 \
    --no-warmup --single-turn --simple-io \
    -p 'Reply with only OK.' -n 2 --temp 0 \
    > "$state_dir/load-smoke.log" 2>&1
  printf 'complete\n' > "$state_dir/LOAD_SMOKE_COMPLETE"
fi

CPU_EVAL_RUN_DIR="$run_dir" \
LLAMA_SERVER_BIN="$llama_build/bin/llama-server" \
LLAMA_PARALLEL=1 LLAMA_CONTEXT=8192 LLAMA_THREADS=4 \
  "$runtime/launch_llama_cpu_eval_server.sh" "$model" "$label" "$port"

if [[ ! -s "$state_dir/cute.json" ]]; then
  "$python_bin" "$runtime/eval_llama_cpp_cute_quick.py" \
    --manifest "$cute_manifest" \
    --server "http://127.0.0.1:${port}" \
    --model-label "$label" \
    --model-sha256 "$model_sha" \
    --out "$state_dir/cute.json" \
    > "$run_dir/cute.log" 2>&1
fi
if [[ ! -s "$state_dir/math-code.json" ]]; then
  "$python_bin" "$runtime/eval_llama_cpp_math_code_canary.py" \
    --manifest "$math_code_manifest" \
    --server "http://127.0.0.1:${port}" \
    --model-label "$label" \
    --model-sha256 "$model_sha" \
    --out "$state_dir/math-code.json" \
    > "$run_dir/math-code.log" 2>&1
fi
cleanup

if [[ ! -s "$state_dir/wikitext.json" ]]; then
  "$python_bin" "$runtime/eval_llama_cpp_wikitext_quick.py" \
    --model "$model" \
    --model-label "$label" \
    --corpus "$wiki_corpus" \
    --llama-perplexity "$llama_build/bin/llama-perplexity" \
    --chunks 4 --context 512 --threads 4 \
    --out "$state_dir/wikitext.json" \
    > "$run_dir/wikitext.log" 2>&1
fi

"$python_bin" - "$state_dir" "$model_sha" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
model_sha = sys.argv[2]
cute = json.loads((root / "cute.json").read_text())
math_code = json.loads((root / "math-code.json").read_text())
wiki = json.loads((root / "wikitext.json").read_text())
summaries = {row["task"]: row for row in math_code["summaries"]}
cute_correct = cute["summaries"][-1]["correct"]
math_correct = summaries["math500"]["correct"]
code_correct = summaries["code_trace"]["correct"]
wiki_ratio = wiki["final_ppl"] / 5.7847
valid = (
    cute["model_sha256"] == model_sha
    and math_code["model_sha256"] == model_sha
    and wiki["model_sha256"] == model_sha
    and wiki["valid"] is True
)
passed = bool(
    valid and cute_correct >= 7 and math_correct >= 4
    and code_correct >= 4 and wiki_ratio <= 1.25
)
payload = {
    "schema": "qwen38_27b_llamacpp_ternary_quick_gate_v1",
    "terminal": True,
    "valid": valid,
    "thresholds": {
        "minimum_cute_correct": 7,
        "minimum_math500_correct": 4,
        "minimum_code_trace_correct": 4,
        "maximum_wikitext_ppl_ratio_vs_Q4": 1.25,
    },
    "observed": {
        "cute_correct": cute_correct,
        "math500_correct": math_correct,
        "code_trace_correct": code_correct,
        "wikitext_ppl": wiki["final_ppl"],
        "wikitext_ppl_ratio_vs_Q4": wiki_ratio,
    },
    "passed": passed,
}
temporary = root / "quick-gate.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "quick-gate.json")
PY

if "$python_bin" - "$state_dir/quick-gate.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("passed") is True else 1)
PY
then
  CPU_EVAL_RUN_DIR="$run_dir" \
  LLAMA_SERVER_BIN="$llama_build/bin/llama-server" \
  LLAMA_PARALLEL=1 LLAMA_CONTEXT=8192 LLAMA_THREADS=4 \
    "$runtime/launch_llama_cpu_eval_server.sh" "$model" "$label" "$port"
  if [[ ! -s "$state_dir/mmlu.json" ]]; then
    resume_args=()
    if [[ -s "$state_dir/mmlu.json.partial" ]]; then
      resume_args+=(--resume)
    fi
    PYTHONPATH="$repo" "$python_bin" "$runtime/eval_llama_cpp_mmlu.py" \
      --manifest "$mmlu_manifest" \
      --model "$model" \
      --model-label "$label" \
      --model-sha256 "$model_sha" \
      --server "http://127.0.0.1:${port}" \
      --workers 1 --timeout 900 --checkpoint-every 4 \
      --out "$state_dir/mmlu.json" "${resume_args[@]}" \
      > "$run_dir/mmlu.log" 2>&1
  fi
  cleanup
else
  "$python_bin" - "$state_dir" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
content = (root / "quick-gate.json").read_bytes()
payload = {
    "schema": "qwen38_27b_llamacpp_ternary_mmlu_skip_v1",
    "terminal": True,
    "valid_nonpromotion": True,
    "reason": "the frozen quick capability gate rejected",
    "quick_gate_sha256": hashlib.sha256(content).hexdigest(),
    "mmlu_launched": False,
}
temporary = root / "mmlu-skip.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "mmlu-skip.json")
PY
fi

if [[ ! -s "$state_dir/prefill-t4.json" ]]; then
  "$llama_build/bin/llama-bench" -m "$model" -p 512 -n 0 -t 4 \
    -b 512 -ub 512 -r 3 -ngl 0 --load-mode mmap -o json \
    > "$state_dir/prefill-t4.json.tmp" 2> "$run_dir/prefill-t4.log"
  mv "$state_dir/prefill-t4.json.tmp" "$state_dir/prefill-t4.json"
fi
if [[ ! -s "$state_dir/decode-t4.json" ]]; then
  "$llama_build/bin/llama-bench" -m "$model" -p 0 -n 32 -t 4 \
    -b 512 -ub 512 -r 3 -ngl 0 --load-mode mmap -o json \
    > "$state_dir/decode-t4.json.tmp" 2> "$run_dir/decode-t4.log"
  mv "$state_dir/decode-t4.json.tmp" "$state_dir/decode-t4.json"
fi

"$python_bin" - "$state_dir" "$format" "$model" "$model_sha" "$model_bytes" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
fmt, model, model_sha, model_bytes = sys.argv[2:]
cute = json.loads((root / "cute.json").read_text())
math_code = json.loads((root / "math-code.json").read_text())
wiki = json.loads((root / "wikitext.json").read_text())
quick = json.loads((root / "quick-gate.json").read_text())
prefill = json.loads((root / "prefill-t4.json").read_text())
decode = json.loads((root / "decode-t4.json").read_text())
assert prefill and decode
assert all(row["model_n_params"] == 26895998464 for row in prefill + decode)
assert all(row["model_filename"] == model for row in prefill + decode)
mmlu_path = root / "mmlu.json"
mmlu = json.loads(mmlu_path.read_text()) if mmlu_path.is_file() else None
if mmlu is not None:
    assert mmlu["valid"] and mmlu["status"] == "completed"
    assert mmlu["summary"]["n"] == 64
summary = {
    "schema": "qwen38_27b_llamacpp_ternary_control_summary_v1",
    "format": fmt,
    "logical_scope": "Qwen3.8-27B text graph; vision and MTP/NextN excluded",
    "logical_parameters": 26895998464,
    "model": model,
    "model_sha256": model_sha,
    "model_bytes": int(model_bytes),
    "effective_file_bits_per_26895998464_parameters": int(model_bytes) * 8 / 26895998464,
    "cute": cute["summaries"],
    "math_code": math_code["summaries"],
    "wikitext_ppl": wiki["final_ppl"],
    "quick_gate": quick,
    "mmlu": mmlu["summary"] if mmlu is not None else None,
    "mmlu_skipped_after_quick_rejection": mmlu is None,
    "descriptive_throughput": {
        "prefill_t4": prefill,
        "decode_t4": decode,
        "paired_or_authoritative": False,
    },
    "complete": True,
    "terminal": True,
    "claim_boundary": "whole-text-model llama.cpp control characterization; descriptive throughput is not a paired performance claim",
}
temporary = root / "summary.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "summary.json")
for name in ("cute.json", "math-code.json", "wikitext.json", "quick-gate.json", "prefill-t4.json", "decode-t4.json", "summary.json"):
    assert (root / name).is_file()
PY
sha256sum "$state_dir"/*.json > "$state_dir/artifacts.sha256"
printf 'complete\n' > "$state_dir/QUALITY_COMPLETE"
