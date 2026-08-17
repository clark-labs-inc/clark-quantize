#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
hf_bin=${HF_BIN:-/workspace/venvs/quant-cpu/bin/hf}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp}
model=${MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-cpu-q4-baseline-v1}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1}
wiki_corpus=${WIKI_CORPUS:-/workspace/quant-runs/cq3-4b-ptq-scope-20260817/wikitext-2-raw/wiki.test.raw}
port=${PORT:-8094}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_cpu_q4_baseline_preregistration_v1.json"
cute_manifest="$runtime_dir/cute_quick_v1.json"
math_code_manifest="$runtime_dir/math_code_canary_v1.json"
mmlu_manifest="$runtime_dir/qwen35_4b_cpu_layer30_mmlu_v1.json"
label=qwen38-27b-q4-k-m-v1
expected_model_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
expected_model_bytes=18973870432
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

mkdir -p "$run_dir" "$state_dir" "$(dirname "$model")"
cd "$repo"
if [[ -e "$state_dir/QUALITY_COMPLETE" ]]; then
  echo "quality marker already exists: $state_dir/QUALITY_COMPLETE"
  exit 0
fi
if [[ -n "$(git status --short)" ]]; then
  echo "27B baseline requires a clean source checkout" >&2
  exit 2
fi
if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a frozen Qwen 4B service is active; refusing to share the node" >&2
  exit 2
fi

require_sha256 "$prereg" 66ab9467b366b43afe1eb373a721f1d39578a486ace5dc53e3c7131ae29dc919
require_sha256 "$cute_manifest" 19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
require_sha256 "$math_code_manifest" 1ec0b41ce3ca3597d65013ac0c8a48c208c4e03b05f0109be61319252c09dab0
require_sha256 "$mmlu_manifest" e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba
require_sha256 "$wiki_corpus" 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
require_sha256 "$llama_build/bin/llama-server" c1c5af73afc267dfd9256e8f97bd871e7842353dd25518587f68261d731a24c9
require_sha256 "$llama_build/bin/llama-perplexity" 9941787d0c8d35ed67dce640dc3c3a0cc55750f672d31dcb1e6dcd641fd15f78

"$python_bin" - <<'PY'
import json
from pathlib import Path

receipt = json.loads(Path("/workspace/models/Qwen3.8-27B/CLARK_DOWNLOAD_RECEIPT.json").read_text())
assert receipt == {
    "bytes": 55563006776,
    "index_sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
    "model_id": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "schema": "qwen38_model_download_v1",
    "shards": 18,
}
PY

git rev-parse HEAD > "$state_dir/source.commit"
git -C /workspace/src/llama.cpp rev-parse HEAD > "$state_dir/llama.commit"
if [[ "$(cat "$state_dir/llama.commit")" != 4df29be4f4c3673f428170fda944a5b19f743bb8 ]]; then
  echo "unexpected llama.cpp source revision" >&2
  exit 2
fi

if [[ ! -s "$model" ]]; then
  "$hf_bin" download ggml-org/Qwen3.8-27B-GGUF \
    Qwen3.8-27B-Q4_K_M.gguf \
    --revision 0669b98607d47046c7c2b3f801011d54a08cfccf \
    --local-dir "$(dirname "$model")" \
    > "$run_dir/download.log" 2>&1
fi
if [[ "$(stat -c %s "$model")" != "$expected_model_bytes" ]]; then
  echo "unexpected 27B Q4 byte count" >&2
  exit 2
fi
require_sha256 "$model" "$expected_model_sha"
printf 'complete\n' > "$state_dir/DOWNLOAD_COMPLETE"

sha256sum "$prereg" "$cute_manifest" "$math_code_manifest" "$mmlu_manifest" \
  "$wiki_corpus" "$model" "$llama_build/bin/llama-server" \
  "$llama_build/bin/llama-perplexity" > "$state_dir/inputs.sha256"

if [[ ! -s "$state_dir/load-smoke.log" ]]; then
  /usr/bin/time -v "$llama_build/bin/llama-cli" \
    -m "$model" -t 4 -tb 4 -c 2048 -b 512 -ub 512 \
    --no-warmup -p 'Reply with only OK.' -n 2 --temp 0 \
    > "$state_dir/load-smoke.log" 2>&1
fi

CPU_EVAL_RUN_DIR="$run_dir" \
LLAMA_PARALLEL=1 \
LLAMA_CONTEXT=32768 \
LLAMA_THREADS=4 \
  "$runtime_dir/launch_llama_cpu_eval_server.sh" "$model" "$label" "$port"

if [[ ! -s "$state_dir/cute.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_cute_quick.py" \
    --manifest "$cute_manifest" \
    --server "http://127.0.0.1:${port}" \
    --model-label "$label" \
    --model-sha256 "$expected_model_sha" \
    --out "$state_dir/cute.json" \
    > "$run_dir/cute.log" 2>&1
fi

if [[ ! -s "$state_dir/math-code.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
    --manifest "$math_code_manifest" \
    --server "http://127.0.0.1:${port}" \
    --model-label "$label" \
    --model-sha256 "$expected_model_sha" \
    --out "$state_dir/math-code.json" \
    > "$run_dir/math-code.log" 2>&1
fi

if [[ ! -s "$state_dir/mmlu.json" ]]; then
  resume_args=()
  if [[ -s "$state_dir/mmlu.json.partial" ]]; then
    resume_args+=(--resume)
  fi
  PYTHONPATH="$repo" "$python_bin" "$runtime_dir/eval_llama_cpp_mmlu.py" \
    --manifest "$mmlu_manifest" \
    --model "$model" \
    --model-label "$label" \
    --model-sha256 "$expected_model_sha" \
    --server "http://127.0.0.1:${port}" \
    --workers 1 \
    --checkpoint-every 4 \
    --out "$state_dir/mmlu.json" \
    "${resume_args[@]}" \
    > "$run_dir/mmlu.log" 2>&1
fi

cleanup

if [[ ! -s "$state_dir/wikitext.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_wikitext_quick.py" \
    --model "$model" \
    --model-label "$label" \
    --corpus "$wiki_corpus" \
    --llama-perplexity "$llama_build/bin/llama-perplexity" \
    --chunks 4 --context 512 --threads 4 \
    --out "$state_dir/wikitext.json" \
    > "$run_dir/wikitext.log" 2>&1
fi

"$python_bin" - "$state_dir" "$expected_model_sha" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
model_sha = sys.argv[2]
cute = json.loads((root / "cute.json").read_text())
math_code = json.loads((root / "math-code.json").read_text())
mmlu = json.loads((root / "mmlu.json").read_text())
wiki = json.loads((root / "wikitext.json").read_text())
assert cute["schema"] == "cpu_cute_quick_result_v1"
assert cute["model_sha256"] == model_sha and len(cute["records"]) == 20
assert cute["summaries"][-1]["task"] == "cute_quick_20"
assert cute["summaries"][-1]["n"] == 20
assert math_code["schema"] == "cpu_math_code_canary_result_v1"
assert math_code["model_sha256"] == model_sha and len(math_code["records"]) == 14
assert {row["task"]: row["n"] for row in math_code["summaries"]} == {
    "math500": 8,
    "code_trace": 6,
}
assert mmlu["valid"] and mmlu["status"] == "completed"
assert mmlu["summary"]["n"] == 64 and mmlu["contract"]["model_sha256"] == model_sha
assert wiki["valid"] and wiki["model_sha256"] == model_sha
summary = {
    "schema": "qwen38_27b_cpu_q4_baseline_summary_v1",
    "model_sha256": model_sha,
    "cute": cute["summaries"],
    "math_code": math_code["summaries"],
    "mmlu": mmlu["summary"],
    "wikitext_ppl": wiki["final_ppl"],
    "all_receipts_valid": True,
}
temporary = root / "summary.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "summary.json")
PY

sha256sum "$state_dir/cute.json" "$state_dir/math-code.json" \
  "$state_dir/mmlu.json" "$state_dir/wikitext.json" "$state_dir/summary.json" \
  > "$state_dir/artifacts.sha256"
printf 'complete\n' > "$state_dir/QUALITY_COMPLETE"
