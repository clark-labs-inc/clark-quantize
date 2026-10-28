#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
model=${MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M-layer0-MLP-CQ3_0.gguf}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-qwen38-selective-v1}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-layer0-mlp-direct-cq3-quality-v1}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-quality-v1}
conversion_state=${CONVERSION_STATE:-/workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-v1}
baseline_state=${BASELINE_STATE:-/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1}
wiki_corpus=${WIKI_CORPUS:-/workspace/quant-runs/cq3-4b-ptq-scope-20260817/wikitext-2-raw/wiki.test.raw}
port=${PORT:-8095}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_layer0_mlp_direct_cq3_quality_preregistration_v1.json"
amendment="$runtime_dir/qwen38_27b_infrastructure_retry_amendment_v1.json"
cute_manifest="$runtime_dir/cute_quick_v1.json"
math_code_manifest="$runtime_dir/math_code_canary_v1.json"
mmlu_manifest="$runtime_dir/qwen35_4b_cpu_layer30_mmlu_v1.json"
label=qwen38-27b-q4-layer0-mlp-cq3-v1
server_unit="llama-cpu-eval-${label}.service"
expected_model_bytes=18894489952

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
  (cd / && sha256sum -c "$state_dir/artifacts.sha256")
  echo "candidate quality marker and artifact hashes are valid"
  exit 0
fi
if [[ -n "$(git status --short)" ]]; then
  echo "27B candidate quality run requires a clean research checkout" >&2
  exit 2
fi
if [[ ! -s "$conversion_state/CANDIDATE_READY" || ! -s "$conversion_state/conversion-receipt.json" ]]; then
  echo "the byte-scope-proven 27B candidate is not ready" >&2
  exit 3
fi
if [[ ! -s "$baseline_state/QUALITY_COMPLETE" || ! -s "$baseline_state/summary.json" ]]; then
  echo "the pinned 27B Q4 quality baseline is not complete" >&2
  exit 3
fi
if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a frozen Qwen 4B service is active; refusing to share the node" >&2
  exit 2
fi

require_sha256 "$prereg" 0549ff2ea7ff20a63e5d37403b7556f93ebe1d45f890da8c6bf9ea100f31b899
require_sha256 "$amendment" 8adcdd2178e1d681465d8983913d71d3a47078933f83ce812dca751bb9687959
require_sha256 "$cute_manifest" 19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
require_sha256 "$math_code_manifest" 1ec0b41ce3ca3597d65013ac0c8a48c208c4e03b05f0109be61319252c09dab0
require_sha256 "$mmlu_manifest" e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba
require_sha256 "$wiki_corpus" 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
require_sha256 "$runtime_dir/eval_llama_cpp_cute_quick.py" 9c40a48a0c3f0544156a20ccccdc12f7e90bbae8afd391bf640ead1ff234c8ab
require_sha256 "$runtime_dir/eval_llama_cpp_math_code_canary.py" fa3d0f870b3b0ed1d7d907cc8d146854b164e6d75cfa9d2bd20743979af74677
require_sha256 "$runtime_dir/eval_llama_cpp_mmlu.py" a604ca9a528231fa17de78ab2a7ffed7c1b9a0b9a9ca74459c2d894f1f3612d3
require_sha256 "$runtime_dir/eval_llama_cpp_wikitext_quick.py" 54458dc377a1d07aaf26df90823f65f973b625f55a6ac73895d0c6ec9a666b07
require_sha256 "$runtime_dir/launch_llama_cpu_eval_server.sh" c972d6528779bf7630fb39a48d248239788a657c2f8530e6ff3a9e9cd2f19b57
if [[ "$(stat -c %s "$model")" != "$expected_model_bytes" ]]; then
  echo "unexpected candidate byte count" >&2
  exit 2
fi

model_sha=$("$python_bin" - "$conversion_state/conversion-receipt.json" \
  "$conversion_state/gguf-scope.json" "$model" "$llama_build" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

conversion_path, scope_path, model, build = map(Path, sys.argv[1:])
conversion = json.loads(conversion_path.read_text())
scope = json.loads(scope_path.read_text())
if not conversion.get("valid") or not scope.get("valid"):
    raise SystemExit("candidate conversion or scope receipt is invalid")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

actual = sha256(model)
if conversion["candidate"]["sha256"] != actual:
    raise SystemExit("candidate SHA-256 differs from conversion receipt")
if scope["models"]["candidate"]["sha256"] != actual:
    raise SystemExit("candidate SHA-256 differs from scope receipt")
for name in ("llama-server", "llama-perplexity"):
    if conversion["binaries"][name] != sha256(build / "bin" / name):
        raise SystemExit(f"{name} differs from conversion receipt")
print(actual)
PY
)

git rev-parse HEAD > "$state_dir/source.commit"
sha256sum "$prereg" "$amendment" "$cute_manifest" "$math_code_manifest" "$mmlu_manifest" \
  "$wiki_corpus" "$runtime_dir/eval_llama_cpp_cute_quick.py" \
  "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
  "$runtime_dir/eval_llama_cpp_mmlu.py" \
  "$runtime_dir/eval_llama_cpp_wikitext_quick.py" \
  "$llama_build/bin/llama-server" "$llama_build/bin/llama-perplexity" \
  "$model" > "$state_dir/inputs.sha256"

CPU_EVAL_RUN_DIR="$run_dir" \
LLAMA_SERVER_BIN="$llama_build/bin/llama-server" \
LLAMA_PARALLEL=1 \
LLAMA_CONTEXT=8192 \
LLAMA_THREADS=4 \
  "$runtime_dir/launch_llama_cpu_eval_server.sh" "$model" "$label" "$port"

if [[ ! -s "$state_dir/cute.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_cute_quick.py" \
    --manifest "$cute_manifest" \
    --server "http://127.0.0.1:${port}" \
    --model-label "$label" \
    --model-sha256 "$model_sha" \
    --out "$state_dir/cute.json" \
    > "$run_dir/cute.log" 2>&1
fi

if [[ ! -s "$state_dir/math-code.json" ]]; then
  "$python_bin" "$runtime_dir/eval_llama_cpp_math_code_canary.py" \
    --manifest "$math_code_manifest" \
    --server "http://127.0.0.1:${port}" \
    --model-label "$label" \
    --model-sha256 "$model_sha" \
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
    --model-sha256 "$model_sha" \
    --server "http://127.0.0.1:${port}" \
    --workers 1 \
    --timeout 900 \
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

"$python_bin" - "$prereg" "$baseline_state/summary.json" "$state_dir" "$model_sha" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

prereg, baseline_path, root = map(Path, sys.argv[1:4])
model_sha = sys.argv[4]
baseline = json.loads(baseline_path.read_text())
cute = json.loads((root / "cute.json").read_text())
math_code = json.loads((root / "math-code.json").read_text())
mmlu = json.loads((root / "mmlu.json").read_text())
wiki = json.loads((root / "wikitext.json").read_text())

def summary_by_task(rows):
    return {row["task"]: row for row in rows}

baseline_cute = summary_by_task(baseline["cute"])["cute_quick_20"]["correct"]
baseline_math_code = summary_by_task(baseline["math_code"])
candidate_cute = summary_by_task(cute["summaries"])["cute_quick_20"]["correct"]
candidate_math_code = summary_by_task(math_code["summaries"])
baseline_mmlu = int(baseline["mmlu"]["correct"])
candidate_mmlu = int(mmlu["summary"]["correct"])
baseline_ppl = float(baseline["wikitext_ppl"])
candidate_ppl = float(wiki["final_ppl"])

receipts_valid = (
    cute["model_sha256"] == model_sha
    and len(cute["records"]) == 20
    and math_code["model_sha256"] == model_sha
    and len(math_code["records"]) == 14
    and mmlu["valid"]
    and mmlu["status"] == "completed"
    and mmlu["summary"]["n"] == 64
    and mmlu["contract"]["model_sha256"] == model_sha
    and wiki["valid"]
    and wiki["model_sha256"] == model_sha
)
thresholds = {
    "cute_correct": max(0, int(baseline_cute) - 2),
    "math500_correct": max(0, int(baseline_math_code["math500"]["correct"]) - 1),
    "code_trace_correct": max(0, int(baseline_math_code["code_trace"]["correct"]) - 1),
    "mmlu_correct": max(baseline_mmlu - 3, math.ceil(0.90 * baseline_mmlu)),
    "wikitext_ppl_ratio_max": 1.05,
}
observed = {
    "cute_correct": int(candidate_cute),
    "math500_correct": int(candidate_math_code["math500"]["correct"]),
    "code_trace_correct": int(candidate_math_code["code_trace"]["correct"]),
    "mmlu_correct": candidate_mmlu,
    "wikitext_ppl": candidate_ppl,
    "wikitext_ppl_ratio": candidate_ppl / baseline_ppl,
}
checks = {
    "all_receipts_valid": receipts_valid,
    "cute": observed["cute_correct"] >= thresholds["cute_correct"],
    "math500": observed["math500_correct"] >= thresholds["math500_correct"],
    "code_trace": observed["code_trace_correct"] >= thresholds["code_trace_correct"],
    "mmlu": observed["mmlu_correct"] >= thresholds["mmlu_correct"],
    "wikitext": observed["wikitext_ppl_ratio"] <= thresholds["wikitext_ppl_ratio_max"],
}
result = {
    "schema": "qwen38_27b_layer0_mlp_direct_cq3_quality_summary_v1",
    "preregistration": {
        "path": str(prereg),
        "sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
    },
    "candidate_model_sha256": model_sha,
    "baseline": {
        "summary_path": str(baseline_path),
        "cute_correct": baseline_cute,
        "math500_correct": baseline_math_code["math500"]["correct"],
        "code_trace_correct": baseline_math_code["code_trace"]["correct"],
        "mmlu_correct": baseline_mmlu,
        "wikitext_ppl": baseline_ppl,
    },
    "thresholds": thresholds,
    "observed": observed,
    "checks": checks,
    "promotion_pass": all(checks.values()),
    "claim_boundary": "one direct Q4-to-CQ3 Qwen3.8-27B layer-0 MLP triplet only",
}
temporary = root / "summary.json.tmp"
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "summary.json")
if not receipts_valid:
    raise SystemExit("one or more quality receipts are invalid")
PY

sha256sum "$state_dir/cute.json" "$state_dir/math-code.json" \
  "$state_dir/mmlu.json" "$state_dir/wikitext.json" "$state_dir/summary.json" \
  > "$state_dir/artifacts.sha256"
printf 'complete\n' > "$state_dir/QUALITY_COMPLETE"
if "$python_bin" - "$state_dir/summary.json" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["promotion_pass"] else 1)
PY
then
  printf 'pass\n' > "$state_dir/PROMOTION_PASS"
else
  printf 'rejected\n' > "$state_dir/PROMOTION_REJECTED"
fi
