#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-row-scale-v3}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
server=${LLAMA_SERVER:-/workspace/build/llama.cpp/bin/llama-server}
target=${Q3_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q3_K_M.gguf}
draft=${MTP_MODEL:-/workspace/models/controls/mtp-Qwen3.8-27B-Q4_0.gguf}
baseline_state=${Q3_BASELINE_STATE:-/workspace/quant-state/qwen38-27b-q3-k-m-baseline-v1}
output_dir=${OUTPUT_DIR:-/workspace/quant-runs/qwen38-27b-q3-mtp-admission-v1}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-q3-mtp-admission-v1}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime_dir/run_qwen38_27b_mtp_parity_v1.py"
mtp_sha=051a1764cff8c4f3ee6ae8b00593a0364c7539c67fa50ffc58f3f96509fca38e
server_sha=c1c5af73afc267dfd9256e8f97bd871e7842353dd25518587f68261d731a24c9

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

mkdir -p "$output_dir" "$state_dir"
cd "$repo"
if [[ -e "$state_dir/ADMISSION_TERMINAL" ]]; then
  echo "Q3 plus MTP admission is already terminal"
  exit 0
fi
if [[ -n "$(git status --short)" ]]; then
  echo "Q3 plus MTP admission requires a clean source checkout" >&2
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
if [[ ! -s "$baseline_state/BASELINE_COMPLETE" || ! -s "$baseline_state/summary.json" ]]; then
  echo "Q3 baseline is not terminal" >&2
  exit 2
fi
if ! "$python_bin" - "$baseline_state/summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["pass"] else 1)
PY
then
  echo "Q3 baseline did not pass its frozen admission gate" >&2
  exit 2
fi
expected_target_sha=$("$python_bin" - "$baseline_state/materialization.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["model_sha256"])
PY
)
require_sha256 "$target" "$expected_target_sha"
require_sha256 "$draft" "$mtp_sha"
require_sha256 "$server" "$server_sha"
git rev-parse HEAD > "$state_dir/source.commit"

set +e
"$python_bin" "$runner" \
  --server "$server" \
  --target "$target" \
  --draft "$draft" \
  --output-dir "$output_dir" \
  --widths 0,3,4,6,8 \
  --n-predict 64 \
  --threads 4 \
  --cpu-list 0,2,4,6 \
  --context 512 \
  --batch 128 \
  --ubatch 64 \
  --port-base 18200 \
  --draft-p-min 0.0 \
  --server-verbosity 5 \
  --trace-speculative \
  > "$state_dir/runner.stdout" 2> "$state_dir/runner.stderr"
runner_status=$?
set -e
if [[ "$runner_status" -ne 0 && "$runner_status" -ne 2 ]]; then
  echo "MTP runner failed before a scientific terminal state: $runner_status" >&2
  exit "$runner_status"
fi
if [[ ! -s "$output_dir/summary.json" ]]; then
  echo "MTP runner produced no terminal summary" >&2
  exit 2
fi

"$python_bin" - "$output_dir/summary.json" "$state_dir/summary.json" "$expected_target_sha" <<'PY'
import json
import sys
from pathlib import Path

source_path, output_path, expected_target_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
source = json.loads(source_path.read_text())
assert source["target"]["sha256"] == expected_target_sha
rows = source["configurations"]
target = next(row for row in rows if row["configuration"] == "target")
speculative = [row for row in rows if row["configuration"] != "target"]
best = max(speculative, key=lambda row: row["accepted_tokens_per_second_weighted"])
all_parity = all(row["exact_token_parity"] for row in rows)
checks = {
    "all_widths_exact_greedy_parity": all_parity,
    "best_accepted_tps_at_least_2_75": best["accepted_tokens_per_second_weighted"] >= 2.75,
    "best_speedup_vs_target_at_least_1_40": best["accepted_decode_speedup_vs_target"] >= 1.40,
    "best_mean_committed_per_target_call_at_least_3": best["mean_committed_tokens_per_target_call"] >= 3.0,
    "best_combined_weight_bytes_per_accepted_token_at_most_7e9": best["combined_weight_bytes_per_accepted_token_lower_bound"] <= 7_000_000_000,
}
payload = {
    "schema": "qwen38_27b_q3_mtp_admission_summary_v1",
    "terminal": True,
    "pass": all(checks.values()),
    "checks": checks,
    "target": target,
    "best_speculative": best,
    "all_configurations": rows,
    "source_summary_sha256": __import__("hashlib").sha256(source_path.read_bytes()).hexdigest(),
    "claim_boundary": "Four-prompt development admission only; not heldout, final-node, 100 tok/s, capability, or novelty evidence.",
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output_path)
PY
sha256sum "$target" "$draft" "$server" "$output_dir/summary.json" "$state_dir/summary.json" \
  > "$state_dir/artifacts.sha256"
printf 'terminal\n' > "$state_dir/ADMISSION_TERMINAL"
