#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp}
llama_bench=${LLAMA_BENCH:-/workspace/build/llama.cpp/bin/llama-bench}
model=${MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
quality_dir=${QUALITY_DIR:-/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1}
out_dir=${OUT_DIR:-/workspace/quant-state/qwen38-27b-cpu-throughput-v1}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_cpu_throughput_preregistration_v1.json"
amendment="$runtime_dir/qwen38_27b_infrastructure_retry_amendment_v1.json"
model_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34

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

mkdir -p "$out_dir"
cd "$repo"
if [[ -e "$out_dir/THROUGHPUT_COMPLETE" ]]; then
  echo "throughput marker already exists: $out_dir/THROUGHPUT_COMPLETE"
  exit 0
fi
if [[ -n "$(git status --short)" ]]; then
  echo "27B throughput requires a clean source checkout" >&2
  exit 2
fi
if [[ ! -s "$quality_dir/QUALITY_COMPLETE" ]]; then
  echo "27B quality admission marker is absent" >&2
  exit 2
fi
llama_conflicts=$(ps -eo pid=,comm=,args= | awk \
  '$2 ~ /^llama-(server|cli|bench|perplexi)/ {print}')
if [[ -n "$llama_conflicts" ]]; then
  echo "another llama.cpp workload is active" >&2
  printf '%s\n' "$llama_conflicts" >&2
  exit 2
fi
training_conflicts=$(ps -eo pid=,comm=,args= | awk \
  '$2 ~ /^(python|python3|torchrun)$/ && $0 ~ /(train_mlp|qwen.*qat|activation.*cache)/ {print}')
if [[ -n "$training_conflicts" ]]; then
  echo "a QAT or cache workload is active" >&2
  printf '%s\n' "$training_conflicts" >&2
  exit 2
fi

require_sha256 "$prereg" 37d9321b0679cc7df5f7de7d5cbda82ef46b2152005472a756a4e9dcdf25aa11
require_sha256 "$amendment" 8adcdd2178e1d681465d8983913d71d3a47078933f83ce812dca751bb9687959
require_sha256 "$model" "$model_sha"
require_sha256 "$llama_bench" 01f85116456ffd6dbe4723801c86c2b75ed1b920a60416aef4f21d376114f7a9
sha256sum -c "$quality_dir/artifacts.sha256"

"$python_bin" - "$quality_dir" "$model_sha" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = sys.argv[2]
summary = json.loads((root / "summary.json").read_text())
assert summary["schema"] == "qwen38_27b_cpu_q4_baseline_summary_v1"
assert summary["model_sha256"] == expected
assert summary["all_receipts_valid"] is True
PY

git rev-parse HEAD > "$out_dir/source.commit"
git -C "$llama_source" rev-parse HEAD > "$out_dir/llama.commit"
if [[ "$(cat "$out_dir/llama.commit")" != 4df29be4f4c3673f428170fda944a5b19f743bb8 ]]; then
  echo "unexpected llama.cpp revision" >&2
  exit 2
fi
lscpu --json > "$out_dir/lscpu.json"
free -b > "$out_dir/memory-before.txt"
sha256sum "$prereg" "$amendment" "$model" "$llama_bench" "$quality_dir/summary.json" \
  > "$out_dir/inputs.sha256"

run_lane() {
  local threads=$1
  local cpus=$2
  local mask=$3
  local prefill="$out_dir/prefill-t${threads}.json"
  local decode="$out_dir/decode-t${threads}.json"

  if [[ ! -s "$prefill" ]]; then
    taskset -c "$cpus" "$llama_bench" \
      -m "$model" -p 128,512,2048 -n 0 \
      -t "$threads" -C "$mask" --cpu-strict 1 \
      -b 512 -ub 512 -r 5 --poll 50 -ngl 0 \
      --load-mode mmap -o json \
      > "$prefill.tmp" 2> "$out_dir/prefill-t${threads}.log"
    mv "$prefill.tmp" "$prefill"
  fi

  if [[ ! -s "$decode" ]]; then
    taskset -c "$cpus" "$llama_bench" \
      -m "$model" -p 0 -n 128 \
      -t "$threads" -C "$mask" --cpu-strict 1 \
      -b 512 -ub 512 -r 5 --poll 50 -ngl 0 \
      --load-mode mmap -o json \
      > "$decode.tmp" 2> "$out_dir/decode-t${threads}.log"
    mv "$decode.tmp" "$decode"
  fi
}

run_lane 1 0 0x1
run_lane 2 0,2 0x5
run_lane 4 0,2,4,6 0x55

"$python_bin" - "$out_dir" "$model" "$model_sha" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
model = str(Path(sys.argv[2]).resolve())
model_sha = sys.argv[3]
rows = []
for threads in (1, 2, 4):
    for mode in ("prefill", "decode"):
        path = root / f"{mode}-t{threads}.json"
        payload = json.loads(path.read_text())
        expected_rows = 3 if mode == "prefill" else 1
        if not isinstance(payload, list) or len(payload) != expected_rows:
            raise SystemExit(f"unexpected row count in {path}: {len(payload)}")
        for raw in payload:
            if raw["model_filename"] != model:
                raise SystemExit(f"model mismatch in {path}")
            if int(raw["n_threads"]) != threads or raw["cpu_strict"] is not True:
                raise SystemExit(f"CPU contract mismatch in {path}")
            samples = [float(value) for value in raw["samples_ts"]]
            if len(samples) != 5 or not all(math.isfinite(v) and v > 0 for v in samples):
                raise SystemExit(f"invalid timing samples in {path}")
            is_prefill = int(raw["n_prompt"]) in {128, 512, 2048} and int(raw["n_gen"]) == 0
            is_decode = int(raw["n_prompt"]) == 0 and int(raw["n_gen"]) == 128
            if (mode == "prefill" and not is_prefill) or (mode == "decode" and not is_decode):
                raise SystemExit(f"test-shape mismatch in {path}")
            rows.append(
                {
                    "mode": mode,
                    "threads": threads,
                    "cpu_mask": raw["cpu_mask"],
                    "prompt_tokens": int(raw["n_prompt"]),
                    "generated_tokens": int(raw["n_gen"]),
                    "mean_tokens_per_second": float(raw["avg_ts"]),
                    "stddev_tokens_per_second": float(raw["stddev_ts"]),
                    "median_tokens_per_second": statistics.median(samples),
                    "minimum_tokens_per_second": min(samples),
                    "maximum_tokens_per_second": max(samples),
                    "samples_tokens_per_second": samples,
                }
            )

if len(rows) != 12:
    raise SystemExit(f"expected 12 result rows, got {len(rows)}")
summary = {
    "schema": "qwen38_27b_cpu_throughput_summary_v1",
    "model": model,
    "model_sha256": model_sha,
    "quality_marker": str(Path("/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1/QUALITY_COMPLETE")),
    "rows": rows,
    "valid": True,
    "claim_boundary": "Qwen3.8-27B Q4_K_M llama-bench baseline; load, TTFT, energy, and candidate comparisons remain separate receipts",
}
temporary = root / "summary.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "summary.json")
PY

free -b > "$out_dir/memory-after.txt"
sha256sum "$out_dir"/prefill-t*.json "$out_dir"/decode-t*.json \
  "$out_dir/summary.json" > "$out_dir/artifacts.sha256"
printf 'complete\n' > "$out_dir/THROUGHPUT_COMPLETE"
