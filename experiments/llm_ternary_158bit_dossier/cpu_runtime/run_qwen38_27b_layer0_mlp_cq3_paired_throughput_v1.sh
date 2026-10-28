#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
baseline=${BASELINE_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
candidate=${CANDIDATE_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M-layer0-MLP-CQ3_0.gguf}
llama_bench=${LLAMA_BENCH:-/workspace/build/llama.cpp-qwen38-selective-v1/bin/llama-bench}
conversion_state=${CONVERSION_STATE:-/workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-v1}
baseline_quality=${BASELINE_QUALITY:-/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1}
candidate_quality=${CANDIDATE_QUALITY:-/workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-quality-v1}
out_dir=${OUT_DIR:-/workspace/quant-state/qwen38-27b-layer0-mlp-cq3-paired-throughput-v1}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_layer0_mlp_cq3_paired_throughput_preregistration_v1.json"
amendment="$runtime_dir/qwen38_27b_infrastructure_retry_amendment_v1.json"
expected_baseline_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34

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
  (cd / && sha256sum -c "$out_dir/artifacts.sha256")
  echo "paired throughput marker and artifact hashes are valid"
  exit 0
fi
if [[ -n "$(git status --short)" ]]; then
  echo "27B paired throughput requires a clean research checkout" >&2
  exit 2
fi
for marker in \
  "$baseline_quality/QUALITY_COMPLETE" \
  "$candidate_quality/QUALITY_COMPLETE" \
  "$candidate_quality/PROMOTION_PASS" \
  "$conversion_state/CANDIDATE_READY"; do
  if [[ ! -s "$marker" ]]; then
    echo "required admission marker is absent: $marker" >&2
    exit 3
  fi
done
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
  echo "a QAT or activation-cache workload is active" >&2
  printf '%s\n' "$training_conflicts" >&2
  exit 2
fi

require_sha256 "$prereg" 917b109201eecec4bbabd06bae99ad449ecb127f4a4f1221f850c7d04d0d1a58
require_sha256 "$amendment" 8adcdd2178e1d681465d8983913d71d3a47078933f83ce812dca751bb9687959
require_sha256 "$baseline" "$expected_baseline_sha"
(cd / && sha256sum -c "$conversion_state/artifacts.sha256")
(cd / && sha256sum -c "$baseline_quality/artifacts.sha256")
(cd / && sha256sum -c "$candidate_quality/artifacts.sha256")

readarray -t pinned < <("$python_bin" - "$conversion_state/conversion-receipt.json" \
  "$conversion_state/gguf-scope.json" "$candidate" "$llama_bench" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

conversion_path, scope_path, candidate, llama_bench = map(Path, sys.argv[1:])
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

candidate_sha = sha256(candidate)
bench_sha = sha256(llama_bench)
if candidate_sha != conversion["candidate"]["sha256"]:
    raise SystemExit("candidate SHA-256 differs from conversion receipt")
if candidate_sha != scope["models"]["candidate"]["sha256"]:
    raise SystemExit("candidate SHA-256 differs from scope receipt")
if bench_sha != conversion["binaries"]["llama-bench"]:
    raise SystemExit("llama-bench differs from conversion receipt")
print(candidate_sha)
print(bench_sha)
PY
)
candidate_sha=${pinned[0]}
bench_sha=${pinned[1]}

git rev-parse HEAD > "$out_dir/source.commit"
lscpu --json > "$out_dir/lscpu.json"
free -b > "$out_dir/memory-before.txt"
sha256sum "$prereg" "$amendment" "$baseline" "$candidate" "$llama_bench" \
  "$conversion_state/conversion-receipt.json" "$baseline_quality/summary.json" \
  "$candidate_quality/summary.json" > "$out_dir/inputs.sha256"

run_lane() {
  local label=$1
  local model=$2
  local threads=$3
  local cpus=$4
  local mask=$5
  local prefill="$out_dir/${label}-prefill-t${threads}.json"
  local decode="$out_dir/${label}-decode-t${threads}.json"

  if [[ ! -s "$prefill" ]]; then
    taskset -c "$cpus" "$llama_bench" \
      -m "$model" -p 128,512,2048 -n 0 \
      -t "$threads" -C "$mask" --cpu-strict 1 \
      -b 512 -ub 512 -r 5 --poll 50 -ngl 0 \
      --load-mode mmap -o json \
      > "$prefill.tmp" 2> "$out_dir/${label}-prefill-t${threads}.log"
    mv "$prefill.tmp" "$prefill"
  fi

  if [[ ! -s "$decode" ]]; then
    taskset -c "$cpus" "$llama_bench" \
      -m "$model" -p 0 -n 128 \
      -t "$threads" -C "$mask" --cpu-strict 1 \
      -b 512 -ub 512 -r 5 --poll 50 -ngl 0 \
      --load-mode mmap -o json \
      > "$decode.tmp" 2> "$out_dir/${label}-decode-t${threads}.log"
    mv "$decode.tmp" "$decode"
  fi
}

run_lane baseline "$baseline" 1 0 0x1
run_lane candidate "$candidate" 1 0 0x1
run_lane candidate "$candidate" 2 0,2 0x5
run_lane baseline "$baseline" 2 0,2 0x5
run_lane baseline "$baseline" 4 0,2,4,6 0x55
run_lane candidate "$candidate" 4 0,2,4,6 0x55

"$python_bin" - "$out_dir" "$baseline" "$candidate" \
  "$expected_baseline_sha" "$candidate_sha" "$llama_bench" "$bench_sha" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = {
    "baseline": {"path": str(Path(sys.argv[2]).resolve()), "sha256": sys.argv[4]},
    "candidate": {"path": str(Path(sys.argv[3]).resolve()), "sha256": sys.argv[5]},
}
llama_bench = str(Path(sys.argv[6]).resolve())
bench_sha = sys.argv[7]
rows = {"baseline": {}, "candidate": {}}
serialized = []
for label in ("baseline", "candidate"):
    for threads in (1, 2, 4):
        for mode in ("prefill", "decode"):
            path = root / f"{label}-{mode}-t{threads}.json"
            payload = json.loads(path.read_text())
            expected_rows = 3 if mode == "prefill" else 1
            if not isinstance(payload, list) or len(payload) != expected_rows:
                raise SystemExit(f"unexpected row count in {path}")
            for raw in payload:
                if raw["model_filename"] != models[label]["path"]:
                    raise SystemExit(f"model mismatch in {path}")
                if int(raw["n_threads"]) != threads or raw["cpu_strict"] is not True:
                    raise SystemExit(f"CPU contract mismatch in {path}")
                samples = [float(value) for value in raw["samples_ts"]]
                if len(samples) != 5 or not all(math.isfinite(v) and v > 0 for v in samples):
                    raise SystemExit(f"invalid timing samples in {path}")
                prompt = int(raw["n_prompt"])
                generated = int(raw["n_gen"])
                if mode == "prefill" and not (prompt in {128, 512, 2048} and generated == 0):
                    raise SystemExit(f"prefill shape mismatch in {path}")
                if mode == "decode" and not (prompt == 0 and generated == 128):
                    raise SystemExit(f"decode shape mismatch in {path}")
                key = (mode, threads, prompt, generated)
                row = {
                    "model": label,
                    "mode": mode,
                    "threads": threads,
                    "prompt_tokens": prompt,
                    "generated_tokens": generated,
                    "mean_tokens_per_second": float(raw["avg_ts"]),
                    "stddev_tokens_per_second": float(raw["stddev_ts"]),
                    "median_tokens_per_second": statistics.median(samples),
                    "minimum_tokens_per_second": min(samples),
                    "maximum_tokens_per_second": max(samples),
                    "samples_tokens_per_second": samples,
                }
                rows[label][key] = row
                serialized.append(row)

if len(rows["baseline"]) != 12 or set(rows["baseline"]) != set(rows["candidate"]):
    raise SystemExit("paired row keys are incomplete or differ")
comparisons = []
for key in sorted(rows["baseline"]):
    baseline_row = rows["baseline"][key]
    candidate_row = rows["candidate"][key]
    speedup = candidate_row["median_tokens_per_second"] / baseline_row["median_tokens_per_second"]
    comparisons.append(
        {
            "mode": key[0],
            "threads": key[1],
            "prompt_tokens": key[2],
            "generated_tokens": key[3],
            "baseline_median_tokens_per_second": baseline_row["median_tokens_per_second"],
            "candidate_median_tokens_per_second": candidate_row["median_tokens_per_second"],
            "candidate_over_baseline_speedup": speedup,
        }
    )
speedups = [row["candidate_over_baseline_speedup"] for row in comparisons]
geometric_mean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
checks = {
    "row_count": len(comparisons) == 12,
    "no_row_below_baseline": min(speedups) >= 1.0,
    "geometric_mean_at_least_1_05": geometric_mean >= 1.05,
}
summary = {
    "schema": "qwen38_27b_layer0_mlp_cq3_paired_throughput_summary_v1",
    "models": models,
    "llama_bench": {"path": llama_bench, "sha256": bench_sha},
    "rows": serialized,
    "comparisons": comparisons,
    "minimum_speedup": min(speedups),
    "geometric_mean_speedup": geometric_mean,
    "checks": checks,
    "valid": True,
    "speed_promotion_pass": all(checks.values()),
    "claim_boundary": "paired performance of one selectively ternarized Qwen3.8-27B MLP triplet only",
}
temporary = root / "summary.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "summary.json")
PY

free -b > "$out_dir/memory-after.txt"
sha256sum "$out_dir"/baseline-prefill-t*.json "$out_dir"/baseline-decode-t*.json \
  "$out_dir"/candidate-prefill-t*.json "$out_dir"/candidate-decode-t*.json \
  "$out_dir/summary.json" > "$out_dir/artifacts.sha256"
printf 'complete\n' > "$out_dir/THROUGHPUT_COMPLETE"
if "$python_bin" - "$out_dir/summary.json" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["speed_promotion_pass"] else 1)
PY
then
  printf 'pass\n' > "$out_dir/SPEED_PROMOTION_PASS"
else
  printf 'rejected\n' > "$out_dir/SPEED_PROMOTION_REJECTED"
fi
