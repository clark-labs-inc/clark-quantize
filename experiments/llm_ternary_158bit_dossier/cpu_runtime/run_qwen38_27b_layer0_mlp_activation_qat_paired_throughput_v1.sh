#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
baseline=${BASELINE_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
candidate=${CANDIDATE_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M-layer0-MLP-CQ3_0-activation-QAT-v1.gguf}
llama_bench=${LLAMA_BENCH:-/workspace/build/llama.cpp-qwen38-selective-v1/bin/llama-bench}
baseline_quality=${BASELINE_QUALITY:-/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1}
candidate_quality=${CANDIDATE_QUALITY:-/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-quality-v1}
materialization=${MATERIALIZATION_RECEIPT:-/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-materialization-v1.json}
out_dir=${OUT_DIR:-/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-paired-throughput-v1}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_layer0_mlp_activation_qat_paired_throughput_preregistration_v1.json"
summarizer="$runtime_dir/summarize_qwen38_27b_paired_throughput.py"
expected_baseline_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34

mkdir -p "$out_dir"
cd "$repo"
if [[ -e "$out_dir/THROUGHPUT_COMPLETE" ]]; then
  (cd / && sha256sum -c "$out_dir/artifacts.sha256")
  echo "trained Qwen3.8-27B paired throughput marker and hashes are valid"
  exit 0
fi
if [[ -n "$(git status --short)" ]]; then
  echo "trained Qwen3.8-27B paired throughput requires a clean checkout" >&2
  exit 2
fi
for marker in \
  "$baseline_quality/QUALITY_COMPLETE" \
  "$candidate_quality/QUALITY_COMPLETE" \
  "$candidate_quality/PROMOTION_PASS"; do
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

readarray -t pinned < <("$python_bin" - "$prereg" "$materialization" \
  "$baseline" "$candidate" "$llama_bench" <<'PY'
import hashlib, json, sys
from pathlib import Path
prereg_path, material_path, baseline, candidate, bench = map(Path, sys.argv[1:])
prereg = json.loads(prereg_path.read_text())
material = json.loads(material_path.read_text())
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()
baseline_sha, candidate_sha, bench_sha = sha256(baseline), sha256(candidate), sha256(bench)
if prereg["scope"]["active_model_only"] != "Qwen/Qwen3.8-27B":
    raise SystemExit("active model scope changed")
if baseline_sha != prereg["scope"]["baseline_sha256"]:
    raise SystemExit("baseline SHA-256 mismatch")
if not material.get("valid") or material["candidate"]["sha256"] != candidate_sha:
    raise SystemExit("candidate differs from materialization receipt")
print(candidate_sha)
print(bench_sha)
PY
)
candidate_sha=${pinned[0]}
bench_sha=${pinned[1]}
if [[ "$(sha256sum "$baseline" | awk '{print $1}')" != "$expected_baseline_sha" ]]; then
  echo "baseline SHA-256 mismatch" >&2
  exit 2
fi
(cd / && sha256sum -c "$baseline_quality/artifacts.sha256")
(cd / && sha256sum -c "$candidate_quality/artifacts.sha256")

git rev-parse HEAD > "$out_dir/source.commit"
lscpu --json > "$out_dir/lscpu.json"
free -b > "$out_dir/memory-before.txt"
sha256sum "$prereg" "$summarizer" "$baseline" "$candidate" "$llama_bench" \
  "$materialization" "$baseline_quality/summary.json" "$candidate_quality/summary.json" \
  > "$out_dir/inputs.sha256"

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
      -m "$model" -p 128,512,2048 -n 0 -t "$threads" -C "$mask" --cpu-strict 1 \
      -b 512 -ub 512 -r 5 --poll 50 -ngl 0 --load-mode mmap -o json \
      > "$prefill.tmp" 2> "$out_dir/${label}-prefill-t${threads}.log"
    mv "$prefill.tmp" "$prefill"
  fi
  if [[ ! -s "$decode" ]]; then
    taskset -c "$cpus" "$llama_bench" \
      -m "$model" -p 0 -n 128 -t "$threads" -C "$mask" --cpu-strict 1 \
      -b 512 -ub 512 -r 5 --poll 50 -ngl 0 --load-mode mmap -o json \
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

"$python_bin" "$summarizer" \
  --root "$out_dir" --baseline "$baseline" --candidate "$candidate" \
  --baseline-sha256 "$expected_baseline_sha" --candidate-sha256 "$candidate_sha" \
  --llama-bench "$llama_bench" --llama-bench-sha256 "$bench_sha" \
  --schema qwen38_27b_layer0_mlp_activation_qat_paired_throughput_summary_v1 \
  --claim-boundary 'paired performance of one activation-trained Qwen3.8-27B MLP triplet only' \
  --minimum-every-row 1.0 --minimum-geometric-mean 1.05

free -b > "$out_dir/memory-after.txt"
sha256sum "$out_dir"/baseline-prefill-t*.json "$out_dir"/baseline-decode-t*.json \
  "$out_dir"/candidate-prefill-t*.json "$out_dir"/candidate-decode-t*.json \
  "$out_dir/summary.json" > "$out_dir/artifacts.sha256"
printf 'complete\n' > "$out_dir/THROUGHPUT_COMPLETE"
if "$python_bin" - "$out_dir/summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["speed_promotion_pass"] else 1)
PY
then
  printf 'pass\n' > "$out_dir/SPEED_PROMOTION_PASS"
else
  printf 'rejected\n' > "$out_dir/SPEED_PROMOTION_REJECTED"
fi
