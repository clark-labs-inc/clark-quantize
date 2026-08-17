#!/usr/bin/env bash
set -euo pipefail

runtime_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runtime_dir/../../../.." && pwd)
prereg="$runtime_dir/qwen38_27b_cq3_x86_pair_preregistration_v1.json"
base_patch="$runtime_dir/llama_cpp_4df29be_cq3.patch"
candidate_patch="$runtime_dir/llama_cpp_4df29be_cq3_x86_pair_v1.patch"
benchmark_source="$runtime_dir/qwen38_27b_cq3_x86_pair_bench_v1.inc"
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-cq3-x86-pair-v1}
source_seed=${LLAMA_SOURCE_SEED:-/workspace/src/llama.cpp}
source_dir=${LLAMA_SOURCE_DIR:-/workspace/src/llama.cpp-qwen38-27b-cq3-x86-pair-v1}
build_dir=${LLAMA_BUILD_DIR:-/workspace/build/qwen38-27b-cq3-x86-pair-v1}
upstream_revision=4df29be4f4c3673f428170fda944a5b19f743bb8

if [[ -s "$state_dir/BENCHMARK_COMPLETE" ]]; then
  echo "benchmark already complete: $state_dir/BENCHMARK_COMPLETE"
  exit 0
fi

for unit in \
  quant-qwen38-27b-cpu-q4-baseline-v1.service \
  quant-qwen38-27b-postbaseline-chain-v1.service \
  quant-qwen38-27b-paired-throughput-watch-v1.service; do
  if systemctl is-active --quiet "$unit"; then
    echo "refusing to compete with active Qwen3.8-27B service: $unit" >&2
    exit 75
  fi
done

available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
free_disk_kib=$(df -Pk /workspace | awk 'NR == 2 {print $4}')
if (( available_kib < 8*1024*1024 )); then
  echo "infrastructure-invalid: less than 8 GiB available memory" >&2
  exit 75
fi
if (( free_disk_kib < 8*1024*1024 )); then
  echo "infrastructure-invalid: less than 8 GiB free disk" >&2
  exit 75
fi
if [[ -n "$(swapon --noheadings --show 2>/dev/null)" ]]; then
  echo "infrastructure-invalid: swap is active" >&2
  exit 75
fi

check_sha256() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 mismatch for $path: expected $expected got $actual" >&2
    exit 65
  fi
}

check_sha256 b5c584a338018ff7e509bc160c195693102fb6e265b6cbe4935d6e54d7cd67c1 "$base_patch"
check_sha256 684568733cd3cc3464fa55ba0def22ed05e33684352c1f8ee5215c95afc832ad "$candidate_patch"
check_sha256 18ad5425d182c8aec18d895e2ba649604ff3e6028c78d3dbc7c81369dabd354a "$benchmark_source"

mkdir -p "$state_dir" "$build_dir"
sha256sum "$prereg" "$base_patch" "$candidate_patch" "$benchmark_source" > "$state_dir/inputs.sha256"
git -C "$repo_root" rev-parse HEAD > "$state_dir/repository.commit"
lscpu -J > "$state_dir/lscpu.json"
cmake --version > "$state_dir/cmake.version"
c++ --version > "$state_dir/compiler.version"

if [[ ! -s "$state_dir/SOURCE_READY" ]]; then
  if [[ -e "$source_dir" ]]; then
    echo "source directory exists without SOURCE_READY receipt: $source_dir" >&2
    exit 73
  fi
  git clone --shared --no-checkout "$source_seed" "$source_dir"
  git -C "$source_dir" switch --detach "$upstream_revision"
  git -C "$source_dir" apply "$base_patch"
  git -C "$source_dir" apply "$candidate_patch"
  install -m 0644 "$benchmark_source" "$source_dir/tests/qwen38_27b_cq3_x86_pair_bench_v1.inc"
  git -C "$source_dir" diff --check
  {
    sha256sum \
      "$source_dir/ggml/src/ggml-cpu/arch/x86/quants.c" \
      "$source_dir/ggml/src/ggml-cpu/quants.h" \
      "$source_dir/tests/test-cq3.cpp" \
      "$source_dir/tests/qwen38_27b_cq3_x86_pair_bench_v1.inc"
  } > "$state_dir/source-files.sha256"
  git -C "$source_dir" diff --stat > "$state_dir/source.diff.stat"
  printf 'ready\n' > "$state_dir/SOURCE_READY"
else
  [[ "$(git -C "$source_dir" rev-parse HEAD)" == "$upstream_revision" ]]
  sha256sum -c "$state_dir/source-files.sha256"
fi

cmake -S "$source_dir" -B "$build_dir" \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  > "$state_dir/cmake.configure.log" 2>&1
cmake --build "$build_dir" --target test-cq3 -j 4 \
  > "$state_dir/cmake.build.log" 2>&1

binary="$build_dir/bin/test-cq3"
if [[ ! -x "$binary" ]]; then
  echo "missing test-cq3 binary: $binary" >&2
  exit 70
fi
sha256sum "$binary" > "$state_dir/binary.sha256"
taskset -c 0 "$binary" \
  > "$state_dir/correctness.stdout" 2> "$state_dir/correctness.stderr"
printf 'pass\n' > "$state_dir/CORRECTNESS_PASS"

taskset -c 0 env CLARK_QWEN38_27B_CQ3_PAIR_BENCH=1 "$binary" \
  > "$state_dir/benchmark.stdout" 2> "$state_dir/benchmark.stderr"
grep '"schema":"qwen38_27b_cq3_x86_pair_benchmark_v1"' "$state_dir/benchmark.stdout" \
  | tail -n 1 > "$state_dir/benchmark.json"

python3 - "$state_dir" "$prereg" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

state = pathlib.Path(sys.argv[1])
prereg = pathlib.Path(sys.argv[2])
bench = json.loads((state / "benchmark.json").read_text())
if bench.get("schema") != "qwen38_27b_cq3_x86_pair_benchmark_v1":
    raise SystemExit("invalid benchmark schema")
if bench.get("exact_control_candidate") is not True:
    raise SystemExit("control/candidate mismatch")
if len(bench.get("control_us", [])) != 9 or len(bench.get("candidate_us", [])) != 9:
    raise SystemExit("incomplete benchmark repetitions")
if not all(float(x) > 0 for key in ("control_us", "candidate_us") for x in bench[key]):
    raise SystemExit("non-positive timing")
speedup = float(bench["speedup"])
promoted = speedup >= 1.03

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

receipt = {
    "schema": "qwen38_27b_cq3_x86_pair_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "preregistration": {"path": str(prereg), "sha256": sha256(prereg)},
    "repository_commit": (state / "repository.commit").read_text().strip(),
    "correctness": {
        "test_cq3_exit_code": 0,
        "bit_exact_control_candidate": True,
        "marker": str(state / "CORRECTNESS_PASS"),
    },
    "benchmark": bench,
    "promotion": {
        "minimum_speedup": 1.03,
        "observed_speedup": speedup,
        "pass": promoted,
        "scope": "Qwen3.8-27B exact-shape AVX2 paired gate/up microkernel only",
        "breakthrough_claim": False,
    },
    "artifacts": {
        name: {"path": str(state / name), "sha256": sha256(state / name)}
        for name in (
            "benchmark.json",
            "benchmark.stdout",
            "benchmark.stderr",
            "correctness.stdout",
            "correctness.stderr",
            "binary.sha256",
            "inputs.sha256",
            "source-files.sha256",
            "lscpu.json",
            "cmake.version",
            "compiler.version",
        )
    },
    "infrastructure": {
        "swap_active_after": bool(os.popen("swapon --noheadings --show 2>/dev/null").read().strip()),
        "valid": True,
    },
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / ("PROMOTION_PASS" if promoted else "PROMOTION_REJECTED")).write_text(
    ("pass" if promoted else "rejected") + "\n"
)
PY

if [[ -n "$(swapon --noheadings --show 2>/dev/null)" ]]; then
  echo "infrastructure-invalid: swap became active" >&2
  exit 75
fi
printf 'complete\n' > "$state_dir/BENCHMARK_COMPLETE"
cat "$state_dir/receipt.json"
