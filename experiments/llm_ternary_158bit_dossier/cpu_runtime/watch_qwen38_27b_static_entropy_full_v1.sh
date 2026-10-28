#!/usr/bin/env bash
set -euo pipefail

readonly bundle="${BUNDLE:-/workspace/tmp/clark-quantize-qwen38-static-entropy-full.bundle}"
readonly bundle_branch="${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}"
readonly checkout="/workspace/tmp/clark-quantize-qwen38-static-entropy-full"
readonly sample_unit="${SAMPLE_UNIT:-quant-qwen38-27b-static-entropy-codec-sample-v1.service}"
readonly sample_receipt="/workspace/quant-state/qwen38-27b-static-entropy-codec-sample-v1.json"
readonly rate_receipt="/workspace/quant-state/qwen38-27b-ternary-type-class-rate-v1.json"
readonly receipt="/workspace/quant-state/qwen38-27b-static-entropy-full-v1.json"
readonly log="/workspace/quant-runs/qwen38-27b-static-entropy-full-v1.stdout"
readonly python_bin="/workspace/venvs/quant-cpu/bin/python"

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "infrastructure-invalid: full static-export bundle lacks one pinned branch head" >&2
  exit 64
fi
readonly expected_commit=${bundle_commits[0]}

while systemctl is-active --quiet "$sample_unit"; do
  sleep 15
done
if [[ ! -s "$sample_receipt" || ! -s "$rate_receipt" ]]; then
  echo "infrastructure-invalid: full static-export parent receipt is absent" >&2
  exit 70
fi
if ! "$python_bin" - "$sample_receipt" "$rate_receipt" <<'PY'
import json, sys
sample, rate = (json.load(open(path)) for path in sys.argv[1:])
passed = (
    sample.get("schema") == "qwen38_27b_static_entropy_codec_sample_receipt_v1"
    and sample.get("gate", {}).get("passed") is True
    and rate.get("schema") == "qwen38_27b_ternary_type_class_rate_receipt_v1"
    and rate.get("gate", {}).get("passed") is True
)
raise SystemExit(0 if passed else 1)
PY
then
  "$python_bin" - "$sample_receipt" "$rate_receipt" "$receipt" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
sample, rate, output = map(Path, sys.argv[1:])
receipt = {
    "schema": "qwen38_27b_static_entropy_full_skip_receipt_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parents": {
        "sample": {"path": str(sample), "sha256": hashlib.sha256(sample.read_bytes()).hexdigest()},
        "rate": {"path": str(rate), "sha256": hashlib.sha256(rate.read_bytes()).hexdigest()},
    },
    "terminal": True,
    "valid_nonpromotion": True,
    "experiment_executed": False,
    "reason": "the frozen complete rate or real-chunk static-codec gate rejected",
    "gate": {"passed": False, "executed": False, "parent_gates_passed": False},
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
  exit 0
fi

if [[ -e "$checkout" ]]; then
  echo "infrastructure-invalid: full static-export checkout path already exists" >&2
  exit 71
fi
if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((8 * 1024 * 1024)) ]]; then
  echo "infrastructure-invalid: less than 8 GiB memory is available" >&2
  exit 72
fi
if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((20 * 1024 * 1024)) ]]; then
  echo "infrastructure-invalid: less than 20 GiB disk is available" >&2
  exit 73
fi
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "infrastructure-invalid: swap is active" >&2
  exit 74
fi

git clone --quiet --no-checkout "$bundle" "$checkout"
git -C "$checkout" checkout --quiet --detach "$expected_commit"
if [[ $(git -C "$checkout" rev-parse HEAD) != "$expected_commit" ]]; then
  echo "infrastructure-invalid: full static-export checkout commit changed" >&2
  exit 75
fi
cd "$checkout"
cc -O3 -std=c11 -Wall -Wextra -Werror -shared -fPIC \
  clark_quantize/base3_native.c -o clark_quantize/_base3_native.so
"$python_bin" -m pytest -q \
  clark_quantize/test_base3.py \
  clark_quantize/test_qwen38_static_ternary_codec.py \
  clark_quantize/test_qwen38_static_chunk_store.py
mkdir -p "$(dirname "$receipt")" "$(dirname "$log")"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
taskset -c 0-3 "$python_bin" \
  experiments/llm_ternary_158bit_dossier/cpu_runtime/export_qwen38_27b_static_entropy_full_v1.py \
  --workers 4 --batch-chunks 16 --out "$receipt" | tee "$log"
