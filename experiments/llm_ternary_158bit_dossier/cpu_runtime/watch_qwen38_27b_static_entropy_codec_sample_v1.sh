#!/usr/bin/env bash
set -euo pipefail

readonly bundle="${BUNDLE:-/workspace/tmp/clark-quantize-qwen38-static-entropy.bundle}"
readonly bundle_branch="${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}"
readonly checkout="/workspace/tmp/clark-quantize-qwen38-static-entropy"
readonly rate_unit="${RATE_UNIT:-quant-qwen38-27b-ternary-type-class-rate-v1.service}"
readonly hash_unit="${HASH_UNIT:-quant-qwen38-27b-hash-residual-screen-v1.service}"
readonly rate_receipt="/workspace/quant-state/qwen38-27b-ternary-type-class-rate-v1.json"
readonly receipt="/workspace/quant-state/qwen38-27b-static-entropy-codec-sample-v1.json"
readonly log="/workspace/quant-runs/qwen38-27b-static-entropy-codec-sample-v1.stdout"
readonly python_bin="/workspace/venvs/quant-cpu/bin/python"

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "infrastructure-invalid: static entropy bundle lacks one pinned branch head" >&2
  exit 64
fi
readonly expected_commit=${bundle_commits[0]}

for prerequisite in "$rate_unit" "$hash_unit"; do
  while systemctl is-active --quiet "$prerequisite"; do
    sleep 15
  done
done
if [[ ! -s "$rate_receipt" ]]; then
  echo "infrastructure-invalid: complete Qwen3.8-27B rate receipt is absent" >&2
  exit 70
fi

if ! "$python_bin" - "$rate_receipt" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
valid = (
    value.get("schema") == "qwen38_27b_ternary_type_class_rate_receipt_v1"
    and isinstance(value.get("gate"), dict)
    and isinstance(value["gate"].get("passed"), bool)
)
raise SystemExit(0 if valid and value["gate"]["passed"] else 1)
PY
then
  "$python_bin" - "$rate_receipt" "$receipt" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
parent, output = map(Path, sys.argv[1:])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_static_entropy_codec_sample_skip_receipt_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_rate_receipt": str(parent),
    "parent_rate_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "valid_nonpromotion": True,
    "experiment_executed": False,
    "reason": "the frozen complete nonuniform Qwen3.8-27B rate gate rejected",
    "gate": {"passed": False, "executed": False, "parent_rate_gate_passed": False},
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
  exit 0
fi

if [[ -e "$checkout" ]]; then
  echo "infrastructure-invalid: static entropy checkout path already exists" >&2
  exit 71
fi
if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((4 * 1024 * 1024)) ]]; then
  echo "infrastructure-invalid: less than 4 GiB memory is available" >&2
  exit 72
fi
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "infrastructure-invalid: swap is active" >&2
  exit 73
fi

git clone --quiet --no-checkout "$bundle" "$checkout"
git -C "$checkout" checkout --quiet --detach "$expected_commit"
if [[ $(git -C "$checkout" rev-parse HEAD) != "$expected_commit" ]]; then
  echo "infrastructure-invalid: static entropy checkout commit changed" >&2
  exit 74
fi
cd "$checkout"
cc -O3 -std=c11 -Wall -Wextra -Werror -shared -fPIC \
  clark_quantize/base3_native.c -o clark_quantize/_base3_native.so
"$python_bin" -m pytest -q \
  clark_quantize/test_base3.py \
  clark_quantize/test_qwen38_static_ternary_codec.py
mkdir -p "$(dirname "$receipt")" "$(dirname "$log")"
taskset -c 0 "$python_bin" \
  experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_static_entropy_codec_sample_v1.py \
  --out "$receipt" | tee "$log"
