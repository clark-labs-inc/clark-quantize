#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen Qwen3.8-27B export source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed export source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent=${PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-full-scale-codebook4-v1.json}
parent_watch=${PARENT_WATCH_STATE:-/workspace/quant-state/qwen38-27b-full-scale-codebook4-watch-v1}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-cq3-export-watch-v1}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

units=(
  quant-qwen38-27b-cpu-q4-baseline-v1.service
  quant-qwen38-27b-postbaseline-chain-v1.service
  quant-qwen38-27b-paired-throughput-watch-v1.service
  quant-qwen38-27b-cq3-x86-pair-watch-v1.service
  quant-qwen38-27b-scale-codebook4-watch-v1.service
  quant-qwen38-27b-block203-codec-watch-v2.service
  quant-qwen38-27b-block203-radix243-watch-v1.service
  quant-qwen38-27b-full-scale-codebook4-watch-v1.service
)
while :; do
  active=false
  for unit in "${units[@]}"; do
    if systemctl is-active --quiet "$unit"; then
      active=true
      break
    fi
  done
  if [[ "$active" == false ]]; then
    break
  fi
  sleep 30
done

mkdir -p "$state_dir"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B research repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B export source commit" >&2
  exit 65
fi

"$python_bin" - "$parent" "$parent_watch" "$state_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

parent = Path(sys.argv[1])
parent_watch = Path(sys.argv[2])
state = Path(sys.argv[3])
if not parent.is_file():
    if not (parent_watch / "PARENT_REJECTED").is_file():
        raise SystemExit("complete 666-tensor scale receipt is missing without a terminal parent rejection")
    content = (parent_watch / "receipt.json").read_bytes()
    reason = "the mapped 402-tensor prerequisite rejected before supplemental scale fitting"
    parent_path = parent_watch / "receipt.json"
else:
    content = parent.read_bytes()
    payload = json.loads(content)
    if (
        payload.get("schema") == "qwen38_27b_full_scale_codebook4_receipt_v1"
        and payload.get("complete") is True
        and payload.get("gate", {}).get("passed") is True
    ):
        raise SystemExit(0)
    reason = "the complete frozen 666-tensor scale-codebook gate did not pass"
    parent_path = parent
receipt = {
    "schema": "qwen38_27b_cq3_export_parent_rejection_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_receipt": str(parent_path),
    "parent_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "valid_nonpromotion": True,
    "reason": reason,
    "artifact_export_launched": False,
    "model": "Qwen/Qwen3.8-27B",
}
temporary = state / "receipt.json.tmp"
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(state / "receipt.json")
(state / "PARENT_REJECTED").write_text("rejected\n")
PY
if [[ -s "$state_dir/PARENT_REJECTED" ]]; then
  exit 0
fi

scripts/cpu/build_ternary_cpu.sh > "$state_dir/build-native.stdout" 2> "$state_dir/build-native.stderr"
exec "$python_bin" "$runtime/run_qwen38_27b_cq3_export_v1.py"
