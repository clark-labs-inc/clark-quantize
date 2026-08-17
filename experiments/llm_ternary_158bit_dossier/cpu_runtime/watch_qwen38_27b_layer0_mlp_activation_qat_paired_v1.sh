#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen paired-throughput source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-layer0-mlp-activation-qat-watch-v1.service}
parent_state=${PARENT_STATE:-/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-watch-v1}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-paired-watch-v1}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
throughput=/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-paired-throughput-v1

while systemctl is-active --quiet "$parent_unit"; do
  sleep 30
done
if [[ ! -s "$parent_state/WATCH_TERMINAL" || ! -s "$parent_state/receipt.json" ]]; then
  echo "parent Qwen3.8-27B activation-QAT watch lacks a terminal receipt" >&2
  exit 3
fi

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B paired-throughput repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B paired-throughput commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_layer0_mlp_activation_qat_paired_throughput_preregistration_v1.json" \
  "$runtime/run_qwen38_27b_layer0_mlp_activation_qat_paired_throughput_v1.sh" \
  "$runtime/summarize_qwen38_27b_paired_throughput.py" \
  > "$state/source.sha256"

if [[ -s "$parent_state/PAIRED_THROUGHPUT_REQUIRED" ]]; then
  "$runtime/run_qwen38_27b_layer0_mlp_activation_qat_paired_throughput_v1.sh" \
    > "$state/paired-throughput.stdout" 2> "$state/paired-throughput.stderr"
  printf 'complete\n' > "$state/PAIRED_THROUGHPUT_COMPLETE"
else
  "$python_bin" - "$parent_state" "$state" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
parent, state = map(Path, sys.argv[1:])
content = (parent / "receipt.json").read_bytes()
receipt = {
    "schema": "qwen38_27b_activation_qat_paired_throughput_skip_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_receipt": str(parent / "receipt.json"),
    "parent_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "valid_nonpromotion": True,
    "paired_throughput_launched": False,
    "reason": "activation-QAT numeric or capability promotion did not pass",
}
(state / "paired-throughput-skip.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
  printf 'skipped\n' > "$state/PAIRED_THROUGHPUT_SKIPPED"
fi

"$python_bin" - "$state" "$throughput" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, throughput = map(Path, sys.argv[1:])
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
receipt = {
    "schema": "qwen38_27b_layer0_mlp_activation_qat_paired_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "paired_throughput_complete": (state / "PAIRED_THROUGHPUT_COMPLETE").is_file(),
    "paired_throughput_skipped": (state / "PAIRED_THROUGHPUT_SKIPPED").is_file(),
    "throughput_summary": str(throughput / "summary.json") if (throughput / "summary.json").is_file() else None,
    "artifacts": files,
    "claim_boundary": "watch orchestration only; speed claims remain in the paired summary",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
