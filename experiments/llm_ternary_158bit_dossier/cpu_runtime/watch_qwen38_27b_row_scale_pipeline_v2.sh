#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen Qwen3.8-27B row-scale-v2 bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-llamacpp-ternary-controls-v1.service}
parent_receipt=${PARENT_RECEIPT:-/workspace/quant-state/qwen38-27b-llamacpp-ternary-controls-watch-v1/receipt.json}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-row-scale-pipeline-v2}
mapped_result=${MAPPED_RESULT:-/workspace/quant-state/qwen38-27b-row-normalized-scale-codebook4-v1.json}
mapped_manifest=${MAPPED_MANIFEST:-/workspace/quant-runs/qwen38-row-normalized-scale-codebook4-v1/manifest.json}
mapped_watch=${MAPPED_WATCH:-/workspace/quant-state/qwen38-27b-row-scale-pipeline-v2/mapped-watch-receipt.json}
full_result=${FULL_RESULT:-/workspace/quant-state/qwen38-27b-full-row-normalized-scale-codebook4-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime

# shellcheck source=wait_for_qwen38_terminal_receipt.sh
source "$(dirname "$0")/wait_for_qwen38_terminal_receipt.sh"
qwen38_wait_for_terminal_receipt "$parent_unit" "$parent_receipt" 30

mapfile -t bundle_commits < <(
  git bundle list-heads "$bundle" "$bundle_branch" |
    awk -v branch="$bundle_branch" '$2 == branch {print $1}'
)
if (( ${#bundle_commits[@]} != 1 )) ||
  [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "frozen Qwen3.8-27B row-scale-v2 bundle is not single-headed" >&2
  exit 64
fi
expected_commit=${bundle_commits[0]}

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B row-scale-v2 repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B row-scale-v2 commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_materiality_priority_amendment_v1.json" \
  "$runtime/qwen38_27b_row_scale_orchestration_amendment_v2.json" \
  "$runtime/qwen38_27b_row_normalized_scale_codebook4_preregistration_v1.json" \
  "$runtime/qwen38_27b_full_row_normalized_scale_codebook4_preregistration_v1.json" \
  "$runtime/watch_qwen38_27b_row_scale_pipeline_v2.sh" \
  "$runtime/run_qwen38_27b_row_normalized_scale_codebook4_v1.py" \
  "$runtime/run_qwen38_27b_full_row_normalized_scale_codebook4_v1.py" \
  clark_quantize/qwen38_row_scale_codebook.py \
  clark_quantize/qwen38_row_scale_storage.py \
  > "$state/source.sha256"

"$python_bin" -m pytest -q \
  clark_quantize/test_qwen38_row_scale_codebook.py \
  clark_quantize/test_qwen38_row_scale_storage.py \
  > "$state/tests.stdout" 2> "$state/tests.stderr"

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
  echo "another QAT or activation-cache process is active" >&2
  printf '%s\n' "$training_conflicts" >&2
  exit 2
fi
if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((8 * 1024 * 1024)) ]]; then
  echo "less than 8 GiB is available for row-scale fitting" >&2
  exit 72
fi
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "swap is active during row-scale admission" >&2
  exit 74
fi
if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((10 * 1024 * 1024)) ]]; then
  echo "less than 10 GiB disk is free for row-scale fitting" >&2
  exit 75
fi
free -b > "$state/memory-before.txt"
df -B1 /workspace > "$state/disk-before.txt"

set +e
PYTHONPATH="$repo" "$python_bin" \
  "$runtime/run_qwen38_27b_row_normalized_scale_codebook4_v1.py" \
  > "$state/mapped.stdout" 2> "$state/mapped.stderr"
mapped_status=$?
set -e
if [[ ! -s "$mapped_result" || ! -s "$mapped_manifest" ]]; then
  echo "mapped row-scale runner exited without both terminal receipts" >&2
  exit 70
fi
"$python_bin" - "$mapped_result" "$mapped_manifest" "$mapped_status" <<'PY'
import hashlib, json, sys
result, manifest = sys.argv[1:3]
status = int(sys.argv[3])
result_content = open(result, "rb").read()
manifest_content = open(manifest, "rb").read()
if result_content != manifest_content:
    raise SystemExit("mapped row-scale result and manifest differ")
payload = json.loads(result_content)
if payload.get("complete") is not True:
    raise SystemExit("mapped row-scale receipt is nonterminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("mapped row-scale exit and gate disagree")
if status not in (0, 2):
    raise SystemExit(f"mapped row-scale unexpected exit {status}")
PY

"$python_bin" - "$mapped_manifest" "$mapped_watch" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
result, watch = map(Path, sys.argv[1:])
content = result.read_bytes()
payload = json.loads(content)
receipt = {
    "schema": "qwen38_27b_row_normalized_scale_codebook4_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "static_reconstruction_pass": payload.get("gate", {}).get("passed") is True,
    "valid_nonpromotion": payload.get("gate", {}).get("passed") is not True,
    "result_receipt": str(result),
    "result_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "claim_boundary": "mapped static scale reconstruction only; v2 orchestration repair",
}
temporary = watch.with_suffix(watch.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, watch)
PY
printf 'terminal\n' > "$state/MAPPED_TERMINAL"

if (( mapped_status == 0 )); then
  set +e
  PYTHONPATH="$repo" "$python_bin" \
    "$runtime/run_qwen38_27b_full_row_normalized_scale_codebook4_v1.py" \
    --mapped-watch-receipt "$mapped_watch" \
    --receipt "$full_result" \
    > "$state/full.stdout" 2> "$state/full.stderr"
  full_status=$?
  set -e
  if [[ ! -s "$full_result" ]]; then
    echo "full row-scale runner exited without a terminal receipt" >&2
    exit 70
  fi
  "$python_bin" - "$full_result" "$full_status" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
status = int(sys.argv[2])
if payload.get("complete") is not True:
    raise SystemExit("full row-scale receipt is nonterminal")
if (payload.get("gate", {}).get("passed") is True) != (status == 0):
    raise SystemExit("full row-scale exit and gate disagree")
if status not in (0, 2):
    raise SystemExit(f"full row-scale unexpected exit {status}")
PY
else
  "$python_bin" - "$mapped_result" "$full_result" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
parent, result = map(Path, sys.argv[1:])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_full_row_scale_parent_rejection_skip_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "complete": True,
    "valid_nonpromotion": True,
    "full_fit_launched": False,
    "reason": "mapped 402-tensor row-normalized reconstruction gate rejected",
    "gate": {"passed": False},
}
temporary = result.with_suffix(result.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, result)
PY
  full_status=2
fi
printf 'terminal\n' > "$state/FULL_TERMINAL"

free -b > "$state/memory-after.txt"
df -B1 /workspace > "$state/disk-after.txt"
"$python_bin" - "$state" "$mapped_result" "$full_result" "$parent_receipt" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, mapped, full, parent = map(Path, sys.argv[1:])
mapped_content = mapped.read_bytes()
full_content = full.read_bytes()
parent_content = parent.read_bytes()
mapped_payload = json.loads(mapped_content)
full_payload = json.loads(full_content)
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "PIPELINE_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
receipt = {
    "schema": "qwen38_27b_row_scale_pipeline_receipt_v2",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "mapped_result": str(mapped),
    "mapped_result_sha256": hashlib.sha256(mapped_content).hexdigest(),
    "mapped_gate_passed": mapped_payload.get("gate", {}).get("passed") is True,
    "full_result": str(full),
    "full_result_sha256": hashlib.sha256(full_content).hexdigest(),
    "full_gate_passed": full_payload.get("gate", {}).get("passed") is True,
    "parent_receipt": str(parent),
    "parent_receipt_sha256": hashlib.sha256(parent_content).hexdigest(),
    "valid_nonpromotion": full_payload.get("gate", {}).get("passed") is not True,
    "gate": {"passed": full_payload.get("gate", {}).get("passed") is True},
    "artifacts": files,
    "claim_boundary": "static scale reconstruction and complete storage only; no capability or breakthrough claim",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "PIPELINE_TERMINAL").write_text("terminal\n")
PY
