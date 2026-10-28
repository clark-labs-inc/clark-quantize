#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v44}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
preregistration=${PREREGISTRATION:-/workspace/quant-state/qwen38-all-layer-mixed-trained-v44/preregistration.json}
direct_receipt=${DIRECT_RECEIPT:-/workspace/quant-state/qwen38-all-layer-mixed-direct-v43/receipt.json}
full_context_receipt=${FULL_CONTEXT_RECEIPT:-/workspace/quant-state/qwen38-full-context-block-cache-v42/receipt.json}
artifact_root=${ARTIFACT_ROOT:-/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44}
receipt=${RECEIPT:-/workspace/quant-state/qwen38-all-layer-mixed-trained-v44/receipt.json}
completion_marker=${COMPLETION_MARKER:-/workspace/quant-state/qwen38-all-layer-mixed-trained-v44/complete.json}
unit=${UNIT:-quant-qwen38-27b-all-layer-mixed-trained-v44}
parent_unit=quant-qwen38-27b-all-layer-mixed-direct-v43.service
runner=experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_all_layer_mixed_train_v44.py

if systemctl is-active --quiet "$parent_unit"; then
  echo "refusing to overlap v44 training with active v43 parent" >&2
  exit 2
fi
if systemctl is-active --quiet "$unit.service"; then
  systemctl show "$unit.service" -p ActiveState -p SubState -p MainPID -p NRestarts
  exit 0
fi
if [[ $(systemctl show "$unit.service" -p LoadState --value 2>/dev/null || true) != not-found ]]; then
  echo "$unit.service already exists but is not active; inspect it before relaunch" >&2
  exit 2
fi

test -x "$python_bin"
test -f "$repo/$runner"
test -f "$preregistration"
test -f "$direct_receipt"
test -f "$full_context_receipt"

"$python_bin" - "$repo" "$runner" "$preregistration" "$direct_receipt" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


repo = Path(sys.argv[1])
runner = repo / sys.argv[2]
preregistration_path = Path(sys.argv[3])
direct_receipt_path = Path(sys.argv[4])
preregistration = json.loads(preregistration_path.read_text())
direct_receipt = json.loads(direct_receipt_path.read_text())
head = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
tracked = subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
    text=True,
).strip()
frozen_runner = preregistration.get("frozen_inputs", {}).get("runner", {})
if (
    tracked
    or preregistration.get("schema")
    != "qwen38_27b_all_layer_mixed_training_preregistration_v44"
    or preregistration.get("status") != "frozen_after_v43_pass_before_training"
    or preregistration.get("scope", {}).get("active_model_only")
    != "Qwen/Qwen3.8-27B"
    or preregistration.get("scope", {}).get("target_layers") != list(range(64))
    or frozen_runner.get("repository_commit") != head
    or frozen_runner.get("sha256") != sha256(runner)
    or preregistration.get("parent_direct", {}).get("receipt_sha256")
    != sha256(direct_receipt_path)
    or direct_receipt.get("gate", {}).get("passed") is not True
    or len(direct_receipt.get("layers", [])) != 64
    or direct_receipt.get("complete") is not True
    or direct_receipt.get("terminal") is not True
):
    raise SystemExit("v44 launcher identity or passed-parent gate changed")
print(
    json.dumps(
        {
            "event": "v44_launch_identity_passed",
            "repository_commit": head,
            "runner_sha256": sha256(runner),
            "preregistration_sha256": sha256(preregistration_path),
            "direct_receipt_sha256": sha256(direct_receipt_path),
            "terminal": True,
        },
        sort_keys=True,
    )
)
PY

if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a Qwen 4B-era service is active; refusing to share the node" >&2
  exit 2
fi

systemd-run \
  --unit="$unit" \
  --description="Qwen3.8-27B all-layer mixed training v44" \
  --property=Type=exec \
  --property=CPUQuota=400% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=MemoryMax=31000000000 \
  --property=MemorySwapMax=0 \
  --property=Restart=on-failure \
  --property=RestartPreventExitStatus=2 \
  --property=RestartSec=5s \
  --property=StartLimitIntervalSec=6h \
  --property=StartLimitBurst=3 \
  --property=WorkingDirectory="$repo" \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 \
  --setenv=TOKENIZERS_PARALLELISM=false \
  /usr/bin/taskset -c 0,2,4,6 \
  "$python_bin" -u "$runner" \
  --preregistration "$preregistration" \
  --direct-receipt "$direct_receipt" \
  --full-context-receipt "$full_context_receipt" \
  --artifact-root "$artifact_root" \
  --receipt "$receipt" \
  --completion-marker "$completion_marker" \
  --threads 4

systemctl show "$unit.service" \
  -p ActiveState -p SubState -p MainPID -p NRestarts \
  -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax -p AllowedCPUs
