#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v45}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
preregistration=${PREREGISTRATION:-/workspace/quant-state/qwen38-all-layer-sensitivity-residual-v45/preregistration.json}
training_receipt=${TRAINING_RECEIPT:-/workspace/quant-state/qwen38-all-layer-mixed-trained-v44/receipt.json}
artifact_root=${ARTIFACT_ROOT:-/workspace/quant-runs/qwen38-all-layer-sensitivity-residual-v45}
receipt=${RECEIPT:-/workspace/quant-state/qwen38-all-layer-sensitivity-residual-v45/receipt.json}
completion_marker=${COMPLETION_MARKER:-/workspace/quant-state/qwen38-all-layer-sensitivity-residual-v45/complete.json}
unit=${UNIT:-quant-qwen38-27b-all-layer-sensitivity-residual-v45}
parent_unit=quant-qwen38-27b-all-layer-mixed-trained-v44.service
runner=experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_all_layer_sensitivity_residual_v45.py

if systemctl is-active --quiet "$parent_unit"; then
  echo "refusing to overlap v45 residual fitting with active v44 training" >&2
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
test -f "$training_receipt"

"$python_bin" - "$repo" "$runner" "$preregistration" "$training_receipt" <<'PY'
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
training_receipt_path = Path(sys.argv[4])
preregistration = json.loads(preregistration_path.read_text())
training_receipt = json.loads(training_receipt_path.read_text())
head = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
tracked = subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
    text=True,
).strip()
frozen = preregistration.get("frozen_inputs", {})
layers = training_receipt.get("layers", [])
if (
    tracked
    or preregistration.get("schema")
    != "qwen38_27b_all_layer_sensitivity_residual_preregistration_v45"
    or preregistration.get("status")
    != "frozen_after_v44_pass_before_residual_fit"
    or preregistration.get("scope", {}).get("active_model_only")
    != "Qwen/Qwen3.8-27B"
    or frozen.get("runner_repository_commit") != head
    or frozen.get("runner_sha256") != sha256(runner)
    or frozen.get("training_receipt_sha256") != sha256(training_receipt_path)
    or training_receipt.get("schema")
    != "qwen38_27b_all_layer_mixed_training_receipt_v44"
    or training_receipt.get("gate", {}).get("passed") is not True
    or len(layers) != 64
    or [layer.get("layer") for layer in layers] != list(range(64))
    or not all(layer.get("passed") is True for layer in layers)
    or training_receipt.get("complete") is not True
    or training_receipt.get("terminal") is not True
):
    raise SystemExit("v45 launcher identity or passed-parent gate changed")
print(
    json.dumps(
        {
            "event": "v45_launch_identity_passed",
            "repository_commit": head,
            "runner_sha256": sha256(runner),
            "preregistration_sha256": sha256(preregistration_path),
            "training_receipt_sha256": sha256(training_receipt_path),
            "training_layers": len(layers),
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
  --description="Qwen3.8-27B all-layer sensitivity residual v45" \
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
  --training-receipt "$training_receipt" \
  --artifact-root "$artifact_root" \
  --receipt "$receipt" \
  --completion-marker "$completion_marker" \
  --threads 4

systemctl show "$unit.service" \
  -p ActiveState -p SubState -p MainPID -p NRestarts \
  -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax -p AllowedCPUs
