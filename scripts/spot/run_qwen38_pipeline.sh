#!/usr/bin/env bash
# Reboot-safe, receipt-gated continuation for the A100 Qwen3.8-27B experiment.
set -euo pipefail

REPO="${QUANT_REPO:-/workspace/clark-quantize}"
PYTHON="${QUANT_PYTHON:-/workspace/venvs/quant/bin/python}"
STATE="${QUANT_STATE_ROOT:-/workspace/quant-state}"
RUNS="${QUANT_RUN_ROOT:-/workspace/quant-runs}"
HARNESS="${REPO}/scripts/spot/run_spot_job.sh"
CHECKPOINTS="${RUNS}/qwen38-qat/checkpoints"
MODEL_DIR="${QUANT_MODEL_DIR:-/workspace/models/Qwen3.8-27B}"
BASE3_NATIVE_SOURCE="${REPO}/clark_quantize/base3_native.c"
BASE3_NATIVE_LIBRARY="${REPO}/clark_quantize/_base3_native.so"
REQUIRE_NATIVE_BASE3="${QUANT_REQUIRE_NATIVE_BASE3:-1}"

test -x "${PYTHON}"
test -x "${HARNESS}"
test -d "${MODEL_DIR}"
mkdir -p "${STATE}" "${RUNS}" "${CHECKPOINTS}"

# GPU device nodes and the driver can lag multi-user.target after a reboot.
# Wait a bounded five minutes before declaring recovery unavailable.
gpu_ready=false
for _ in $(seq 1 60); do
  if nvidia-smi -L >/dev/null 2>&1; then
    gpu_ready=true
    break
  fi
  sleep 5
done
if [[ "${gpu_ready}" != true ]]; then
  echo "NVIDIA GPU did not become ready within five minutes" >&2
  exit 5
fi

receipt_passed() {
  "${PYTHON}" - "$1" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    receipt = json.loads(path.read_text())
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if receipt.get("promotion", {}).get("passed") is True else 1)
PY
}

receipt_schema() {
  "${PYTHON}" - "$1" "$2" <<'PY'
import json
from pathlib import Path
import sys

try:
    receipt = json.loads(Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if receipt.get("schema") == sys.argv[2] else 1)
PY
}

run_job() {
  local name="$1"
  shift
  runuser --user ubuntu -- "${HARNESS}" "${name}" -- "$@"
}

run_scientific_job() {
  local name="$1"
  local receipt="$2"
  local schema="$3"
  shift 3
  if [[ -e "${receipt}" ]]; then
    if receipt_schema "${receipt}" "${schema}"; then
      echo "${name}: terminal receipt already exists; refusing automatic rerun"
      return 0
    fi
    echo "${name}: malformed or wrong-schema receipt requires inspection: ${receipt}" >&2
    return 4
  fi
  set +e
  run_job "${name}" "$@"
  local rc=$?
  set -e
  if receipt_schema "${receipt}" "${schema}"; then
    echo "${name}: accepted terminal scientific receipt after exit ${rc}"
    return 0
  fi
  echo "${name}: exited ${rc} without a valid terminal receipt" >&2
  if [[ "${rc}" -eq 0 ]]; then
    return 1
  fi
  return "${rc}"
}

run_root_job() {
  local name="$1"
  shift
  "${HARNESS}" "${name}" -- "$@"
}

# Enforce retention before doing any work. Incomplete dot-prefixed atomic saves
# remain untouched for diagnosis and cause later checkpoint writes to fail closed.
"${PYTHON}" "${REPO}/scripts/spot/prune_qat_checkpoints.py" \
  --root "${CHECKPOINTS}" --keep 2

if [[ "${REQUIRE_NATIVE_BASE3}" == "1" ]]; then
  if [[ ! -f "${BASE3_NATIVE_LIBRARY}" || "${BASE3_NATIVE_SOURCE}" -nt "${BASE3_NATIVE_LIBRARY}" ]]; then
    run_root_job base3-native-build bash "${REPO}/scripts/spot/build_base3_native.sh"
  fi
  PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" - <<'PY'
from clark_quantize.base3 import native_decoder_available
if not native_decoder_available():
    raise SystemExit("native ideal-base3 decoder is unavailable")
PY
fi

# A manually started transient pack may already own the job lock. Let it finish
# (or exhaust its own restart policy) before deciding whether a resume is needed.
while systemctl is-active --quiet quant-qwen38-base3-full.service; do
  sleep 30
done

if ! receipt_passed "${STATE}/qwen38-base3-full.json"; then
  run_job qwen38-base3-full "${PYTHON}" \
    "${REPO}/experiments/qwen38_27b_spot/full_base3_pack.py" \
    --workers 14 --max-in-flight 28
fi

if ! receipt_passed "${STATE}/qwen38-retained-export.json"; then
  run_job qwen38-retained-export "${PYTHON}" \
    "${REPO}/experiments/qwen38_27b_spot/export_retained_tensors.py"
fi

run_scientific_job \
  qwen38-base3-quality \
  "${STATE}/qwen38-base3-quality.json" \
  clark_qwen38_base3_behavioral_quality_v1 \
  "${PYTHON}" "${REPO}/experiments/qwen38_27b_spot/behavioral_quality_gate.py"

if ! "${PYTHON}" - "${STATE}/clark-scientist-update.json" \
  "${REPO}/experiments/qwen38_27b_spot/source_provenance.json" <<'PY'
import json
import hashlib
from pathlib import Path
import subprocess
import sys

try:
    receipt = json.loads(Path(sys.argv[1]).read_text())
    provenance = json.loads(Path(sys.argv[2]).read_text())
    expected = {
        "clark_revision": provenance["clark_private_product"]["revision"],
        "clark_desktop_revision": provenance["clark_desktop_foundation"]["revision"],
        "autoresearch_revision": provenance["clark_autoresearch"]["revision"],
        "compaction_revision": provenance["clark_agent_compaction"]["revision"],
    }
    expected_version = provenance["clark_private_product"]["version"]
    if (
        receipt.get("schema") != "clark_scientist_quant_update_v1"
        or receipt.get("installed_path") != "/usr/local/bin/clark-code-headless"
        or any(receipt.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("Clark worker source receipt is stale")
    digest = hashlib.sha256()
    with open("/usr/local/bin/clark-code-headless", "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != receipt.get("worker_sha256"):
        raise ValueError("installed Clark worker hash differs from receipt")
    self_test = json.loads(subprocess.check_output(
        ["/usr/local/bin/clark-code-headless", "--self-test"], text=True
    ))
    valid = (
        self_test.get("status") == "passed"
        and self_test.get("worker") == "clark-code-headless"
        and self_test.get("protocol_version") == 1
        and self_test.get("specialist_count") == 2
        and self_test.get("worker_version") == expected_version
        and receipt.get("self_test") == self_test
    )
except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
then
  run_root_job clark-worker-refresh bash "${REPO}/scripts/spot/update_clark_scientist.sh"
fi

run_scientific_job \
  qwen38-radix5-performance \
  "${STATE}/qwen38-radix5-performance.json" \
  clark_qwen38_radix5_performance_v1 \
  "${PYTHON}" "${REPO}/experiments/qwen38_27b_spot/radix5_performance_gate.py"

run_scientific_job \
  qwen38-layer-sensitivity \
  "${STATE}/qwen38-layer-sensitivity.json" \
  clark_qwen38_layer_sensitivity_v1 \
  "${PYTHON}" "${REPO}/experiments/qwen38_27b_spot/layer_sensitivity_gate.py"
if ! receipt_passed "${STATE}/qwen38-layer-sensitivity.json"; then
  echo "layer sensitivity produced a valid non-promoting receipt; no QAT target is authorized" >&2
  exit 3
fi

run_scientific_job \
  qwen38-qat-teacher-cache \
  "${STATE}/qwen38-qat-teacher-cache.json" \
  clark_qwen38_qat_teacher_cache_receipt_v1 \
  "${PYTHON}" "${REPO}/experiments/qwen38_27b_spot/qat_teacher_cache.py"
if ! receipt_passed "${STATE}/qwen38-qat-teacher-cache.json"; then
  echo "teacher cache produced a valid non-promoting receipt; QAT cannot start" >&2
  exit 3
fi

QAT_RECEIPT="${STATE}/qwen38-one-layer-qat.json"
if [[ -s "${QAT_RECEIPT}" ]]; then
  if receipt_schema "${QAT_RECEIPT}" "clark_qwen38_one_layer_qat_v1"; then
    echo "QAT terminal receipt already exists; refusing an automatic rerun"
    "${PYTHON}" "${REPO}/scripts/spot/prune_qat_checkpoints.py" \
      --root "${CHECKPOINTS}" --keep 2
    exit 0
  fi
  echo "QAT receipt is malformed or has the wrong schema; refusing an automatic rerun" >&2
  exit 4
fi

complete_checkpoints="$(${PYTHON} - "${CHECKPOINTS}" <<'PY'
from pathlib import Path
import json
import re
import sys

root = Path(sys.argv[1])
count = 0
for path in root.iterdir():
    if not re.fullmatch(r"checkpoint-[0-9]+", path.name) or not path.is_dir():
        continue
    manifest = path / "checkpoint.json"
    if manifest.is_file() and json.loads(manifest.read_text()).get("complete") is True:
        count += 1
print(count)
PY
)"
qat_extra_args=()
if [[ "${complete_checkpoints}" == "1" && -d "${CHECKPOINTS}/checkpoint-10" ]]; then
  echo "QAT receipt is absent with one inspected checkpoint-10; running receipt finalization without retraining"
  qat_extra_args+=(--finalize-checkpoint)
elif [[ "${complete_checkpoints}" != "0" ]]; then
  echo "QAT receipt is absent but ${complete_checkpoints} unexpected complete checkpoint(s) exist; inspect before retry" >&2
  exit 4
fi

run_scientific_job \
  qwen38-one-layer-qat \
  "${QAT_RECEIPT}" \
  clark_qwen38_one_layer_qat_v1 \
  "${PYTHON}" "${REPO}/experiments/qwen38_27b_spot/one_layer_qat.py" "${qat_extra_args[@]}"

"${PYTHON}" "${REPO}/scripts/spot/prune_qat_checkpoints.py" \
  --root "${CHECKPOINTS}" --keep 2
