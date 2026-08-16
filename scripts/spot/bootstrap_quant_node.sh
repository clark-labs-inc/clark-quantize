#!/usr/bin/env bash
# Bootstrap the pinned CUDA 13 research environment used by Qwen3.8-27B probes.
set -euo pipefail

REPO="${QUANT_REPO:-/workspace/clark-quantize}"
VENV="${QUANT_VENV:-/workspace/venvs/quant}"
QUANT_OWNER="${QUANT_OWNER:-${SUDO_USER:-$(id -un)}}"
QUANT_GROUP="$(id -gn "${QUANT_OWNER}")"
GPU_CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
case "${GPU_CAPABILITY}" in
  8.0|12.0) ;;
  *)
    echo "unsupported GPU compute capability: ${GPU_CAPABILITY}" >&2
    exit 1
    ;;
esac

if [[ "$(id -u)" -eq 0 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y python3-pip python3-venv build-essential gcc-12 g++-12 ninja-build git-lfs \
    aria2 pigz numactl cuda-compiler-13-0
fi

mkdir -p "$(dirname "${VENV}")" /workspace/models /workspace/quant-runs \
  /workspace/quant-state /workspace/hf-cache
python3 -m venv "${VENV}"
source "${VENV}/bin/activate"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_CAPABILITY}}"
export CC="${CC:-/usr/bin/gcc-12}"
export CXX="${CXX:-/usr/bin/g++-12}"
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.13.0' torchvision
python -m pip install \
  'transformers==5.15.0' 'accelerate==1.14.0' 'huggingface-hub[hf_xet]==1.27.0' \
  safetensors datasets sentencepiece protobuf psutil pytest numpy scipy openai

# FLA is required for usable Qwen Gated DeltaNet throughput. Keep a failure
# visible in the receipt; do not silently call the torch fallback production-ready.
python -m pip install 'flash-linear-attention==0.5.2'
python -m pip install 'causal-conv1d==1.6.2.post1' --no-build-isolation

python - <<'PY'
import json
import importlib
import os
from pathlib import Path
import torch

expected_capability = [int(part) for part in os.environ["TORCH_CUDA_ARCH_LIST"].split(".")]
packages = {}
package_errors = {}
for name in ("transformers", "accelerate", "triton", "fla", "causal_conv1d", "safetensors"):
    try:
        importlib.import_module(name)
        packages[name] = True
    except BaseException as error:
        packages[name] = False
        package_errors[name] = f"{type(error).__name__}: {error}"
receipt = {
    "schema": "clark_quant_environment_v1",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "expected_capability": expected_capability,
    "packages": packages,
    "package_errors": package_errors,
}
path = Path("/workspace/quant-state/environment.json")
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
assert receipt["cuda_available"]
assert receipt["capability"] == expected_capability
assert all(receipt["packages"].values()), receipt["package_errors"]
PY

install -m 0755 "${REPO}/scripts/spot/watchdog.sh" /usr/local/bin/quant-spot-watchdog
install -m 0644 "${REPO}/scripts/spot/quant-spot-watchdog.service" \
  /etc/systemd/system/quant-spot-watchdog.service
systemctl daemon-reload
systemctl enable --now quant-spot-watchdog.service

# The provisioner may run under sudo even though interactive experiment jobs
# run as the SSH account. Keep the workspace usable by that single-tenant
# operator while system services continue to run as root.
chown -R "${QUANT_OWNER}:${QUANT_GROUP}" \
  /workspace/venvs /workspace/models /workspace/quant-runs \
  /workspace/quant-state /workspace/hf-cache

echo "bootstrap complete: source ${VENV}/bin/activate"
