#!/usr/bin/env bash
# Align extension builds with the CUDA 13 PyTorch wheel on a supported node.
set -euo pipefail

VENV="${QUANT_VENV:-/workspace/venvs/quant}"
GPU_CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
case "${GPU_CAPABILITY}" in
  8.0|12.0) ;;
  *)
    echo "unsupported GPU compute capability: ${GPU_CAPABILITY}" >&2
    exit 1
    ;;
esac
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y cuda-compiler-13-0 gcc-12 g++-12

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${GPU_CAPABILITY}"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
source "${VENV}/bin/activate"

nvcc --version
python -m pip install 'causal-conv1d==1.6.2.post1' --no-build-isolation
python - <<'PY'
import json
from pathlib import Path

import causal_conv1d
import torch

path = Path("/workspace/quant-state/environment.json")
receipt = json.loads(path.read_text())
receipt["packages"]["causal_conv1d"] = True
receipt.setdefault("package_errors", {}).pop("causal_conv1d", None)
receipt["extension_cuda_home"] = "/usr/local/cuda-13.0"
receipt["extension_cuda_compiler"] = "13.0"
receipt["extension_compute_capability"] = list(torch.cuda.get_device_capability(0))
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print({
    "causal_conv1d": getattr(causal_conv1d, "__version__", "unknown"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
})
PY
