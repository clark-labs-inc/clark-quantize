#!/usr/bin/env bash
# Provision an x86-64 CPU research node for Clark ternary Qwen experiments.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run this bootstrap as root" >&2
  exit 1
fi

REPO="${QUANT_REPO:-/workspace/clark-quantize}"
VENV="${QUANT_VENV:-/workspace/venvs/quant-cpu}"
PYTHON_VERSION="${QUANT_PYTHON_VERSION:-3.13}"
TORCH_VERSION="${QUANT_TORCH_VERSION:-2.13.0}"
LLAMA_CPP_REV="${LLAMA_CPP_REV:-4df29be4f4c3673f428170fda944a5b19f743bb8}"
LLAMA_SRC="${LLAMA_SRC:-/workspace/src/llama.cpp}"
LLAMA_BUILD="${LLAMA_BUILD:-/workspace/build/llama.cpp}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
apt-get update
apt-get install -y \
  aria2 build-essential ca-certificates ccache clang cmake curl git git-lfs \
  htop hwloc jq libopenblas-dev libomp-dev ninja-build numactl pigz \
  pkg-config ripgrep rsync tmux zstd
git lfs install --system

install -d -m 0755 \
  /workspace/build \
  /workspace/hf-cache \
  /workspace/models \
  /workspace/quant-runs \
  /workspace/quant-state \
  /workspace/src \
  /workspace/venvs

if ! command -v uv >/dev/null 2>&1; then
  installer="$(mktemp)"
  curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o "${installer}"
  env UV_INSTALL_DIR=/usr/local/bin sh "${installer}"
  rm -f "${installer}"
fi

uv python install "${PYTHON_VERSION}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  uv venv --python "${PYTHON_VERSION}" "${VENV}"
fi

uv pip install --python "${VENV}/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==${TORCH_VERSION}"
uv pip install --python "${VENV}/bin/python" \
  "accelerate==1.14.0" \
  "datasets>=4.0" \
  "diffusers>=0.38" \
  "huggingface-hub[hf_xet]==1.27.0" \
  "numpy>=2.0" \
  "openai>=1.0" \
  "pillow>=11.0" \
  "protobuf>=5.0" \
  "psutil>=6.0" \
  "pytest>=8.0" \
  "safetensors>=0.4" \
  "scipy>=1.14" \
  "sentencepiece>=0.2" \
  "transformers==5.15.0"

if [[ ! -d "${REPO}/clark_quantize" ]]; then
  echo "Clark repository is not present at ${REPO}" >&2
  exit 1
fi

site_packages="$("${VENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${REPO}" > "${site_packages}/clark_quantize_repo.pth"

if [[ ! -d "${LLAMA_SRC}/.git" ]]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "${LLAMA_SRC}"
fi
git -C "${LLAMA_SRC}" fetch --depth 1 origin "${LLAMA_CPP_REV}"
git -C "${LLAMA_SRC}" checkout --detach "${LLAMA_CPP_REV}"

if ! git -C "${LLAMA_SRC}" grep -Eqi 'gated[_ -]?delta[_ -]?net'; then
  echo "Pinned llama.cpp revision lacks the required Gated DeltaNet graph/operator" >&2
  exit 1
fi

cmake -S "${LLAMA_SRC}" -B "${LLAMA_BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DLLAMA_CURL=OFF
cmake --build "${LLAMA_BUILD}" --parallel "${BUILD_JOBS}"

for binary in llama-bench llama-cli llama-quantize; do
  if [[ -x "${LLAMA_BUILD}/bin/${binary}" ]]; then
    ln -sfn "${LLAMA_BUILD}/bin/${binary}" "/usr/local/bin/${binary}"
  fi
done

install -m 0644 /dev/null /etc/profile.d/quant-cpu.sh
cat > /etc/profile.d/quant-cpu.sh <<'EOF'
export QUANT_REPO=/workspace/clark-quantize
export QUANT_VENV=/workspace/venvs/quant-cpu
export HF_HOME=/workspace/hf-cache
export HF_HUB_CACHE=/workspace/hf-cache/hub
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PATH="/workspace/venvs/quant-cpu/bin:${PATH}"
EOF

export QUANT_REPO="${REPO}"
export QUANT_VENV="${VENV}"
export LLAMA_SRC
export LLAMA_BUILD
export LLAMA_CPP_REV
"${VENV}/bin/python" - <<'PY'
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import torch


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


packages = {}
for name in (
    "accelerate",
    "datasets",
    "diffusers",
    "huggingface-hub",
    "numpy",
    "safetensors",
    "scipy",
    "sentencepiece",
    "torch",
    "transformers",
):
    packages[name] = importlib.metadata.version(name)

receipt = {
    "schema": "clark_quant_cpu_environment_v1",
    "host": platform.node(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "packages": packages,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_threads": torch.get_num_threads(),
    "logical_cpus": os.cpu_count(),
    "cpu_model": command("bash", "-lc", "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"),
    "cpu_flags": command("bash", "-lc", "lscpu | sed -n 's/^Flags:[[:space:]]*//p'"),
    "repo_commit": command("git", "-C", os.environ["QUANT_REPO"], "rev-parse", "HEAD"),
    "repo_branch": command("git", "-C", os.environ["QUANT_REPO"], "branch", "--show-current"),
    "llama_cpp_commit": command("git", "-C", os.environ["LLAMA_SRC"], "rev-parse", "HEAD"),
    "llama_cpp_expected_commit": os.environ["LLAMA_CPP_REV"],
    "llama_build": os.environ["LLAMA_BUILD"],
}
path = Path("/workspace/quant-state/cpu_environment.json")
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))

assert not receipt["torch_cuda_available"]
assert receipt["llama_cpp_commit"] == receipt["llama_cpp_expected_commit"]
PY

echo "CPU bootstrap complete"
echo "activate with: source ${VENV}/bin/activate"
echo "receipt: /workspace/quant-state/cpu_environment.json"
