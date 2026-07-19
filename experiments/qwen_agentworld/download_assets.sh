#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-/workspace/agentworld-venv}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen-AgentWorld-35B-A3B}"
DATASET_ID="${DATASET_ID:-Qwen/AgentWorldBench}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen-AgentWorld-35B-A3B}"
DATA_DIR="${DATA_DIR:-/workspace/AgentWorldBench}"
REPO_DIR="${REPO_DIR:-/workspace/Qwen-AgentWorld}"

mkdir -p /workspace/models "$DATA_DIR"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --depth 1 https://github.com/QwenLM/Qwen-AgentWorld.git "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only || true
fi

"$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    "$DATASET_ID",
    repo_type="dataset",
    local_dir="$DATA_DIR",
    max_workers=8,
)
snapshot_download(
    "$MODEL_ID",
    local_dir="$MODEL_DIR",
    max_workers=8,
)
print("download complete")
PY

du -sh "$MODEL_DIR" "$DATA_DIR" "$REPO_DIR"
