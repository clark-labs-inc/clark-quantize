#!/usr/bin/env bash
# Download the exact CPU-program parent models and write a compact receipt.
set -euo pipefail

VENV="${QUANT_VENV:-/workspace/venvs/quant-cpu}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/models}"
STATE_ROOT="${STATE_ROOT:-/workspace/quant-state}"

if [[ ! -x "${VENV}/bin/hf" ]]; then
  echo "Hugging Face CLI is missing from ${VENV}" >&2
  exit 1
fi

install -d -m 0755 "${MODEL_ROOT}" "${STATE_ROOT}"
exec 9>"${STATE_ROOT}/qwen-parent-download.lock"
if ! flock -n 9; then
  echo "another Qwen parent download is already running" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

"${VENV}/bin/hf" download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir "${MODEL_ROOT}/Qwen3.5-4B"

"${VENV}/bin/hf" download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir "${MODEL_ROOT}/Qwen3.8-27B"

export MODEL_ROOT STATE_ROOT
"${VENV}/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(os.environ["MODEL_ROOT"])
STATE = Path(os.environ["STATE_ROOT"])
SPECS = (
    ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("Qwen/Qwen3.8-27B", "Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


models = []
for repo_id, directory, revision in SPECS:
    root = ROOT / directory
    config = root / "config.json"
    index = root / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise RuntimeError(f"incomplete model metadata in {root}")
    parsed = json.loads(index.read_text())
    shards = sorted(set(parsed["weight_map"].values()))
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"missing model shards in {root}: {missing}")
    shard_bytes = sum((root / name).stat().st_size for name in shards)
    models.append({
        "repo_id": repo_id,
        "revision": revision,
        "directory": str(root),
        "config_sha256": sha256(config),
        "index_sha256": sha256(index),
        "shards": len(shards),
        "safetensors_bytes": shard_bytes,
        "index_total_size": parsed.get("metadata", {}).get("total_size"),
        "complete": True,
    })

receipt = {
    "schema": "clark_qwen_parent_models_v1",
    "models": models,
    "total_safetensors_bytes": sum(model["safetensors_bytes"] for model in models),
}
temporary = STATE / "qwen_parent_models.json.tmp"
final = STATE / "qwen_parent_models.json"
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, final)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
