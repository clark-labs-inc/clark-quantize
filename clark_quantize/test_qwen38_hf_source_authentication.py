from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


RUNTIME = Path(__file__).resolve().parents[1] / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime"
)
sys.path.insert(0, str(RUNTIME))

from verify_qwen38_27b_hf_source_v1 import authenticate_source  # noqa: E402


MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def build_source(root: Path) -> dict[str, Path]:
    model_dir = root / "model"
    model_dir.mkdir()
    shards = {
        "model-00001-of-00002.safetensors": b"first authenticated shard",
        "model-00002-of-00002.safetensors": b"second authenticated shard",
    }
    for name, content in shards.items():
        (model_dir / name).write_bytes(content)
    config = model_dir / "config.json"
    write_json(config, {"model_type": "qwen3_5"})
    index = model_dir / "model.safetensors.index.json"
    tensor_bytes = sum(len(content) for content in shards.values())
    write_json(
        index,
        {
            "metadata": {"total_size": tensor_bytes},
            "weight_map": {"a": next(iter(shards)), "b": list(shards)[1]},
        },
    )
    tree = root / "tree.json"
    write_json(
        tree,
        {
            "format_version": 1,
            "files": {
                name: {
                    "lfs_sha256": hashlib.sha256(content).hexdigest(),
                    "lfs_size": len(content),
                }
                for name, content in shards.items()
            },
        },
    )
    parent = root / "parents.json"
    write_json(
        parent,
        {
            "models": [
                {
                    "complete": True,
                    "repo_id": MODEL,
                    "revision": REVISION,
                    "config_sha256": sha256(config),
                    "index_sha256": sha256(index),
                    "index_total_size": tensor_bytes,
                    "safetensors_bytes": tensor_bytes,
                    "shards": len(shards),
                }
            ]
        },
    )
    download = model_dir / "CLARK_DOWNLOAD_RECEIPT.json"
    write_json(
        download,
        {
            "schema": "qwen38_model_download_v1",
            "model_id": MODEL,
            "revision": REVISION,
            "index_sha256": sha256(index),
            "bytes": tensor_bytes,
            "shards": len(shards),
        },
    )
    preregistration = root / "preregistration.json"
    write_json(
        preregistration,
        {
            "model": MODEL,
            "revision": REVISION,
            "source_authentication": {
                "parent_receipt_sha256": sha256(parent),
                "download_receipt_sha256": sha256(download),
                "hf_tree_manifest_sha256": sha256(tree),
                "config_sha256": sha256(config),
                "model_index_sha256": sha256(index),
                "safetensors_bytes": tensor_bytes,
                "shards": len(shards),
            },
        },
    )
    return {
        "model_dir": model_dir,
        "parent": parent,
        "download": download,
        "tree": tree,
        "preregistration": preregistration,
    }


def authenticate(paths: dict[str, Path]) -> dict[str, object]:
    return authenticate_source(
        model_dir=paths["model_dir"],
        parent_receipt_path=paths["parent"],
        download_receipt_path=paths["download"],
        tree_manifest_path=paths["tree"],
        preregistration_path=paths["preregistration"],
    )


def test_authenticates_every_indexed_shard(tmp_path: Path) -> None:
    paths = build_source(tmp_path)
    receipt = authenticate(paths)
    assert receipt["valid"] is True
    assert receipt["terminal"] is True
    assert receipt["shard_count"] == 2
    assert receipt["safetensors_bytes"] == sum(
        row["bytes"] for row in receipt["shards"]
    )


def test_rejects_changed_shard_bytes(tmp_path: Path) -> None:
    paths = build_source(tmp_path)
    (paths["model_dir"] / "model-00002-of-00002.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte count mismatch|SHA-256 mismatch"):
        authenticate(paths)
