#!/usr/bin/env python3
"""Resumably download and verify the pinned Qwen3.8-27B checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import snapshot_download


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.model_dir,
        max_workers=args.workers,
    )
    index = args.model_dir / "model.safetensors.index.json"
    if not index.exists():
        raise RuntimeError(f"missing {index}")
    payload = json.loads(index.read_text())
    shards = sorted(set(payload["weight_map"].values()))
    missing = [name for name in shards if not (args.model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} model shards, first={missing[0]}")
    receipt = {
        "schema": "qwen38_model_download_v1",
        "model_id": args.model_id,
        "revision": args.revision,
        "shards": len(shards),
        "bytes": sum((args.model_dir / name).stat().st_size for name in shards),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
    }
    atomic_json(args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
