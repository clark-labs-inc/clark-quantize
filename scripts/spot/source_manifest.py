#!/usr/bin/env python3
"""Hash the exact scoped source transferred to ephemeral and durable nodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCOPES = (
    Path("clark_quantize"),
    Path("experiments/qwen38_27b_spot"),
    Path("scripts/spot"),
    Path("requirements.txt"),
)


def build_manifest(repo: Path) -> dict:
    entries = []
    for scope in SCOPES:
        root = repo / scope
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            content = path.read_bytes()
            entries.append({
                "path": str(path.relative_to(repo)),
                "bytes": len(content),
                "mode": path.stat().st_mode & 0o777,
                "sha256": hashlib.sha256(content).hexdigest(),
            })
    entries.sort(key=lambda item: item["path"])
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "clark_quant_scoped_source_v1",
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/workspace/clark-quantize"))
    parser.add_argument("--digest-only", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args.repo.resolve())
    print(manifest["manifest_sha256"] if args.digest_only else json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
