#!/usr/bin/env python3
"""Retain at most two complete Qwen QAT checkpoints under one explicit root."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


SCHEMA = "clark_qwen38_qat_checkpoint_v1"
NAME = re.compile(r"checkpoint-(\d+)$")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def completed_checkpoints(root: Path) -> tuple[list[tuple[int, Path]], list[str]]:
    root = root.resolve()
    complete = []
    ignored = []
    for path in sorted(root.iterdir()):
        match = NAME.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != root:
            raise ValueError(f"unsafe checkpoint path: {path}")
        manifest_path = path / "checkpoint.json"
        if not manifest_path.is_file():
            ignored.append(path.name)
            continue
        manifest = json.loads(manifest_path.read_text())
        step = int(match.group(1))
        if (
            manifest.get("schema") != SCHEMA
            or manifest.get("complete") is not True
            or manifest.get("step") != step
        ):
            raise ValueError(f"invalid completed-checkpoint manifest: {manifest_path}")
        complete.append((step, path))
    if len({step for step, _ in complete}) != len(complete):
        raise ValueError("duplicate checkpoint steps")
    return sorted(complete), ignored


def prune(root: Path, keep: int = 2, dry_run: bool = False) -> dict:
    if keep not in (1, 2):
        raise ValueError("Qwen QAT retention must keep one or two checkpoints")
    if not root.is_absolute():
        raise ValueError("checkpoint root must be absolute")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    complete, ignored = completed_checkpoints(root)
    doomed = complete[:-keep]
    retained = complete[-keep:]
    if not dry_run:
        for _, path in doomed:
            if path.parent != root or path.is_symlink():
                raise ValueError(f"refusing to prune unsafe path: {path}")
            shutil.rmtree(path)
    receipt = {
        "schema": "clark_qwen38_qat_retention_v1",
        "root": str(root),
        "keep": keep,
        "dry_run": dry_run,
        "removed": [path.name for _, path in doomed],
        "retained": [path.name for _, path in retained],
        "ignored_incomplete": ignored,
        "policy": "at most two complete checkpoints plus one atomically promoted in-progress save",
    }
    atomic_json(root / "retention.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".retention.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps(prune(args.root, keep=args.keep, dry_run=args.dry_run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
