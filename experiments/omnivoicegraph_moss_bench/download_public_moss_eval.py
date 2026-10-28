#!/usr/bin/env python3
"""Download public MOSS evaluation mirrors with pinned revisions and receipts.

This intentionally stores the audio outside the git checkout.  The generated
receipt is suitable for a release manifest; the audio itself remains subject
to each upstream dataset's terms and is not copied into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DATASETS = {
    "aishell4": "zhaochenyang20/AISHELL4",
    "movies800time": "zhaochenyang20/movies800time",
    "movies800": "zhaochenyang20/movies800",
    "googletime": "zhaochenyang20/googletime",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*", choices=sorted(DATASETS))
    args = parser.parse_args()

    api = HfApi()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = args.only or list(DATASETS)
    receipt_path = args.out_dir / "download_receipt.json"
    if args.only and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        receipt["created_unix"] = int(time.time())
        receipt.setdefault("datasets", {})
    else:
        receipt = {
            "schema": "omnivoicegraph.moss_public_eval_download_receipt.v1",
            "created_unix": int(time.time()),
            "datasets": {},
            "note": (
                "Public mirrors used for local reproducibility. Audio and references "
                "are not redistributed by this repository; verify upstream terms."
            ),
        }

    for name in wanted:
        repo_id = DATASETS[name]
        info = api.dataset_info(repo_id)
        revision = info.sha
        dataset_dir = args.out_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        entries = list(api.list_repo_tree(repo_id, repo_type="dataset", revision=revision, recursive=True))
        files = [entry for entry in entries if entry.__class__.__name__ == "RepoFile"]
        files.sort(key=lambda entry: entry.path)
        manifest_files = []
        for entry in files:
            print(f"[{name}] {entry.path} ({entry.size} bytes)", flush=True)
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=entry.path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(dataset_dir),
                )
            )
            manifest_files.append(
                {
                    "path": entry.path,
                    "size": entry.size,
                    "sha256": sha256(local_path),
                }
            )
        receipt["datasets"][name] = {
            "repo_id": repo_id,
            "revision": revision,
            "files": manifest_files,
            "bytes": sum(item["size"] for item in manifest_files),
        }

    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {receipt_path}")


if __name__ == "__main__":
    main()
