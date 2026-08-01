#!/usr/bin/env python3
"""Extract and fingerprint embedded videos from a pinned Humyn parquet shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as parquet


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def probe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,pix_fmt:format=duration,size",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table = parquet.read_table(args.shard)
    rows = []
    for index in range(table.num_rows):
        video = table["video"][index].as_py()
        payload = video["bytes"]
        if not isinstance(payload, bytes):
            raise TypeError(f"row {index} has no embedded video bytes")
        filename = str(table["file_name"][index].as_py())
        if Path(filename).name != filename:
            raise ValueError(f"unsafe embedded filename: {filename}")
        destination = args.out / filename
        payload_sha = sha256_bytes(payload)
        if destination.exists() and sha256_file(destination) != payload_sha:
            raise FileExistsError(f"existing extracted video differs: {destination}")
        if not destination.exists():
            destination.write_bytes(payload)
        rows.append({
            "row": index,
            "file_name": filename,
            "type": str(table["type"][index].as_py()),
            "bytes": len(payload),
            "sha256": payload_sha,
            "probe": probe(destination),
        })
    receipt = {
        "format": "backdropkit-humyn-video-shard-extraction-v1",
        "source": {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "repo_path": args.repo_path,
            "license": args.license,
            "shard_bytes": args.shard.stat().st_size,
            "shard_sha256": sha256_file(args.shard),
        },
        "videos": rows,
    }
    receipt_path = args.out / "extraction-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
