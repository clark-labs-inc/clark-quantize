#!/usr/bin/env python3
"""Prepare MOSS-style JSONL manifests from the public HF evaluation mirrors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_REPOS = {
    "aishell4": "zhaochenyang20/AISHELL4",
    "movies800time": "zhaochenyang20/movies800time",
    "movies800": "zhaochenyang20/movies800",
    "googletime": "zhaochenyang20/googletime",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.data_root / "download_receipt.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, repo_id in DATASET_REPOS.items():
        root = args.data_root / dataset / "validation"
        metadata_path = root / "metadata.jsonl"
        revision = receipt["datasets"][dataset]["revision"]
        rows = []
        for index, line in enumerate(metadata_path.read_text().splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            rel_audio = Path(item["file_name"])
            audio_path = root / rel_audio
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            rows.append(
                {
                    "id": audio_path.stem,
                    "audio": str(audio_path),
                    "reference": item["transcription"],
                    "source": repo_id,
                    "source_revision": revision,
                    "split": "validation",
                    "canonical_moss_claim": "unverified_public_mirror",
                    "release_policy": "do_not_redistribute_audio_or_reference",
                    "row_index": index,
                }
            )
        out_path = args.out_dir / f"{dataset}.jsonl"
        with out_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{dataset}: {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
