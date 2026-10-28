#!/usr/bin/env python3
"""Fetch the public AliMeeting reference transcripts used by DiariST.

OpenSLR publishes the AliMeeting audio and describes the corpus as having
high-quality transcriptions, but the small Hugging Face mirror we inspected
contains only audio/timestamps/speaker activity.  DiariST's preparation script
publishes the matching reference files from Microsoft Research.  This helper
keeps that reference layer outside git, converts it to the MOSS timestamped
format, and records hashes and split cardinalities for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path


SOURCES = {
    "train": "https://www.microsoft.com/en-us/research/uploads/prod/2023/09/train.txt",
    "dev": "https://www.microsoft.com/en-us/research/uploads/prod/2023/09/dev.txt",
    "test": "https://www.microsoft.com/en-us/research/uploads/prod/2023/09/test.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "omnivoicegraph-benchmark/1.0"})
    with urllib.request.urlopen(request) as response, path.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)


def parse_reference(path: Path) -> tuple[list[dict], dict[str, dict]]:
    sessions: OrderedDict[str, dict] = OrderedDict()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 6:
                raise ValueError(f"{path}:{line_number}: expected 6 tab-separated fields, got {len(fields)}")
            recording, speaker, start, end, chinese, english = fields
            if not chinese:
                raise ValueError(f"{path}:{line_number}: empty Chinese reference")
            row = sessions.setdefault(
                recording,
                {"id": recording, "speaker_map": OrderedDict(), "segments": []},
            )
            speaker_map = row["speaker_map"]
            if speaker not in speaker_map:
                speaker_map[speaker] = f"S{len(speaker_map) + 1:02d}"
            row["segments"].append(
                {
                    "start": float(start),
                    "end": float(end),
                    "speaker": speaker,
                    "speaker_normalized": speaker_map[speaker],
                    "text_zh": chinese,
                    "text_en": english,
                }
            )

    output: list[dict] = []
    for row in sessions.values():
        segments = row["segments"]
        reference = "".join(
            f"[{segment['start']:.3f}][{segment['speaker_normalized']}]"
            f"{segment['text_zh']}[{segment['end']:.3f}]"
            for segment in segments
        )
        output.append(
            {
                "id": row["id"],
                "reference": reference,
                "segments": segments,
                "speaker_ids_original": list(row["speaker_map"]),
                "speaker_ids_normalized": list(row["speaker_map"].values()),
                "utterances": len(segments),
                "source": "DiariST-AliMeeting",
                "split": path.stem,
                "reference_language": "zh",
            }
        )
    return output, {row["id"]: row for row in output}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(SOURCES), nargs="+", default=["test"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "omnivoicegraph.alimeeting_reference_receipt.v1",
        "created_unix": int(time.time()),
        "license": "CC BY-SA 4.0 (DiariST-AliMeeting; verify original AliMeeting terms before redistribution)",
        "source_repository": "https://github.com/Mu-Y/DiariST",
        "source_script": "diarist/data/download_alimeeting_translation.sh",
        "sources": {},
        "note": "Chinese reference text is used for CER/cpCER; English translation is retained only in raw source files.",
    }

    for split in args.split:
        raw_path = args.out_dir / f"diarist_alimeeting_{split}.txt"
        fetch(SOURCES[split], raw_path)
        rows, by_id = parse_reference(raw_path)
        manifest_path = args.out_dir / f"diarist_alimeeting_{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        receipt["sources"][split] = {
            "url": SOURCES[split],
            "raw_path": str(raw_path),
            "raw_sha256": sha256(raw_path),
            "raw_bytes": raw_path.stat().st_size,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "sessions": len(rows),
            "utterances": sum(row["utterances"] for row in rows),
            "speaker_ids": len({speaker for row in rows for speaker in row["speaker_ids_original"]}),
            "session_ids": [row["id"] for row in rows],
        }

    receipt_path = args.out_dir / "reference_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
