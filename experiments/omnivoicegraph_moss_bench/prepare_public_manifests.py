#!/usr/bin/env python3
"""Materialize deterministic manifests for the public meeting sources.

AISHELL-4's TextGrid files contain transcript text and speaker intervals, so
they can be scored with the MOSS CER/cpCER contract. The public AliMeeting HF
mirror used here exposes audio, timestamps, and speaker IDs but no transcript
field; it is registered as diarization-only until the licensed M2MeT reference
transcripts are supplied.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def textgrid(path: Path) -> tuple[float, list[dict]]:
    lines = path.read_text(errors="replace").splitlines()
    duration = 0.0
    current_speaker = "S00"
    start = end = None
    rows: list[dict] = []
    tier_name = re.compile(r'^\s*name = "([^"]+)"')
    xmin_re = re.compile(r"^\s*xmin = ([0-9.eE+-]+)")
    xmax_re = re.compile(r"^\s*xmax = ([0-9.eE+-]+)")
    text_re = re.compile(r'^\s*text = "(.*)"\s*$')
    for line in lines:
        if m := tier_name.match(line):
            raw = m.group(1)
            current_speaker = "S" + re.sub(r"\D", "", raw).zfill(2)[-2:] if re.search(r"\d", raw) else raw
        elif m := xmin_re.match(line):
            start = float(m.group(1))
        elif m := xmax_re.match(line):
            end = float(m.group(1))
            duration = max(duration, end)
        elif m := text_re.match(line):
            body = m.group(1).replace('\\"', '"').strip()
            body = re.sub(r"<[^>]+>|&", "", body).strip()
            if body and start is not None and end is not None and end > start:
                rows.append({"start": start, "end": end, "speaker": current_speaker, "text": body})
    rows.sort(key=lambda x: (x["start"], x["end"], x["speaker"]))
    return duration, rows


def moss_reference(rows: list[dict]) -> str:
    return "".join(f"[{r['start']:.3f}][{r['speaker']}]{r['text']}[{r['end']:.3f}]" for r in rows)


def prepare_aishell(root: Path, out: Path) -> int:
    count = 0
    wav_dir = root / "test" / "wav"
    tg_dir = root / "test" / "TextGrid"
    with out.open("w") as fh:
        for tg in sorted(tg_dir.glob("*.TextGrid")):
            duration, rows = textgrid(tg)
            audio = wav_dir / (tg.stem + ".flac")
            if not audio.exists():
                continue
            fh.write(json.dumps({"id": tg.stem, "audio": str(audio), "duration": duration, "reference": moss_reference(rows), "source": "AISHELL/AISHELL-4", "split": "test"}, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_alimeeting(root: Path, out: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required for AliMeeting Parquet") from exc
    count = 0
    with out.open("w") as fh:
        for shard in sorted((root / "data").glob("test-*.parquet")):
            table = pq.read_table(shard, columns=["audio", "timestamps_start", "timestamps_end", "speakers"], use_threads=False)
            for index, row in enumerate(table.to_pylist()):
                audio = row.get("audio") or {}
                starts = row.get("timestamps_start") or []
                ends = row.get("timestamps_end") or []
                speakers = row.get("speakers") or []
                fh.write(json.dumps({"id": f"{shard.stem}-{index:06d}", "audio_bytes_in_parquet": True, "audio_path": audio.get("path"), "duration": max(ends, default=0.0), "speaker_activity": [{"start": s, "end": e, "speaker": sp} for s, e, sp in zip(starts, ends, speakers)], "transcript_available": False, "source": "argmaxinc/ali-meetings", "split": "test"}, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aishell", type=Path)
    ap.add_argument("--alimeeting", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.aishell:
        summary["aishell4_rows"] = prepare_aishell(args.aishell, args.out / "aishell4_test.jsonl")
    if args.alimeeting:
        summary["alimeeting_rows"] = prepare_alimeeting(args.alimeeting, args.out / "alimeeting_test_activity_only.jsonl")
    (args.out / "manifest_receipt.json").write_text(json.dumps({"schema": "omnivoicegraph.moss_public_manifest_receipt.v1", **summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
