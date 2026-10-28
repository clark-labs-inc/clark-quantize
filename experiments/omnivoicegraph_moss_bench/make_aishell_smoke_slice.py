#!/usr/bin/env python3
"""Create a small, deterministic AISHELL-4-derived CPU smoke slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for line in args.manifest.read_text().splitlines()[: args.rows]:
        row = json.loads(line)
        info = sf.info(row["audio"])
        frames = min(info.frames, int(args.seconds * info.samplerate))
        audio, sr = sf.read(row["audio"], start=0, stop=frames, dtype="float32", always_2d=False)
        out_audio = args.out / f"{row['id']}-{int(args.seconds)}s.flac"
        sf.write(out_audio, audio, sr, format="FLAC", subtype="PCM_16")
        # Keep only reference segments wholly or partly inside the excerpt.
        import re
        segments = re.findall(r"\[([0-9.]+)\]\[(S\d+)\](.*?)\[([0-9.]+)\](?=\[[0-9.]+\]\[S\d+\]|$)", row["reference"])
        ref = []
        for start, speaker, text, end in segments:
            start_f = float(start)
            if start_f >= args.seconds:
                continue
            end_f = min(args.seconds, float(end))
            ref.append(f"[{start_f:.3f}][{speaker}]{text.strip()}[{end_f:.3f}]")
        out_rows.append({"id": f"{row['id']}-{int(args.seconds)}s", "audio": str(out_audio), "reference": "".join(ref), "source": row["source"], "derived_from": row["id"], "duration": args.seconds, "canonical": False})
    (args.out / "manifest.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out_rows) + "\n")
    (args.out / "receipt.json").write_text(json.dumps({"schema": "omnivoicegraph.moss_derived_smoke_slice.v1", "rows": len(out_rows), "seconds": args.seconds, "canonical": False}, indent=2) + "\n")
    print(json.dumps({"rows": len(out_rows), "seconds": args.seconds, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
