#!/usr/bin/env python3
"""Join official AliMeeting far-field test audio to DiariST references.

The official Test_Ali archive names each far-field mixture as
``Rxxxx_Myyyy_MSzzz.wav`` while DiariST's reference file uses the
``Rxxxx_Myyyy`` session ID.  This creates the 20-row MOSS manifest only after
both sides are present and fails closed on missing or duplicate sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import wave
from pathlib import Path


SESSION_RE = re.compile(r"^(R\d+_M\d+)_MS\d+\.wav$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate
    except ImportError:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / handle.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    references = {}
    for line in args.reference_manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            references[row["id"]] = row
    audio_by_session = {}
    for path in sorted(args.audio_root.glob("*.wav")):
        match = SESSION_RE.match(path.name)
        if not match:
            continue
        session = match.group(1)
        if session in audio_by_session:
            raise SystemExit(f"duplicate audio for {session}: {audio_by_session[session]} and {path}")
        audio_by_session[session] = path

    missing_audio = sorted(set(references) - set(audio_by_session))
    extra_audio = sorted(set(audio_by_session) - set(references))
    if missing_audio or extra_audio:
        raise SystemExit(json.dumps({"missing_audio": missing_audio, "extra_audio": extra_audio}, indent=2))

    rows = []
    audio_receipt = []
    for session in sorted(references):
        path = audio_by_session[session]
        reference = references[session]
        rows.append(
            {
                "id": session,
                "audio": str(path),
                "duration": wav_duration(path),
                "reference": reference["reference"],
                "source": "AliMeeting/Test_Ali/Test_Ali_far",
                "reference_source": "DiariST-AliMeeting",
                "split": "test",
                "canonical_moss_claim": "official_audio_diarist_reference_join_verified",
                "release_policy": "do_not_redistribute_without_upstream_confirmation",
            }
        )
        audio_receipt.append({"id": session, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    receipt = {
        "schema": "omnivoicegraph.alimeeting_official_manifest_receipt.v1",
        "created_unix": int(time.time()),
        "manifest": str(args.out),
        "reference_manifest": str(args.reference_manifest),
        "audio_root": str(args.audio_root),
        "rows": len(rows),
        "total_audio_bytes": sum(item["bytes"] for item in audio_receipt),
        "audio": audio_receipt,
    }
    receipt_path = args.out.with_name("alimeeting_official_manifest_receipt.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "total_audio_bytes": receipt["total_audio_bytes"], "receipt": str(receipt_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
