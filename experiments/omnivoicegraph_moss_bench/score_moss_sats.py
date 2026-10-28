#!/usr/bin/env python3
"""Score MOSS-style speaker-attributed transcripts.

The scorer is deliberately format-agnostic: callers provide JSONL rows with
``reference`` and ``hypothesis`` strings in MOSS's timestamp/speaker format.
It emits CER, minimum-permutation cpCER, and delta_cp in percent.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import unicodedata
from pathlib import Path

TAG_RE = re.compile(r"\[[^\]]*\]")
SEGMENT_RE = re.compile(
    r"\[(?P<start>[^\]]+)\]\[(?P<speaker>S\d+)\](?P<text>.*?)(?=\[[^\]]+\]\[S\d+\]|$)",
    re.DOTALL,
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Keep CJK, letters, digits; MOSS CER is character-level and ignores
    # markup/punctuation/whitespace. This policy is recorded in each receipt.
    return "".join(ch.lower() for ch in text if ch.isalnum() or "\u3400" <= ch <= "\u9fff")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def parse_segments(text: str) -> list[tuple[str, str, str]]:
    out = []
    for m in SEGMENT_RE.finditer(text):
        body = m.group("text").strip()
        # The final segment ends with a bare end-time marker rather than a
        # following speaker marker; do not score that timestamp as content.
        body = re.sub(r"\[[^\]]+\]\s*$", "", body)
        out.append((m.group("start"), m.group("speaker"), normalize(body)))
    return out


def stream_text(segments: list[tuple[str, str, str]]) -> str:
    return "".join(t for _, _, t in sorted(segments, key=lambda x: float(x[0]) if x[0].replace('.', '', 1).isdigit() else 0.0))


def speaker_streams(segments: list[tuple[str, str, str]]) -> dict[str, str]:
    streams: dict[str, list[tuple[str, str]]] = {}
    for start, speaker, text in segments:
        streams.setdefault(speaker, []).append((start, text))
    return {k: "".join(t for _, t in sorted(v, key=lambda x: float(x[0]) if x[0].replace('.', '', 1).isdigit() else 0.0)) for k, v in streams.items()}


def score_row(reference: str, hypothesis: str) -> dict[str, float | int]:
    ref_segments = parse_segments(reference)
    hyp_segments = parse_segments(hypothesis)
    # If a reference is plain text, treat it as one stream; MOSS outputs are
    # expected to carry speaker markers, so this is only a safe fallback.
    ref_text = stream_text(ref_segments) if ref_segments else normalize(TAG_RE.sub("", reference))
    hyp_text = stream_text(hyp_segments) if hyp_segments else normalize(TAG_RE.sub("", hypothesis))
    cer_edits = edit_distance(ref_text, hyp_text)
    cer = 100.0 * cer_edits / max(1, len(ref_text))
    ref_spk = speaker_streams(ref_segments) or {"S00": ref_text}
    hyp_spk = speaker_streams(hyp_segments) or {"S00": hyp_text}
    ref_items = list(ref_spk.values())
    hyp_items = list(hyp_spk.values())
    best = None
    for perm in itertools.permutations(range(len(hyp_items)), min(len(ref_items), len(hyp_items))):
        matched = sum(edit_distance(ref_items[i], hyp_items[j]) for i, j in enumerate(perm))
        matched += sum(len(x) for x in ref_items[len(perm):])
        matched += sum(len(x) for i, x in enumerate(hyp_items) if i not in perm)
        best = matched if best is None else min(best, matched)
    cpc = 100.0 * (best or 0) / max(1, sum(len(x) for x in ref_items))
    return {"ref_chars": len(ref_text), "cer_edits": cer_edits, "CER": cer, "cpCER": cpc, "delta_cp": cpc - cer}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=Path, help="JSONL with reference and hypothesis fields")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scores = []
    for line in args.rows.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            scores.append({"id": row.get("id"), **score_row(row["reference"], row["hypothesis"])})
    totals = {"ref_chars": sum(x["ref_chars"] for x in scores), "cer_edits": sum(x["cer_edits"] for x in scores)}
    # Aggregate CER/cpCER by total edit count, matching the corpus-level intent.
    summary = {
        "schema": "omnivoicegraph.moss_sats_score.v1",
        "normalization": "NFKC; lowercase; alphanumeric and CJK characters only",
        "rows": len(scores),
        "CER": 100.0 * totals["cer_edits"] / max(1, totals["ref_chars"]),
        "cpCER": sum(x["cpCER"] for x in scores) / max(1, len(scores)),
        "delta_cp": sum(x["delta_cp"] for x in scores) / max(1, len(scores)),
        "per_row": scores,
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("rows", "CER", "cpCER", "delta_cp")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
