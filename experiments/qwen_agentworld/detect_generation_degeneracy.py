"""Detect obvious long-horizon generation collapse before paid judging.

This is intentionally cheap and model-free. It catches the failure modes seen
in low-bit AgentWorld smokes: blank output, repeated punctuation, repeated
zeros, quote runs, and tiny-character-set loops.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import string
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def longest_char_run(text: str) -> int:
    if not text:
        return 0
    best = 1
    current = 1
    previous = text[0]
    for char in text[1:]:
        if char == previous:
            current += 1
            best = max(best, current)
        else:
            previous = char
            current = 1
    return best


def repeated_ngram_fraction(text: str, ngram: int) -> float:
    if len(text) < ngram or ngram <= 0:
        return 0.0
    counts = collections.Counter(text[i : i + ngram] for i in range(len(text) - ngram + 1))
    top_count = max(counts.values(), default=0)
    return min(1.0, (top_count * ngram) / max(len(text), 1))


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = collections.Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def analyze_text(
    text: str,
    *,
    min_len: int,
    dominant_char_threshold: float,
    max_run_threshold: float,
    unique_char_threshold: float,
    tiny_unique_chars: int,
    low_entropy_threshold: float,
    repeated_ngram_threshold: float,
    low_alnum_threshold: float,
    high_punct_threshold: float,
    ngram: int,
) -> dict[str, Any]:
    length = len(text)
    stripped = text.strip()
    chars = collections.Counter(text)
    top_char, top_count = chars.most_common(1)[0] if chars else ("", 0)
    whitespace = sum(1 for char in text if char.isspace())
    alnum = sum(1 for char in text if char.isalnum())
    punct = sum(1 for char in text if char in string.punctuation)
    run = longest_char_run(text)
    repeat_frac = repeated_ngram_fraction(text, ngram)

    max_char_fraction = top_count / max(length, 1)
    max_run_fraction = run / max(length, 1)
    unique_char_fraction = len(chars) / max(length, 1)
    whitespace_fraction = whitespace / max(length, 1)
    alnum_fraction = alnum / max(length, 1)
    punct_fraction = punct / max(length, 1)
    entropy = char_entropy(text)

    reasons: list[str] = []
    if not stripped:
        reasons.append("empty_or_whitespace")
    if length >= min_len and max_char_fraction >= dominant_char_threshold:
        reasons.append("dominant_single_character")
    if length >= min_len and max_run_fraction >= max_run_threshold:
        reasons.append("long_single_character_run")
    if length >= min_len and (
        len(chars) <= tiny_unique_chars
        or (unique_char_fraction <= unique_char_threshold and entropy <= low_entropy_threshold)
    ):
        reasons.append("tiny_character_set")
    if length >= min_len and repeat_frac >= repeated_ngram_threshold:
        reasons.append("repeated_character_ngram")
    if length >= min_len and alnum_fraction <= low_alnum_threshold and punct_fraction >= high_punct_threshold:
        reasons.append("punctuation_without_content")

    return {
        "len": length,
        "stripped_len": len(stripped),
        "top_char": top_char,
        "top_char_repr": repr(top_char),
        "max_char_fraction": max_char_fraction,
        "max_run": run,
        "max_run_fraction": max_run_fraction,
        "unique_chars": len(chars),
        "unique_char_fraction": unique_char_fraction,
        "entropy_bits": entropy,
        "whitespace_fraction": whitespace_fraction,
        "alnum_fraction": alnum_fraction,
        "punct_fraction": punct_fraction,
        "repeated_ngram_fraction": repeat_frac,
        "collapsed": bool(reasons),
        "collapse_reasons": reasons,
    }


def row_label(row: dict[str, Any]) -> str:
    return str(row.get("task") or row.get("id") or row.get("task_id") or row.get("domain") or "?")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-len", type=int, default=64)
    parser.add_argument("--ngram", type=int, default=8)
    parser.add_argument("--fail-fraction", type=float, default=0.25)
    parser.add_argument("--dominant-char-threshold", type=float, default=0.65)
    parser.add_argument("--max-run-threshold", type=float, default=0.35)
    parser.add_argument("--unique-char-threshold", type=float, default=0.03)
    parser.add_argument("--tiny-unique-chars", type=int, default=8)
    parser.add_argument("--low-entropy-threshold", type=float, default=1.5)
    parser.add_argument("--repeated-ngram-threshold", type=float, default=0.70)
    parser.add_argument("--low-alnum-threshold", type=float, default=0.02)
    parser.add_argument("--high-punct-threshold", type=float, default=0.80)
    parser.add_argument("--compare-prefix", type=int, default=4000)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions)
    baseline = load_jsonl(args.baseline) if args.baseline else None
    if baseline is not None and len(baseline) != len(predictions):
        raise ValueError(f"row count mismatch: baseline={len(baseline)} predictions={len(predictions)}")

    rows = []
    for index, row in enumerate(predictions):
        gen = row.get("gen") or ""
        metrics = analyze_text(
            gen,
            min_len=args.min_len,
            dominant_char_threshold=args.dominant_char_threshold,
            max_run_threshold=args.max_run_threshold,
            unique_char_threshold=args.unique_char_threshold,
            tiny_unique_chars=args.tiny_unique_chars,
            low_entropy_threshold=args.low_entropy_threshold,
            repeated_ngram_threshold=args.repeated_ngram_threshold,
            low_alnum_threshold=args.low_alnum_threshold,
            high_punct_threshold=args.high_punct_threshold,
            ngram=args.ngram,
        )
        out_row: dict[str, Any] = {
            "index": index,
            "task": row_label(row),
            **metrics,
            "gen_prefix": gen[:240],
        }
        if baseline is not None:
            base_gen = baseline[index].get("gen") or ""
            out_row["baseline_len"] = len(base_gen)
            out_row["baseline_sequence_ratio"] = difflib.SequenceMatcher(
                None,
                base_gen[: args.compare_prefix],
                gen[: args.compare_prefix],
            ).ratio()
        rows.append(out_row)

    collapsed_count = sum(1 for row in rows if row["collapsed"])
    count = len(rows)
    collapsed_fraction = collapsed_count / max(count, 1)
    result = {
        "predictions": str(args.predictions),
        "baseline": str(args.baseline) if args.baseline else None,
        "thresholds": {
            "min_len": args.min_len,
            "ngram": args.ngram,
            "fail_fraction": args.fail_fraction,
            "dominant_char_threshold": args.dominant_char_threshold,
            "max_run_threshold": args.max_run_threshold,
            "unique_char_threshold": args.unique_char_threshold,
            "tiny_unique_chars": args.tiny_unique_chars,
            "low_entropy_threshold": args.low_entropy_threshold,
            "repeated_ngram_threshold": args.repeated_ngram_threshold,
            "low_alnum_threshold": args.low_alnum_threshold,
            "high_punct_threshold": args.high_punct_threshold,
        },
        "rows": rows,
        "summary": {
            "count": count,
            "collapsed": collapsed_count,
            "collapsed_fraction": collapsed_fraction,
            "failed_gate": collapsed_fraction >= args.fail_fraction,
            "avg_len": sum(row["len"] for row in rows) / max(count, 1),
            "avg_entropy_bits": sum(row["entropy_bits"] for row in rows) / max(count, 1),
            "avg_unique_char_fraction": sum(row["unique_char_fraction"] for row in rows) / max(count, 1),
            "avg_repeated_ngram_fraction": sum(row["repeated_ngram_fraction"] for row in rows) / max(count, 1),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
