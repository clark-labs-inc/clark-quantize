"""Cheap non-judge comparison for AgentWorld prediction files."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compare-prefix", type=int, default=4000)
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    candidate = load_jsonl(args.candidate)
    if len(baseline) != len(candidate):
        raise ValueError(f"row count mismatch: {len(baseline)} vs {len(candidate)}")

    rows = []
    for base, cand in zip(baseline, candidate):
        base_gen = base.get("gen") or ""
        cand_gen = cand.get("gen") or ""
        rows.append(
            {
                "task": base.get("task"),
                "id": base.get("id"),
                "baseline_len": len(base_gen),
                "candidate_len": len(cand_gen),
                "baseline_empty": not bool(base_gen),
                "candidate_empty": not bool(cand_gen),
                "exact": base_gen == cand_gen,
                "sequence_ratio": difflib.SequenceMatcher(
                    None,
                    base_gen[: args.compare_prefix],
                    cand_gen[: args.compare_prefix],
                ).ratio(),
            }
        )

    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "rows": rows,
        "summary": {
            "count": len(rows),
            "baseline_empty": sum(1 for row in rows if row["baseline_empty"]),
            "candidate_empty": sum(1 for row in rows if row["candidate_empty"]),
            "exact_matches": sum(1 for row in rows if row["exact"]),
            "avg_sequence_ratio": sum(row["sequence_ratio"] for row in rows) / max(len(rows), 1),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
