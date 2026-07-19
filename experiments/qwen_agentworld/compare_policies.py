"""Compare two Clark Air rate-policy JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_choices(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    return {row["group"]: row for row in data["choices"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    base_data = json.loads(args.base.read_text())
    candidate_data = json.loads(args.candidate.read_text())
    base = load_choices(args.base)
    candidate = load_choices(args.candidate)

    changes = []
    changed_by_rate: dict[str, int] = {}
    for group in sorted(set(base) | set(candidate)):
        base_row = base.get(group)
        candidate_row = candidate.get(group)
        base_rate = base_row.get("rate") if base_row else None
        candidate_rate = candidate_row.get("rate") if candidate_row else None
        if base_rate == candidate_rate:
            continue
        key = f"{base_rate}->{candidate_rate}"
        changed_by_rate[key] = changed_by_rate.get(key, 0) + 1
        changes.append(
            {
                "group": group,
                "base_rate": base_rate,
                "candidate_rate": candidate_rate,
                "base_damage": base_row.get("damage") if base_row else None,
                "candidate_damage": candidate_row.get("damage") if candidate_row else None,
                "candidate_meta": candidate_row.get("meta") if candidate_row else None,
            }
        )

    result = {
        "base_policy": str(args.base),
        "candidate_policy": str(args.candidate),
        "base_summary": base_data.get("summary"),
        "candidate_summary": candidate_data.get("summary"),
        "candidate_measured_overrides": candidate_data.get("measured_overrides"),
        "changed_groups": len(changes),
        "changed_by_rate": changed_by_rate,
        "changes": changes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "candidate_measured_overrides": result["candidate_measured_overrides"],
                "changed_groups": result["changed_groups"],
                "changed_by_rate": changed_by_rate,
                "candidate_compression": (candidate_data.get("summary") or {}).get("compression"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
