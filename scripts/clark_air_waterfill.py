"""Run Clark Air water-filling rate allocation from a JSON candidate table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clark_air.rate import RateCandidate, choose_rates


def load_candidates(path: Path) -> list[RateCandidate]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("candidates", [])
    candidates = []
    for row in rows:
        group = row.get("group", row.get("group_id"))
        if group is None:
            raise ValueError(f"candidate row missing group/group_id: {row}")
        candidates.append(
            RateCandidate(
                group=str(group),
                rate=str(row["rate"]),
                bits=float(row["bits"]),
                damage=float(row["damage"]),
                meta=row.get("meta"),
            )
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--target-bits", required=True, type=float)
    parser.add_argument("--out", default="clark_air_policy.json", type=Path)
    args = parser.parse_args()

    policy = choose_rates(load_candidates(args.candidates), args.target_bits)
    result = {
        "target_bits": policy.target_bits,
        "total_bits": policy.total_bits,
        "total_damage": policy.total_damage,
        "bit_price": policy.bit_price,
        "feasible": policy.feasible,
        "choices": policy.as_rows(),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"feasible={policy.feasible} "
        f"bits={policy.total_bits:.0f}/{policy.target_bits:.0f} "
        f"damage={policy.total_damage:.6g} "
        f"lambda={policy.bit_price:.6g}"
    )


if __name__ == "__main__":
    main()
