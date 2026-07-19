"""Select low-damage binary spillover candidates under per-goal constraints.

Input is a JSON file with candidates shaped like:

{
  "candidates": [
    {
      "name": "transformer_blocks.0.attn1.to_q",
      "bits_saved": 123456,
      "delta": {"text": 0.001, "objects": 0.002, "relations": 0.0}
    }
  ]
}

The deltas are value drops: V(ternary) - V(candidate converted to binary).
This script is the safe first use of a visual reward critic: constrained
mixed-rate selection, not policy-gradient training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_WEIGHTS = {
    "teacher": 0.20,
    "text": 0.18,
    "objects": 0.18,
    "relations": 0.20,
    "geometry": 0.14,
    "composition": 0.10,
}


def parse_json_arg(value: str | None, default: dict) -> dict:
    if not value:
        return dict(default)
    p = Path(value)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(value)


def damage(candidate: dict, weights: dict[str, float]) -> float:
    d = candidate.get("delta", {})
    return float(sum(float(weights.get(k, 0.0)) * max(0.0, float(d.get(k, 0.0))) for k in weights))


def violates_constraints(totals: dict[str, float], constraints: dict[str, float]) -> list[str]:
    bad = []
    for key, limit in constraints.items():
        if float(totals.get(key, 0.0)) > float(limit):
            bad.append(key)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--target-bits", type=float, required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--constraints", default="")
    ap.add_argument("--out-json", default="binary_spillover_selection.json")
    ap.add_argument("--out-md", default="binary_spillover_selection.md")
    args = ap.parse_args()

    payload = json.loads(Path(args.candidates).read_text())
    candidates = payload.get("candidates", payload if isinstance(payload, list) else [])
    weights = parse_json_arg(args.weights, DEFAULT_WEIGHTS)
    constraints = parse_json_arg(args.constraints, {})

    rows = []
    for cand in candidates:
        bits = float(cand.get("bits_saved", 0.0))
        if bits <= 0:
            continue
        dmg = damage(cand, weights)
        rows.append({**cand, "weighted_damage": dmg, "damage_per_bit": dmg / bits})
    rows.sort(key=lambda r: (r["damage_per_bit"], r["weighted_damage"]))

    selected = []
    skipped = []
    totals = {k: 0.0 for k in weights}
    total_bits = 0.0
    total_damage = 0.0
    for row in rows:
        trial = dict(totals)
        for k, v in row.get("delta", {}).items():
            trial[k] = trial.get(k, 0.0) + max(0.0, float(v))
        bad = violates_constraints(trial, constraints)
        if bad:
            skipped.append({**row, "skip_reason": "constraint:" + ",".join(bad)})
            continue
        selected.append(row)
        totals = trial
        total_bits += float(row.get("bits_saved", 0.0))
        total_damage += float(row["weighted_damage"])
        if total_bits >= args.target_bits:
            break

    feasible = total_bits >= args.target_bits
    result = {
        "source": args.candidates,
        "target_bits": args.target_bits,
        "selected_bits": total_bits,
        "selected_weighted_damage": total_damage,
        "goal_totals": totals,
        "weights": weights,
        "constraints": constraints,
        "feasible": feasible,
        "selected": selected,
        "skipped": skipped,
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        f"feasible={feasible} selected_bits={total_bits:.0f} target_bits={args.target_bits:.0f} weighted_damage={total_damage:.6f}",
        "",
        "| rank | bits | damage | ratio | name |",
        "|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(selected, 1):
        lines.append(
            f"| {i} | {float(row['bits_saved']):.0f} | {row['weighted_damage']:.6f} | "
            f"{row['damage_per_bit']:.3e} | `{row.get('name','')}` |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(lines[0])
    for line in lines[3:13]:
        print(line)


if __name__ == "__main__":
    main()
