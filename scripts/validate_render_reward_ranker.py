"""Validate render reward heads against known pairwise visual preferences."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def row_value(row: dict, goal: str) -> float:
    if goal in ("overall", "score"):
        return float(row["score"])
    if goal in row.get("head_mean", {}):
        return float(row["head_mean"][goal])
    if goal in row.get("mean", {}):
        return float(row["mean"][goal])
    raise KeyError(f"goal {goal!r} not present for {row.get('label')}")


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    rx = ranks(xs)
    ry = ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return float(num / max(dx * dy, 1e-12))


def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="JSON from score_render_grid.py")
    ap.add_argument("--preferences", required=True, help="Pairwise preference JSON")
    ap.add_argument("--out-json", default="render_reward_rank_validation.json")
    ap.add_argument("--out-md", default="render_reward_rank_validation.md")
    ap.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args()

    scores = load_json(args.scores)
    prefs = load_json(args.preferences).get("preferences", [])
    rows = [r for r in scores.get("rows", []) if "error" not in r]
    by_label = {r["label"]: r for r in rows}

    results = []
    passed = 0
    missing = 0
    for pref in prefs:
        goal = pref["goal"]
        better = pref["better"]
        worse = pref["worse"]
        if better not in by_label or worse not in by_label:
            results.append({**pref, "status": "missing", "delta": None})
            missing += 1
            continue
        bv = row_value(by_label[better], goal)
        wv = row_value(by_label[worse], goal)
        delta = bv - wv
        ok = delta > args.margin
        passed += int(ok)
        results.append(
            {
                **pref,
                "better_value": bv,
                "worse_value": wv,
                "delta": delta,
                "status": "pass" if ok else "fail",
            }
        )

    checked = len(results) - missing
    accuracy = passed / checked if checked else 0.0

    # Approximate overall ordering stability from pairwise preferences by asking
    # whether score deltas correlate with signed preferred order.
    deltas = [r["delta"] for r in results if r.get("delta") is not None]
    signs = [1.0 for r in results if r.get("delta") is not None]
    rho = spearman(deltas, signs) if len(set(deltas)) > 1 else 0.0

    payload = {
        "scores": args.scores,
        "preferences": args.preferences,
        "checked": checked,
        "missing": missing,
        "passed": passed,
        "accuracy": accuracy,
        "spearman_preference_delta": rho,
        "margin": args.margin,
        "results": results,
        "ready_for_gradient_training": bool(accuracy >= 0.80 and missing == 0),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        f"checked={checked} passed={passed} missing={missing} accuracy={accuracy:.3f}",
        "",
        "| status | goal | better | worse | delta | note |",
        "|---|---|---|---|---:|---|",
    ]
    for r in results:
        delta = "" if r["delta"] is None else f"{r['delta']:.4f}"
        lines.append(
            f"| {r['status']} | {r['goal']} | `{r['better']}` | `{r['worse']}` | {delta} | {r.get('note','')} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(lines[0])
    for r in results:
        delta = "missing" if r["delta"] is None else f"{r['delta']:.4f}"
        print(f"{r['status']:7s} {r['goal']:12s} delta={delta} {r.get('note','')}")


if __name__ == "__main__":
    main()
