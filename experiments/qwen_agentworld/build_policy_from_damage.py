"""Build a behavioral low-bit policy from measured group-damage rows.

This is intentionally conservative. It can either reset groups to BF16 and
enable only measured-safe binary/ternary groups, or materialize the source
policy's currently active groups for one bucket/rate and prune measured-bad
groups. The output is for fake-quant/system-performance validation before
packing, not a claim that the candidate already reaches 8x.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


TERNARY_BPW = 64.0 / 40.0


def load_damage_rows(paths: list[Path], *, metric: str, rate: str, bucket: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text())
        for row in data.get("rows", []):
            if row.get("missing"):
                continue
            if row.get("rate") != rate:
                continue
            if row.get("bucket") != bucket:
                continue
            if metric not in row:
                continue
            group = str(row["group"])
            value = float(row[metric])
            previous = rows.get(group)
            if previous is None or value < float(previous[metric]):
                kept = dict(row)
                kept["damage_source"] = str(path)
                rows[group] = kept
    return rows


def param_count(choice: dict[str, Any]) -> int:
    meta = choice.get("meta") or {}
    return int(meta.get("params", 0))


def bits_for(choice: dict[str, Any], *, rate: str, group_size: int, scale_bits: int) -> float:
    params = float(param_count(choice))
    if rate == "bf16":
        return params * 16.0
    if rate == "fp8":
        return params * 8.0
    if rate == "ternary40_log8":
        return params * (TERNARY_BPW + scale_bits / group_size)
    if rate == "binary_log8":
        return params * (1.0 + scale_bits / group_size)
    raise ValueError(f"unsupported rate: {rate}")


def summarize(choices: list[dict[str, Any]], reference_bits: float) -> dict[str, Any]:
    by_rate: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "bits": 0.0, "params": 0.0})
    by_bucket: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "bits": 0.0, "params": 0.0})
    chosen_bits = 0.0
    for choice in choices:
        meta = choice.get("meta") or {}
        bucket = str(meta.get("bucket", "unknown"))
        rate = str(choice["rate"])
        bits = float(choice["bits"])
        params = float(meta.get("params", 0))
        chosen_bits += bits
        by_rate[rate]["groups"] += 1
        by_rate[rate]["bits"] += bits
        by_rate[rate]["params"] += params
        by_bucket[bucket]["groups"] += 1
        by_bucket[bucket]["bits"] += bits
        by_bucket[bucket]["params"] += params
    return {
        "chosen_bits": chosen_bits,
        "chosen_bytes": chosen_bits / 8.0,
        "chosen_gib": chosen_bits / 8.0 / 1024**3,
        "reference_bits": reference_bits,
        "compression": reference_bits / chosen_bits if chosen_bits else math.inf,
        "by_rate": {k: dict(v) for k, v in sorted(by_rate.items())},
        "by_bucket": {k: dict(v) for k, v in sorted(by_bucket.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-policy", required=True, type=Path)
    parser.add_argument("--damage", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rate", default="ternary40_log8", choices=["ternary40_log8", "binary_log8"])
    parser.add_argument("--bucket", default="moe_expert")
    parser.add_argument("--metric", default="kl_mean")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-metric", type=float)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--scale-bits", type=int)
    parser.add_argument(
        "--base-mode",
        default="reset",
        choices=["reset", "source-filtered"],
        help=(
            "reset: all groups become BF16 except measured-safe selected groups. "
            "source-filtered: source groups matching --bucket/--rate stay active, "
            "then measured groups not selected are pruned to BF16."
        ),
    )
    args = parser.parse_args()

    source = json.loads(args.source_policy.read_text())
    group_size = int(args.group_size or source.get("group_size", 112))
    scale_bits = int(args.scale_bits or source.get("scale_bits", 8))
    measured = load_damage_rows(args.damage, metric=args.metric, rate=args.rate, bucket=args.bucket)

    ranked = sorted(measured.values(), key=lambda row: (float(row[args.metric]), str(row["group"])))
    if args.max_metric is not None:
        ranked = [row for row in ranked if float(row[args.metric]) <= args.max_metric]
    if args.max_groups is not None:
        ranked = ranked[: args.max_groups]
    selected = {str(row["group"]): row for row in ranked}

    choices = []
    for choice in source["choices"]:
        meta = dict(choice.get("meta") or {})
        group = str(choice["group"])
        if group in selected:
            row = selected[group]
            new_meta = dict(meta)
            new_meta["selected_from_damage"] = {
                "metric": args.metric,
                "value": float(row[args.metric]),
                "source": row["damage_source"],
                "relative_rmse": row.get("relative_rmse"),
                "top1_agreement": row.get("top1_agreement"),
            }
            choices.append(
                {
                    **choice,
                    "rate": args.rate,
                    "bits": bits_for(choice, rate=args.rate, group_size=group_size, scale_bits=scale_bits),
                    "damage": float(row[args.metric]),
                    "meta": new_meta,
                }
            )
        elif (
            args.base_mode == "source-filtered"
            and meta.get("bucket") == args.bucket
            and choice.get("rate") == args.rate
            and group not in measured
        ):
            choices.append(
                {
                    **choice,
                    "rate": args.rate,
                    "bits": bits_for(choice, rate=args.rate, group_size=group_size, scale_bits=scale_bits),
                    "meta": meta,
                }
            )
        else:
            choices.append(
                {
                    **choice,
                    "rate": "bf16",
                    "bits": bits_for(choice, rate="bf16", group_size=group_size, scale_bits=scale_bits),
                    "damage": 0.0,
                    "meta": meta,
                }
            )

    reference_bits = (source.get("summary") or {}).get("reference_bits")
    if reference_bits is None:
        reference_bits = sum(bits_for(choice, rate="bf16", group_size=group_size, scale_bits=scale_bits) for choice in source["choices"])

    result = {
        "source_policy": str(args.source_policy),
        "damage_sources": [str(path) for path in args.damage],
        "metric": args.metric,
        "max_metric": args.max_metric,
        "max_groups": args.max_groups,
        "rate": args.rate,
        "bucket": args.bucket,
        "group_size": group_size,
        "scale_bits": scale_bits,
        "base_mode": args.base_mode,
        "measured_groups": len(measured),
        "selected_groups": len(selected),
        "selected": [
            {
                "group": str(row["group"]),
                "metric": float(row[args.metric]),
                "source": row["damage_source"],
                "relative_rmse": row.get("relative_rmse"),
                "top1_agreement": row.get("top1_agreement"),
            }
            for row in ranked
        ],
        "summary": summarize(choices, reference_bits=float(reference_bits)),
        "choices": choices,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "measured_groups": len(measured),
                "selected_groups": len(selected),
                "compression": result["summary"]["compression"],
                "by_rate": result["summary"]["by_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
