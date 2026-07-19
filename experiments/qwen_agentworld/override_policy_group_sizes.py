"""Write a derived policy with per-bucket group-size overrides.

This is for storage oracles such as "make only MLP tensors ternary g32" while
leaving the parent policy otherwise unchanged.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


TERNARY_CODE_BITS = 64.0 / 40.0


def parse_bucket_group_sizes(values: list[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"group-size override must be BUCKET=SIZE, got {value!r}")
        bucket, size_text = value.split("=", 1)
        bucket = bucket.strip()
        if not bucket:
            raise ValueError(f"empty bucket in group-size override {value!r}")
        size = int(size_text)
        if size <= 0:
            raise ValueError(f"group size must be positive for {bucket!r}: {size}")
        overrides[bucket] = size
    return overrides


def rate_bits(rate: str, params: int, group_size: int) -> float | None:
    if rate == "bf16":
        return float(params * 16)
    if rate == "fp8":
        return float(params * 8)
    if rate == "ternary40_fp16":
        return float(params * (TERNARY_CODE_BITS + 16.0 / group_size))
    if rate == "ternary40_log8":
        return float(params * (TERNARY_CODE_BITS + 8.0 / group_size))
    if rate == "binary_fp16":
        return float(params * (1.0 + 16.0 / group_size))
    if rate == "binary_log8":
        return float(params * (1.0 + 8.0 / group_size))
    return None


def summarize(choices: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, dict[str, float]] = {}
    by_rate: dict[str, dict[str, float]] = {}
    total_bits = 0.0
    for choice in choices:
        meta = choice.get("meta") or {}
        bits = float(choice.get("bits", 0.0))
        total_bits += bits
        for table, key in ((by_bucket, str(meta.get("bucket", "unknown"))), (by_rate, str(choice.get("rate")))):
            row = table.setdefault(key, {"groups": 0, "bits": 0.0, "params": 0.0})
            row["groups"] += 1
            row["bits"] += bits
            row["params"] += int(meta.get("params", 0))
    return {
        "groups": len(choices),
        "bits": total_bits,
        "bytes": total_bits / 8.0,
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_rate": dict(sorted(by_rate.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--bucket-group-size",
        action="append",
        required=True,
        help="override group size for one policy bucket, e.g. mlp=32; repeatable",
    )
    args = parser.parse_args()

    overrides = parse_bucket_group_sizes(args.bucket_group_size)
    policy = json.loads(args.policy.read_text())
    derived = deepcopy(policy)
    default_group_size = int(policy.get("group_size", 128))
    changed = 0
    for choice in derived["choices"]:
        meta = choice.setdefault("meta", {})
        bucket = str(meta.get("bucket"))
        if bucket not in overrides:
            continue
        group_size = overrides[bucket]
        choice["group_size"] = group_size
        meta["group_size"] = group_size
        params = int(meta.get("params", 0))
        bits = rate_bits(str(choice.get("rate")), params, group_size)
        if bits is not None:
            choice["bits"] = bits
        changed += 1

    derived["base_policy"] = str(args.policy)
    derived["bucket_group_sizes"] = {**policy.get("bucket_group_sizes", {}), **overrides}
    derived["default_group_size"] = default_group_size
    derived["summary"] = summarize(derived["choices"])
    derived["note"] = "Derived policy with per-bucket group-size overrides."

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(derived, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "changed_choices": changed,
                "bucket_group_sizes": derived["bucket_group_sizes"],
                "summary": derived["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
