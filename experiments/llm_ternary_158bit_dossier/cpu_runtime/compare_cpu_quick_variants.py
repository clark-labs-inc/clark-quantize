#!/usr/bin/env python3
"""Compare two exact-artifact quick-eval variants item by item."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    claimed = report.get("report_content_sha256")
    body = {key: value for key, value in report.items() if key != "report_content_sha256"}
    if not claimed or canonical_digest(body) != claimed:
        raise RuntimeError(f"report content hash mismatch: {path}")
    return report


def records_by_id(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in report["records"]:
        key = (row["task"], row["id"])
        if key in records:
            raise RuntimeError(f"duplicate record: {key}")
        records[key] = row
    return records


def compare_report_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if a["manifest_content_sha256"] != b["manifest_content_sha256"]:
        raise RuntimeError("variant manifests differ")
    a_records = records_by_id(a)
    b_records = records_by_id(b)
    if set(a_records) != set(b_records):
        raise RuntimeError("variant item IDs differ")

    differences: list[dict[str, Any]] = []
    for key in sorted(a_records):
        left = a_records[key]
        right = b_records[key]
        if left["gold"] != right["gold"]:
            raise RuntimeError(f"gold answer differs: {key}")
        if (
            left["output"] != right["output"]
            or left["prediction"] != right["prediction"]
            or bool(left["correct"]) != bool(right["correct"])
        ):
            differences.append(
                {
                    "task": key[0],
                    "id": key[1],
                    "gold": left["gold"],
                    "a": {
                        "correct": bool(left["correct"]),
                        "prediction": left["prediction"],
                        "output": left["output"],
                    },
                    "b": {
                        "correct": bool(right["correct"]),
                        "prediction": right["prediction"],
                        "output": right["output"],
                    },
                }
            )

    keys = sorted(a_records)
    return {
        "manifest_content_sha256": a["manifest_content_sha256"],
        "n": len(keys),
        "a_correct": sum(bool(a_records[key]["correct"]) for key in keys),
        "b_correct": sum(bool(b_records[key]["correct"]) for key in keys),
        "a_only_correct": sum(
            bool(a_records[key]["correct"]) and not bool(b_records[key]["correct"])
            for key in keys
        ),
        "b_only_correct": sum(
            bool(b_records[key]["correct"]) and not bool(a_records[key]["correct"])
            for key in keys
        ),
        "exact_output_equal": sum(
            a_records[key]["output"] == b_records[key]["output"] for key in keys
        ),
        "normalized_prediction_equal": sum(
            a_records[key]["prediction"] == b_records[key]["prediction"]
            for key in keys
        ),
        "correctness_equal": sum(
            bool(a_records[key]["correct"]) == bool(b_records[key]["correct"])
            for key in keys
        ),
        "differences": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-label", required=True)
    parser.add_argument("--a-math", required=True, type=Path)
    parser.add_argument("--a-cute", required=True, type=Path)
    parser.add_argument("--b-label", required=True)
    parser.add_argument("--b-math", required=True, type=Path)
    parser.add_argument("--b-cute", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    reports = {
        "a_math": load_report(args.a_math),
        "a_cute": load_report(args.a_cute),
        "b_math": load_report(args.b_math),
        "b_cute": load_report(args.b_cute),
    }
    if reports["a_math"]["model_sha256"] != reports["a_cute"]["model_sha256"]:
        raise RuntimeError("variant A model differs across reports")
    if reports["b_math"]["model_sha256"] != reports["b_cute"]["model_sha256"]:
        raise RuntimeError("variant B model differs across reports")

    result = {
        "schema": "cpu_quick_variant_comparison_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "a": {
            "label": args.a_label,
            "model_sha256": reports["a_math"]["model_sha256"],
        },
        "b": {
            "label": args.b_label,
            "model_sha256": reports["b_math"]["model_sha256"],
        },
        "cute": compare_report_pair(reports["a_cute"], reports["b_cute"]),
        "math_code": compare_report_pair(reports["a_math"], reports["b_math"]),
        "inputs": {
            "a_math": str(args.a_math.resolve()),
            "a_cute": str(args.a_cute.resolve()),
            "b_math": str(args.b_math.resolve()),
            "b_cute": str(args.b_cute.resolve()),
        },
    }
    result["report_content_sha256"] = canonical_digest(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
