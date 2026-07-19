"""Verify storage-lossless invariants for Clark Air parent-pack artifacts.

This verifier checks the Prism-style claim "lossless for ternary": packed
ternary codes round-trip exactly through the storage codec, scales are stored in
the advertised dtype, and no out-of-codebook symbols are present. It is
intentionally separate from behavior/RMSE verification against the BF16 parent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TERNARY_RATES = {"ternary40_log8", "ternary40_fp16"}
BINARY_RATES = {"binary_log8", "binary_fp16"}
MAX_TRIT_WORD = 3**40


def ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


def ternary_pad_is_zero_code(last_word: np.uint64, pad: int) -> bool:
    if pad <= 0:
        return True
    value = int(last_word)
    first_pad_digit = 40 - pad
    for digit in range(first_pad_digit, 40):
        if (value // (3**digit)) % 3 != 1:
            return False
    return True


def binary_pad_is_zero(last_byte: np.uint8, pad: int) -> bool:
    if pad <= 0:
        return True
    valid_bits = 8 - pad
    return (int(last_byte) >> valid_bits) == 0


def group_size_for_artifact(artifact: Any, group: dict[str, Any], manifest_group_size: int) -> int:
    if group.get("group_size") is not None:
        return int(group["group_size"])
    prefix_key = f"{group['prefix']}_group_size"
    if hasattr(artifact, "files") and prefix_key in artifact.files:
        return int(artifact[prefix_key][0])
    return int(artifact["group_size"][0]) if "group_size" in artifact.files else manifest_group_size


def verify_ternary_group(
    artifact: Any,
    group: dict[str, Any],
    parent: dict[str, Any],
    manifest_group_size: int,
) -> dict[str, Any]:
    prefix = group["prefix"]
    packed = artifact[f"{prefix}_packed"]
    shape = tuple(int(x) for x in artifact[f"{prefix}_shape"].tolist())
    pad = int(artifact[f"{prefix}_pad"][0])
    scales = artifact[f"{prefix}_scales"]
    code_count = int(np.prod(shape))
    expected_pad = (-code_count) % 40
    expected_packed_words = ceil_div(code_count, 40)
    group_size = group_size_for_artifact(artifact, group, manifest_group_size)
    expected_scale_count = code_count // group_size
    checks = {
        "packed_words_are_canonical_base3": bool((packed.astype(np.uint64) < np.uint64(MAX_TRIT_WORD)).all()),
        "packed_word_count_matches_shape": int(packed.size) == expected_packed_words,
        "packed_pad_matches_shape": pad == expected_pad,
        "packed_pad_trits_are_zero_code": ternary_pad_is_zero_code(packed[-1], pad) if packed.size else True,
        "group_size_positive": group_size > 0,
        "scale_dtype_fp16": scales.dtype == np.float16,
        "scale_count_matches_codes": int(scales.size) == expected_scale_count,
    }
    return {
        "tensor": parent["tensor"],
        "group": group["group"],
        "rate": group["rate"],
        "prefix": prefix,
        "params": group.get("params"),
        "packed_words": int(packed.size),
        "code_shape": list(shape),
        "pad": pad,
        "expected_pad": expected_pad,
        "group_size": group_size,
        "scale_dtype": str(scales.dtype),
        "scale_count": int(scales.size),
        "checks": checks,
        "ok": all(checks.values()),
    }


def verify_binary_group(
    artifact: Any,
    group: dict[str, Any],
    parent: dict[str, Any],
    manifest_group_size: int,
) -> dict[str, Any]:
    prefix = group["prefix"]
    packed = artifact[f"{prefix}_packed"]
    shape = tuple(int(x) for x in artifact[f"{prefix}_shape"].tolist())
    pad = int(artifact[f"{prefix}_pad"][0])
    scales = artifact[f"{prefix}_scales"]
    code_count = int(np.prod(shape))
    expected_pad = (-code_count) % 8
    expected_packed_bytes = ceil_div(code_count, 8)
    group_size = group_size_for_artifact(artifact, group, manifest_group_size)
    expected_scale_count = code_count // group_size
    checks = {
        "packed_byte_count_matches_shape": int(packed.size) == expected_packed_bytes,
        "packed_pad_matches_shape": pad == expected_pad,
        "packed_pad_bits_are_zero": binary_pad_is_zero(packed[-1], pad) if packed.size else True,
        "group_size_positive": group_size > 0,
        "scale_dtype_fp16": scales.dtype == np.float16,
        "scale_count_matches_codes": int(scales.size) == expected_scale_count,
    }
    return {
        "tensor": parent["tensor"],
        "group": group["group"],
        "rate": group["rate"],
        "prefix": prefix,
        "params": group.get("params"),
        "packed_bytes": int(packed.size),
        "code_shape": list(shape),
        "pad": pad,
        "expected_pad": expected_pad,
        "group_size": group_size,
        "scale_dtype": str(scales.dtype),
        "scale_count": int(scales.size),
        "checks": checks,
        "ok": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    manifest_group_size = int(manifest["group_size"])
    rows: list[dict[str, Any]] = []
    by_rate: dict[str, int] = {}
    artifact_cache: dict[str, Any] = {}
    try:
        for parent in manifest["packed"]:
            artifact_path = str(parent["artifact"])
            artifact = artifact_cache.get(artifact_path)
            if artifact is None:
                artifact = np.load(artifact_path)
                artifact_cache[artifact_path] = artifact
            for group in parent["groups"]:
                rate = str(group["rate"])
                by_rate[rate] = by_rate.get(rate, 0) + 1
                if rate in TERNARY_RATES:
                    rows.append(verify_ternary_group(artifact, group, parent, manifest_group_size))
                elif rate in BINARY_RATES:
                    rows.append(verify_binary_group(artifact, group, parent, manifest_group_size))
    finally:
        for artifact in artifact_cache.values():
            artifact.close()

    failures = [row for row in rows if not row["ok"]]
    summary = manifest.get("summary") or {}
    result = {
        "manifest": str(args.manifest),
        "summary": {
            "checked_lowbit_groups": len(rows),
            "failed_lowbit_groups": len(failures),
            "by_rate": by_rate,
            "storage_lossless_for_lowbit": not failures,
            "storage_lossless_for_ternary": not any(row for row in rows if row["rate"] in TERNARY_RATES and not row["ok"]),
            "all_ternary_groups_fp16_scale": all(
                row["checks"].get("scale_dtype_fp16", False) for row in rows if row["rate"] in TERNARY_RATES
            ),
            "ternary_group_sizes": sorted({row["group_size"] for row in rows if row["rate"] in TERNARY_RATES}),
            "artifact_bytes": summary.get("artifact_bytes"),
            "source_bytes": summary.get("source_bytes"),
            "compression": summary.get("compression"),
        },
        "failures": failures,
        "rows": rows,
    }
    out = args.out or args.manifest.with_name("lossless_verify.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(out), **result["summary"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
