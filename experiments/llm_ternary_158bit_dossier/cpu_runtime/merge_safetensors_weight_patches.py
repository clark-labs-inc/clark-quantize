#!/usr/bin/env python3
"""Merge two disjoint safetensors weight patches with a hashed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def merge_patches(
    *,
    base_path: Path,
    addition_path: Path,
    out_path: Path,
    receipt_path: Path,
    expected_tensor_count: int,
    required_key_regex: str | None,
) -> dict[str, Any]:
    for path in (base_path, addition_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite output patch: {out_path}")
    partial = out_path.with_name(f".{out_path.name}.tmp")
    if partial.exists():
        raise FileExistsError(f"partial output requires inspection: {partial}")

    base = load_file(str(base_path), device="cpu")
    addition = load_file(str(addition_path), device="cpu")
    if not base or not addition:
        raise ValueError("both input patches must be nonempty")
    overlap = sorted(set(base) & set(addition))
    if overlap:
        raise ValueError(f"patch tensor overlap is forbidden: {overlap}")

    merged = {
        key: tensor.contiguous()
        for key, tensor in sorted({**base, **addition}.items())
    }
    if expected_tensor_count > 0 and len(merged) != expected_tensor_count:
        raise ValueError(
            f"expected {expected_tensor_count} tensors, found {len(merged)}"
        )
    if required_key_regex is not None:
        pattern = re.compile(required_key_regex)
        invalid = [key for key in merged if pattern.fullmatch(key) is None]
        if invalid:
            raise ValueError(f"merged patch contains out-of-scope tensors: {invalid}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(partial))
    os.replace(partial, out_path)
    verified = load_file(str(out_path), device="cpu")
    if set(verified) != set(merged):
        raise RuntimeError("merged patch key verification failed")
    for key in merged:
        if not verified[key].equal(merged[key]):
            raise RuntimeError(f"merged tensor verification failed: {key}")

    result: dict[str, Any] = {
        "schema": "disjoint_safetensors_weight_patch_merge_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": {
            "path": str(base_path.resolve()),
            "sha256": sha256_file(base_path),
            "tensor_names": sorted(base),
        },
        "addition": {
            "path": str(addition_path.resolve()),
            "sha256": sha256_file(addition_path),
            "tensor_names": sorted(addition),
        },
        "overlap": overlap,
        "out": {
            "path": str(out_path.resolve()),
            "sha256": sha256_file(out_path),
            "tensor_names": sorted(merged),
            "tensor_count": len(merged),
        },
        "expected_tensor_count": expected_tensor_count,
        "required_key_regex": required_key_regex,
        "valid": True,
    }
    result["report_content_sha256"] = canonical_digest(result)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.tmp")
    temporary_receipt.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    os.replace(temporary_receipt, receipt_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--addition", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-tensor-count", type=int, default=0)
    parser.add_argument("--required-key-regex")
    args = parser.parse_args()
    if args.expected_tensor_count < 0:
        parser.error("--expected-tensor-count must be non-negative")
    result = merge_patches(
        base_path=args.base,
        addition_path=args.addition,
        out_path=args.out,
        receipt_path=args.receipt,
        expected_tensor_count=args.expected_tensor_count,
        required_key_regex=args.required_key_regex,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

