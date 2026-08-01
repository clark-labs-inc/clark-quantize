#!/usr/bin/env python3
"""Merge per-browser Swin2SR smoke output into one promotion receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(payloads: list[dict]) -> dict:
    hashes = {payload.get("model_sha256") for payload in payloads}
    if None in hashes or len(hashes) != 1:
        raise ValueError("runtime receipts must reference one candidate state")
    fixture_hashes = {payload.get("fixture_sha256") for payload in payloads}
    hard_state_hashes = {
        payload.get("fixture_hard_state_sha256")
        for payload in payloads
    }
    if None in fixture_hashes or len(fixture_hashes) != 1:
        raise ValueError("runtime receipts must reference one fixture")
    if None in hard_state_hashes or len(hard_state_hashes) != 1:
        raise ValueError(
            "runtime receipts must reference one canonical hard state"
        )
    receipts = {}
    for payload in payloads:
        for receipt in payload.get("receipts", []):
            surface = receipt.get("surface")
            if not surface:
                raise ValueError("runtime receipt is missing its surface")
            if surface in receipts:
                raise ValueError(f"duplicate runtime surface: {surface}")
            receipts[surface] = receipt
    return {
        "format": "backdropkit-swin2sr-runtime-receipts-v1",
        "model_sha256": hashes.pop(),
        "fixture_sha256": fixture_hashes.pop(),
        "fixture_hard_state_sha256": hard_state_hashes.pop(),
        "receipts": [receipts[name] for name in sorted(receipts)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = merge([json.loads(path.read_text()) for path in args.input])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
