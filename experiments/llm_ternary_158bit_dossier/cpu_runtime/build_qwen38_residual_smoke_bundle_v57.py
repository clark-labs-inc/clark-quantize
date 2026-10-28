#!/usr/bin/env python3
"""Build a development-only v57 bundle from an exact v37 pilot payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_qwen38_27b_residual_bundle_v56 import (
    LAYER_COUNT,
    LayerPayload,
    sha256_file,
    validate_bundle,
    write_bundle,
)


TRAINING_SHA256 = "6871879dd4413419bfb36e233275481bce291d6bac1c884770c2514c876eeb73"
RESIDUAL_SHA256 = "26dae44ab53bf194b89c46e7a175808865b8c8eb40207dd83ef4edaefec7ba36"
PREREGISTRATION_SHA256 = (
    "4d5b8646887ccee1959fcd12fb8056deaa57ce06941a0f721c57843ba15dbd36"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.layer < 0 or args.layer >= LAYER_COUNT:
        raise ValueError("smoke layer must be in [0, 63]")
    payload_sha = sha256_file(args.payload)
    records = [
        LayerPayload(
            layer=layer,
            active=layer == args.layer,
            path=args.payload if layer == args.layer else None,
            sha256=payload_sha if layer == args.layer else None,
        )
        for layer in range(LAYER_COUNT)
    ]
    write_bundle(
        records,
        args.output,
        training_sha256=TRAINING_SHA256,
        residual_sha256=RESIDUAL_SHA256,
        preregistration_sha256=PREREGISTRATION_SHA256,
    )
    summary = validate_bundle(
        args.output,
        training_sha256=TRAINING_SHA256,
        residual_sha256=RESIDUAL_SHA256,
        preregistration_sha256=PREREGISTRATION_SHA256,
    )
    receipt = {
        "schema": "qwen38_residual_runtime_smoke_bundle_v57",
        "claim_scope": "development parity only",
        "layer": args.layer,
        "source_payload": {
            "path": str(args.payload),
            "sha256": payload_sha,
        },
        "bundle": {
            "path": str(args.output),
            "bytes": summary.bytes,
            "sha256": sha256_file(args.output),
            "root_sha256": summary.root_sha256,
        },
        "environment": {
            "QWEN38_RESIDUAL_TRAINING_SHA256": TRAINING_SHA256,
            "QWEN38_RESIDUAL_RECEIPT_SHA256": RESIDUAL_SHA256,
            "QWEN38_RESIDUAL_PREREG_SHA256": PREREGISTRATION_SHA256,
        },
        "complete": True,
        "terminal": True,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
