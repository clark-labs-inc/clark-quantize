#!/usr/bin/env python3
"""Emit the preregistered whole-checkpoint Qwen3.8-27B storage projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_storage_budget import build_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/quant-state/qwen38-27b-storage-budget-v1.json"))
    args = parser.parse_args()
    receipt = build_receipt(args.model_dir)
    expected = {
        "tensor_count": 1199,
        "total_parameters": 27781427952,
        "ternary_rank_ge_2_tensors": 666,
        "ternary_rank_ge_2_parameters": 27780313088,
        "retained_rank_1_tensors": 533,
        "retained_rank_1_parameters": 1114864,
    }
    if receipt["inventory"] != expected:
        raise RuntimeError(
            f"Qwen3.8-27B inventory changed: expected {expected}, got {receipt['inventory']}"
        )
    candidate = receipt["strategies"][receipt["candidate"]["name"]]
    receipt["gate"] = {
        "maximum_projected_complete_bundle_bpw": 1.63,
        "observed_projected_complete_bundle_bpw": candidate["effective_bits_per_total_parameter"],
        "passed": candidate["effective_bits_per_total_parameter"] <= 1.63,
        "scope": "byte-accounting feasibility only",
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
