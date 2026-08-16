#!/usr/bin/env python3
"""Measure the fast 1.6-bit radix-5 alternative on real Qwen ternary codes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from base3_packing_research import load_codes
from clark_quantize.base3_radix5 import pack_radix5, radix5_metadata, unpack_radix5
from clark_quantize.base3_chunks import atomic_bytes


def validate_preregistration(preregistration: dict, args) -> dict:
    performance = preregistration.get("radix5_performance", {})
    if (
        preregistration.get("schema") != "clark_qwen38_evaluation_preregistration_v1"
        or performance.get("real_qwen_trits") != args.trits
        or performance.get("sample_rows_per_source") != args.sample_rows
        or performance.get("maximum_delta_bits_per_trit_from_log2_3") != 0.016
        or performance.get("minimum_encode_trits_per_second") != 50_000_000
        or performance.get("minimum_decode_trits_per_second") != 50_000_000
        or performance.get("require_exact_roundtrip") is not True
    ):
        raise ValueError("radix-5 arguments differ from preregistration")
    return performance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--preflight", type=Path, default=Path("/workspace/quant-state/qwen38-preflight.json"))
    parser.add_argument("--trits", type=int, default=10_000_000)
    parser.add_argument("--sample-rows", type=int, default=512)
    parser.add_argument("--receipt", type=Path, default=Path("/workspace/quant-state/qwen38-radix5-performance.json"))
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path(__file__).resolve().parent / "evaluation_preregistration.json",
    )
    args = parser.parse_args()
    if args.trits <= 0:
        raise ValueError("trit count must be positive")
    preregistration_content = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_content)
    thresholds = validate_preregistration(preregistration, args)
    preflight = json.loads(args.preflight.read_text())
    if preregistration.get("model_revision") != preflight["download_receipt"]["revision"]:
        raise ValueError("radix-5 preregistration model revision mismatch")
    codes, sources = load_codes(args.model_dir, preflight, args.sample_rows, args.trits)
    source_sha256 = hashlib.sha256(codes.tobytes()).hexdigest()

    started = time.perf_counter()
    frame = pack_radix5(codes)
    encode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    decoded = unpack_radix5(frame)
    decode_seconds = time.perf_counter() - started
    exact = bool(np.array_equal(decoded, codes))
    if not exact:
        raise AssertionError("radix-5 round-trip changed real Qwen trits")
    metadata = radix5_metadata(frame)
    ideal = math.log2(3)
    ledger = preflight["parameter_ledger"]
    overhead_bits = (metadata["payload_bits_per_trit"] - ideal) * ledger["ternary_payload"]
    projected_bytes = ledger["base3_payload_bytes"] + math.ceil(overhead_bits / 8)
    result = {
        "schema": "clark_qwen38_radix5_performance_v1",
        "hypothesis": (
            "Canonical vectorized five-trits-per-byte packing trades no more than 0.016 bits/trit "
            "for at least 50 Mtrits/s encode and decode throughput on real Qwen ternary codes."
        ),
        "model_revision": preflight["download_receipt"]["revision"],
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": hashlib.sha256(preregistration_content).hexdigest(),
        },
        "sample": {"trits": int(codes.size), "source_sha256": source_sha256, "sources": sources},
        "format": {
            **metadata,
            "ideal_bits_per_trit": ideal,
            "delta_bits_per_trit": metadata["payload_bits_per_trit"] - ideal,
            "exact_roundtrip": exact,
            "frame_sha256": hashlib.sha256(frame).hexdigest(),
        },
        "performance": {
            "encode_seconds": encode_seconds,
            "decode_seconds": decode_seconds,
            "encode_mtrits_per_second": codes.size / encode_seconds / 1e6,
            "decode_mtrits_per_second": codes.size / decode_seconds / 1e6,
        },
        "full_model_projection": {
            "extra_bytes_vs_ideal_base3": math.ceil(overhead_bits / 8),
            "complete_payload_bytes": projected_bytes,
            "effective_bits_per_total_parameter": projected_bytes * 8 / ledger["total"],
        },
        "promotion": {
            "passed": (
                exact
                and codes.size == args.trits
                and sum(int(source["used_trits"]) for source in sources) == args.trits
                and metadata["payload_bits_per_trit"] - ideal
                    <= thresholds["maximum_delta_bits_per_trit_from_log2_3"]
                and codes.size / encode_seconds >= thresholds["minimum_encode_trits_per_second"]
                and codes.size / decode_seconds >= thresholds["minimum_decode_trits_per_second"]
            ),
            "scope": "fast 1.6-bit code payload; this is not ideal entropy packing or a quality result",
        },
        "stop_condition": "Ten million real trits or first correctness, canonicality, or throughput failure.",
    }
    atomic_bytes(args.receipt, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["promotion"]["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
