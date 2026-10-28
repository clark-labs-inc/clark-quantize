#!/usr/bin/env python3
"""Emit a deterministic receipt for the canonical Qwen gain-route body."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_gain_hash_residual import gain_hash_storage_projection
from clark_quantize.qwen38_gain_route_codec import (
    deserialize_gain_hash_route_body,
    expand_gain_hash_route_compute,
    pack_gain_hash_route_body,
    serialize_gain_hash_route_body,
    unpack_gain_hash_route_body,
)


CASES = (
    ("gate_or_up_projection", 5120, 17408, 40),
    ("down_projection", 17408, 5120, 136),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts/"
            "qwen38-gain-route-codec-v1.json"
        ),
    )
    args = parser.parse_args()
    records = []
    for index, (name, input_dim, output_rows, sketch_dim) in enumerate(CASES):
        generator = np.random.default_rng(158327 + index)
        buckets = generator.integers(
            0, sketch_dim, size=input_dim, dtype=np.uint16
        )
        signs = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=input_dim
        )
        gain_codes = generator.integers(0, 8, size=input_dim, dtype=np.uint8)
        gain_codebook = np.sort(
            generator.uniform(0.0, 2.0, size=8).astype(np.float16)
        )
        packed = pack_gain_hash_route_body(
            buckets,
            signs,
            gain_codes,
            gain_codebook,
            sketch_dim=sketch_dim,
            gain_bits=3,
        )
        body = serialize_gain_hash_route_body(packed)
        restored = deserialize_gain_hash_route_body(
            body,
            input_dim=input_dim,
            sketch_dim=sketch_dim,
            gain_bits=3,
        )
        decoded = unpack_gain_hash_route_body(restored)
        expanded = expand_gain_hash_route_compute(restored)
        exact_roundtrip = all(
            np.array_equal(observed, expected)
            for observed, expected in zip(
                decoded, (buckets, signs, gain_codes)
            )
        ) and np.array_equal(restored.gain_codebook, gain_codebook)
        exact_repack = serialize_gain_hash_route_body(restored) == body
        storage = gain_hash_storage_projection(
            input_dim=input_dim,
            output_rows=output_rows,
            sketch_dim=sketch_dim,
            gain_bits=3,
            nonzero_per_group=64,
            group=128,
        )
        expected_body_bytes = int(storage["routing_and_gain_index_bytes"]) + int(
            storage["gain_codebook_bytes"]
        )
        compute_bytes = sum(array.nbytes for array in expanded)
        records.append(
            {
                "name": name,
                "input_dim": input_dim,
                "output_rows": output_rows,
                "sketch_dim": sketch_dim,
                "bucket_bits": packed.bucket_bits,
                "bits_per_input": packed.bits_per_input,
                "route_payload_bytes": packed.route_payload_bytes,
                "gain_codebook_bytes": packed.codebook_bytes,
                "canonical_body_bytes": len(body),
                "ledger_body_bytes": expected_body_bytes,
                "compute_layout_bytes": compute_bytes,
                "compute_expansion_ratio": compute_bytes / len(body),
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "exact_roundtrip": exact_roundtrip,
                "exact_repack": exact_repack,
                "body_matches_storage_ledger": len(body) == expected_body_bytes,
                "compute_layout_count_matches": compute_bytes
                == packed.compute_layout_bytes,
            }
        )
    passed = all(
        record["exact_roundtrip"]
        and record["exact_repack"]
        and record["body_matches_storage_ledger"]
        and record["compute_layout_count_matches"]
        for record in records
    )
    codec = REPO / "clark_quantize/qwen38_gain_route_codec.py"
    adapter = REPO / "clark_quantize/qwen38_mask64_packed_gain_hash_fused_cpu.py"
    verifier = Path(__file__).resolve()
    receipt = {
        "schema": "qwen38_27b_gain_route_codec_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "Qwen/Qwen3.8-27B",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources": {
            "codec": {"path": str(codec.relative_to(REPO)), "sha256": sha256_file(codec)},
            "fused_adapter": {
                "path": str(adapter.relative_to(REPO)),
                "sha256": sha256_file(adapter),
            },
            "verifier": {
                "path": str(verifier.relative_to(REPO)),
                "sha256": sha256_file(verifier),
            },
        },
        "representation": {
            "coordinate_order": "bucket LSB first then sign bit then three-bit gain index",
            "gain_codebook": "eight sorted little-endian FP16 values",
            "padding": "zero high bits in the final route byte",
            "compute_expansion": "uint16 buckets int8 signs uint8 gain indices and FP32 codebook",
        },
        "records": records,
        "gate": {
            "required_cases": 2,
            "observed_cases": len(records),
            "require_exact_roundtrip": True,
            "require_exact_repack": True,
            "require_storage_ledger_identity": True,
            "require_compute_layout_count_identity": True,
            "passed": passed and len(records) == 2,
        },
        "complete": True,
        "terminal": True,
        "claim_boundary": {
            "canonical_gain_route_body_proven": passed,
            "exact_Qwen_projection_shapes_covered": True,
            "real_model_routes_or_weights_encoded": False,
            "capability_or_token_throughput_proven": False,
            "semi_lossless_or_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
