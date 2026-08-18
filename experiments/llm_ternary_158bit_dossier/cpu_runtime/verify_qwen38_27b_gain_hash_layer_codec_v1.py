#!/usr/bin/env python3
"""Verify the complete gain-hash layer body and its CQ3 container boundary."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_cq3_container import Qwen38CQ3Reader, Qwen38CQ3Writer
from clark_quantize.qwen38_fixed_support_codec import (
    pack_fixed_support_matrix,
    unpack_fixed_support_matrix,
)
from clark_quantize.qwen38_gain_hash_layer_codec import (
    GainHashLayerComponents,
    deserialize_gain_hash_layer,
    gain_hash_layer_body_layout,
    gain_hash_layer_storage_bits_per_weight,
    pack_gain_hash_layer,
)
from clark_quantize.qwen38_gain_hash_residual import gain_hash_storage_projection
from clark_quantize.qwen38_gain_route_codec import pack_gain_hash_route_body


REAL_SHAPES = (
    ("gate_or_up_projection", 17408, 5120, 40),
    ("down_projection", 5120, 17408, 136),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_components() -> tuple[GainHashLayerComponents, np.ndarray]:
    generator = np.random.default_rng(158358)
    rows, columns, sketch_dim = 3, 256, 5
    codes = np.zeros((rows, columns), dtype=np.int8)
    for group in codes.reshape(-1, 128):
        support = generator.choice(128, size=64, replace=False)
        group[support] = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=64
        )
    buckets = generator.integers(0, sketch_dim, size=columns, dtype=np.uint16)
    signs = generator.choice(
        np.array([-1, 1], dtype=np.int8), size=columns
    )
    gains = generator.integers(0, 8, size=columns, dtype=np.uint8)
    route = pack_gain_hash_route_body(
        buckets,
        signs,
        gains,
        np.sort(generator.uniform(0.0, 2.0, size=8).astype(np.float16)),
        sketch_dim=sketch_dim,
        gain_bits=3,
    )
    return (
        GainHashLayerComponents(
            ternary=pack_fixed_support_matrix(
                codes, group_size=128, nonzero_per_group=64
            ),
            scale_indices=generator.integers(
                0, 16, size=(rows, columns // 128), dtype=np.uint8
            ),
            row_anchors=generator.uniform(0.001, 0.02, size=rows).astype(
                np.float16
            ),
            ratio_codebook=np.sort(
                generator.uniform(0.5, 1.5, size=16).astype(np.float16)
            ),
            route=route,
            correction_codes=generator.integers(
                -7, 8, size=(rows, sketch_dim), dtype=np.int8
            ),
            correction_row_scales=generator.uniform(
                0.0001, 0.01, size=rows
            ).astype(np.float16),
        ),
        codes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO
        / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts/"
            "qwen38-gain-hash-layer-codec-v1.json"
        ),
    )
    args = parser.parse_args()

    source, source_codes = synthetic_components()
    body = pack_gain_hash_layer(source)
    restored = deserialize_gain_hash_layer(body)
    exact_roundtrip = bool(
        np.array_equal(unpack_fixed_support_matrix(restored.ternary), source_codes)
        and np.array_equal(restored.scale_indices, source.scale_indices)
        and np.array_equal(restored.row_anchors, source.row_anchors)
        and np.array_equal(restored.ratio_codebook, source.ratio_codebook)
        and np.array_equal(restored.correction_codes, source.correction_codes)
        and np.array_equal(
            restored.correction_row_scales, source.correction_row_scales
        )
        and pack_gain_hash_layer(restored) == body
    )

    with tempfile.TemporaryDirectory() as temporary:
        container_path = Path(temporary) / "qwen38-gain-layer.cq3"
        with Qwen38CQ3Writer(
            container_path,
            model={"id": "Qwen/Qwen3.8-27B", "revision": "verification"},
            strategy={"name": "fixed-support-gain-hash-v1"},
            parameter_count=source.rows * source.columns,
        ) as writer:
            writer.add_section(
                "synthetic.weight",
                "gain_hash_layer_v1",
                body,
                metadata={
                    "rows": source.rows,
                    "columns": source.columns,
                    "sketch_dim": source.sketch_dim,
                },
            )
            writer.finalize(
                tensors=[
                    {
                        "name": "synthetic.weight",
                        "shape": [source.rows, source.columns],
                        "parameters": source.rows * source.columns,
                        "sections": ["synthetic.weight"],
                    }
                ]
            )
        with Qwen38CQ3Reader(container_path) as reader:
            cq3_section_verified = reader.verify_section("synthetic.weight")
            cq3_section_exact = reader.read_section("synthetic.weight") == body

    records = []
    for name, rows, columns, sketch_dim in REAL_SHAPES:
        layout = gain_hash_layer_body_layout(
            rows=rows,
            columns=columns,
            sketch_dim=sketch_dim,
            gain_bits=3,
        )
        ledger = gain_hash_storage_projection(
            input_dim=columns,
            output_rows=rows,
            sketch_dim=sketch_dim,
            gain_bits=3,
            nonzero_per_group=64,
            group=128,
            header_bytes=64,
        )
        records.append(
            {
                "name": name,
                "rows": rows,
                "columns": columns,
                "parameters": rows * columns,
                "sketch_dim": sketch_dim,
                "component_lengths": {
                    key: layout[key]
                    for key in (
                        "header_bytes",
                        "ternary_bytes",
                        "scale_index_bytes",
                        "row_anchor_bytes",
                        "ratio_codebook_bytes",
                        "route_body_bytes",
                        "correction_code_bytes",
                        "correction_scale_bytes",
                    )
                },
                "canonical_body_bytes": layout["body_bytes"],
                "ledger_total_bytes": ledger["total_bytes"],
                "bits_per_weight": gain_hash_layer_storage_bits_per_weight(
                    rows=rows, columns=columns, sketch_dim=sketch_dim
                ),
                "body_matches_storage_ledger": layout["body_bytes"]
                == ledger["total_bytes"],
            }
        )

    passed = bool(
        exact_roundtrip
        and cq3_section_verified
        and cq3_section_exact
        and len(records) == len(REAL_SHAPES)
        and all(record["body_matches_storage_ledger"] for record in records)
    )
    codec = REPO / "clark_quantize/qwen38_gain_hash_layer_codec.py"
    route_codec = REPO / "clark_quantize/qwen38_gain_route_codec.py"
    container = REPO / "clark_quantize/qwen38_cq3_container.py"
    verifier = Path(__file__).resolve()
    receipt = {
        "schema": "qwen38_27b_gain_hash_layer_codec_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "Qwen/Qwen3.8-27B",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources": {
            "layer_codec": {
                "path": str(codec.relative_to(REPO)),
                "sha256": sha256_file(codec),
            },
            "route_codec": {
                "path": str(route_codec.relative_to(REPO)),
                "sha256": sha256_file(route_codec),
            },
            "cq3_container": {
                "path": str(container.relative_to(REPO)),
                "sha256": sha256_file(container),
            },
            "verifier": {
                "path": str(verifier.relative_to(REPO)),
                "sha256": sha256_file(verifier),
            },
        },
        "representation": {
            "header_bytes": 64,
            "ternary_storage": "exact enumerative 64-of-128 support plus signs",
            "load_time_compute_transcode": "mask64 support/sign body",
            "scale_storage": "four-bit group index plus FP16 row anchor and tensor codebook",
            "correction_storage": "signed row-aligned INT4 plus one FP16 scale per row",
            "route_storage": "canonical bucket sign and three-bit gain body",
            "outer_container": "authenticated CQ3 tensor section",
        },
        "synthetic_exactness": {
            "body_bytes": len(body),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "exact_roundtrip_and_repack": exact_roundtrip,
            "cq3_section_verified": cq3_section_verified,
            "cq3_section_exact": cq3_section_exact,
        },
        "records": records,
        "gate": {
            "required_real_shapes": len(REAL_SHAPES),
            "observed_real_shapes": len(records),
            "require_exact_synthetic_roundtrip": True,
            "require_authenticated_outer_container": True,
            "require_storage_ledger_identity": True,
            "passed": passed,
        },
        "complete": True,
        "terminal": True,
        "claim_boundary": {
            "complete_layer_body_schema_proven": passed,
            "real_Qwen_projection_shapes_counted": True,
            "real_model_weights_encoded": False,
            "quality_or_capability_proven": False,
            "token_throughput_proven": False,
            "whole_model_artifact_proven": False,
            "semi_lossless_or_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
