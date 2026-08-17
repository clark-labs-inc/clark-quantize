#!/usr/bin/env python3
"""Project lossless Huffman storage for every Qwen3.8-27B scale index."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_scale_index_codec import (
    HEADER,
    scale_index_huffman_ledger_from_frequencies,
)
from clark_quantize.qwen38_storage_budget import align_up


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_output(root: Path, descriptor: dict) -> Path:
    path = Path(descriptor["path"])
    return path if path.is_absolute() else root / path


def index_frequencies(path: Path, count: int, expected_sha256: str) -> np.ndarray:
    content = path.read_bytes()
    if len(content) != (count + 1) // 2:
        raise RuntimeError(f"packed scale-index length changed: {path}")
    packed = np.frombuffer(content, dtype=np.uint8)
    if count & 1 and int(packed[-1]) & 0xF0:
        raise RuntimeError(f"noncanonical scale-index padding: {path}")
    indices = np.empty(count, dtype=np.uint8)
    indices[0::2] = packed & np.uint8(0x0F)
    if count > 1:
        indices[1::2] = packed[: count // 2] >> np.uint8(4)
    if hashlib.sha256(indices.tobytes()).hexdigest() != expected_sha256:
        raise RuntimeError(f"unpacked scale-index digest changed: {path}")
    return np.bincount(indices, minlength=16).astype(np.int64)


def validate_frozen_inputs(preregistration: dict) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen Qwen3.8 Huffman input changed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path(
            "/workspace/quant-state/"
            "qwen38-27b-full-row-normalized-scale-codebook4-v1.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/workspace/quant-state/qwen38-27b-scale-index-huffman-v1.json"
        ),
    )
    parser.add_argument(
        "--full-manifest",
        type=Path,
        default=Path(
            "/workspace/quant-runs/"
            "qwen38-full-row-normalized-scale-codebook4-v1/manifest.json"
        ),
    )
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_scale_index_huffman_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_inputs(prereg)

    parent_content = args.parent.read_bytes()
    parent = json.loads(parent_content)
    if (
        parent.get("schema")
        != "qwen38_27b_full_row_normalized_scale_codebook4_receipt_v1"
        or parent.get("complete") is not True
        or parent.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("full Qwen3.8 row-normalized parent did not pass")
    full_manifest_content = args.full_manifest.read_bytes()
    if hashlib.sha256(full_manifest_content).hexdigest() != hashlib.sha256(
        parent_content
    ).hexdigest():
        raise RuntimeError("full Qwen3.8 manifest and receipt diverged")
    parent_root = args.full_manifest.parent
    manifest_path = Path(parent.get("mapped_parent", {}).get("manifest", ""))
    if not manifest_path.is_file():
        raise RuntimeError("mapped Qwen3.8 row-normalized manifest is absent")
    mapped_content = manifest_path.read_bytes()
    mapped = json.loads(mapped_content)
    if (
        mapped.get("schema") != "qwen38_27b_row_normalized_scale_codebook4_receipt_v1"
        or mapped.get("complete") is not True
        or mapped.get("gate", {}).get("passed") is not True
        or hashlib.sha256(mapped_content).hexdigest()
        != parent["mapped_parent"]["sha256"]
    ):
        raise RuntimeError("mapped Qwen3.8 row-normalized parent changed")

    records_with_roots = [
        (manifest_path.parent, record) for record in mapped["records"]
    ] + [
        (parent_root, record) for record in parent["supplemental_records"]
    ]
    scope = prereg["scope"]
    names = [record["tensor"] for _, record in records_with_roots]
    if (
        len(records_with_roots) != int(scope["rank_ge_2_tensors"])
        or len(set(names)) != len(names)
        or sum(int(record["scale_count"]) for _, record in records_with_roots)
        != int(scope["group_128_scale_indices"])
    ):
        raise RuntimeError("Qwen3.8 Huffman tensor or scale-index coverage changed")

    current_scale_sections = 0
    huffman_scale_sections = 0
    fixed_index_bytes = 0
    huffman_payload_bits = 0
    huffman_payload_bytes = 0
    huffman_header_bytes = 0
    global_frequencies = np.zeros(16, dtype=np.int64)
    tensor_records = []
    for root, record in records_with_roots:
        descriptor = record["indices"]
        path = resolve_output(root, descriptor)
        if (
            not path.is_file()
            or path.stat().st_size != int(descriptor["bytes"])
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise RuntimeError(f"authenticated scale-index payload changed: {path}")
        count = int(record["scale_count"])
        frequencies = index_frequencies(
            path, count, record["unpacked_indices_sha256"]
        )
        ledger = scale_index_huffman_ledger_from_frequencies(frequencies)
        if ledger.count != count or ledger.fixed_width_bytes != int(descriptor["bytes"]):
            raise RuntimeError("scale-index Huffman tensor ledger changed")
        anchors = int(record["anchors"]["bytes"])
        codebook = int(record["codebook"]["bytes"])
        current_aligned = align_up(ledger.fixed_width_bytes + anchors + codebook)
        huffman_aligned = align_up(ledger.section_bytes + anchors + codebook)
        current_scale_sections += current_aligned
        huffman_scale_sections += huffman_aligned
        fixed_index_bytes += ledger.fixed_width_bytes
        huffman_payload_bits += ledger.payload_bits
        huffman_payload_bytes += ledger.payload_bytes
        huffman_header_bytes += HEADER.size
        global_frequencies += frequencies
        tensor_records.append(
            {
                "tensor": record["tensor"],
                "scale_indices": count,
                "frequencies": list(ledger.frequencies),
                "code_lengths": list(ledger.code_lengths),
                "symbol_mask": ledger.symbol_mask,
                "fixed_index_bytes": ledger.fixed_width_bytes,
                "huffman_header_bytes": HEADER.size,
                "huffman_payload_bits": ledger.payload_bits,
                "huffman_payload_bytes": ledger.payload_bytes,
                "current_scale_section_aligned_bytes": current_aligned,
                "huffman_scale_section_aligned_bytes": huffman_aligned,
            }
        )

    storage = parent["storage_ledger"]
    if (
        fixed_index_bytes != int(storage["scales"]["index_bytes"])
        or current_scale_sections
        != int(storage["bytes"]["scale_sections_aligned"])
    ):
        raise RuntimeError("recomputed fixed scale storage differs from parent")
    current_bundle = int(storage["bytes"]["complete_bundle"])
    huffman_bundle = current_bundle - current_scale_sections + huffman_scale_sections
    total_parameters = int(storage["parameters"]["total"])
    current_bpw = current_bundle * 8 / total_parameters
    huffman_bpw = huffman_bundle * 8 / total_parameters
    index_reduction = 1.0 - (huffman_header_bytes + huffman_payload_bytes) / fixed_index_bytes
    average_payload_bits = huffman_payload_bits / int(scope["group_128_scale_indices"])
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_tensors": len(tensor_records),
        "observed_scale_indices": int(global_frequencies.sum()),
        "observed_current_complete_bundle_bpw": current_bpw,
        "observed_huffman_complete_bundle_bpw": huffman_bpw,
        "observed_average_huffman_payload_bits_per_scale_index": average_payload_bits,
        "observed_index_bytes_reduction_fraction_before_alignment": index_reduction,
    }
    gate["passed"] = bool(
        len(tensor_records) == int(gate_spec["required_tensors"])
        and int(global_frequencies.sum()) == int(gate_spec["required_scale_indices"])
        and huffman_bpw <= float(gate_spec["maximum_complete_bundle_bpw"])
        and index_reduction
        >= float(gate_spec["minimum_index_bytes_reduction_fraction_before_alignment"])
        and huffman_bundle < current_bundle
    )
    receipt = {
        "schema": "qwen38_27b_scale_index_huffman_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "parent": {
            "path": str(args.parent),
            "sha256": hashlib.sha256(parent_content).hexdigest(),
            "full_manifest": str(args.full_manifest),
            "full_manifest_sha256": hashlib.sha256(full_manifest_content).hexdigest(),
            "mapped_manifest": str(manifest_path),
            "mapped_manifest_sha256": hashlib.sha256(mapped_content).hexdigest(),
        },
        "coverage": {
            "tensors": len(tensor_records),
            "scale_indices": int(global_frequencies.sum()),
            "global_frequencies": global_frequencies.tolist(),
        },
        "storage": {
            "current_complete_bundle_bytes": current_bundle,
            "current_complete_bundle_bpw": current_bpw,
            "fixed_index_bytes": fixed_index_bytes,
            "current_scale_sections_aligned_bytes": current_scale_sections,
            "huffman_header_bytes": huffman_header_bytes,
            "huffman_payload_bits": huffman_payload_bits,
            "huffman_payload_bytes": huffman_payload_bytes,
            "huffman_average_payload_bits_per_scale_index": average_payload_bits,
            "huffman_index_bytes_reduction_fraction_before_alignment": index_reduction,
            "huffman_scale_sections_aligned_bytes": huffman_scale_sections,
            "huffman_complete_bundle_bytes": huffman_bundle,
            "huffman_complete_bundle_bpw": huffman_bpw,
        },
        "tensor_records": tensor_records,
        "gate": gate,
        "claim_boundary": {
            "lossless_size_projection_over_all_scale_indices_proven_if_passed": gate["passed"],
            "encoded_27b_artifact_exported": False,
            "native_decode_or_transcode_benchmarked": False,
            "model_quality_changed": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.receipt, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
