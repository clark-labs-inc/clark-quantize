#!/usr/bin/env python3
"""Authenticate every mapped Qwen3.8 code and bound nonuniform ternary storage."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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

from clark_quantize.base3 import frame_metadata, native_decoder_available, unpack_base3
from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_row_scale_storage import row_normalized_scale_ledger
from clark_quantize.qwen38_storage_budget import (
    StorageStrategy,
    align_up,
    inventory_from_safetensors,
    tensor_code_bits,
)
from clark_quantize.qwen38_ternary_entropy import ternary_type_class_ledger


EXPECTED_MANIFEST_SHA256 = "c98d878d0c5e3ea92be27c5ba59da2c9d7d0154c839dd89a17d0e94eb18b75b4"
EXPECTED_STORAGE_SHA256 = "1e37e00ff6736e6614d77f7dd1a1f69aecae96503e1659b0d027df9a37caec92"
EXPECTED_MAPPED_TENSORS = 402
EXPECTED_MAPPED_TRITS = 26_869_760_000
EXPECTED_CHUNKS = 34_344


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources(preregistration: dict) -> None:
    for label, descriptor in preregistration["frozen_inputs"].items():
        path = REPO / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"frozen ternary entropy source changed: {label}")


def decode_batch(arguments: tuple[str, list[dict]]) -> list[dict]:
    chunks_root_text, records = arguments
    chunks_root = Path(chunks_root_text)
    if not native_decoder_available():
        raise RuntimeError("native uniform base-3 decoder is required")
    output = []
    for record in records:
        index = int(record["index"])
        path = chunks_root / f"{index:08d}.cq3a"
        frame = path.read_bytes()
        if sha256_file(path) != record["frame_sha256"]:
            raise RuntimeError(f"Qwen3.8 frame digest changed: {index}")
        if frame_metadata(frame) != record["frame_metadata"]:
            raise RuntimeError(f"Qwen3.8 frame metadata changed: {index}")
        codes = unpack_base3(frame)
        if (
            codes.size != int(record["trits"])
            or hashlib.sha256(codes.tobytes()).hexdigest() != record["raw_sha256"]
        ):
            raise RuntimeError(f"Qwen3.8 decoded codes changed: {index}")
        counts = np.bincount(codes.astype(np.int16) + 1, minlength=3)
        ledger = ternary_type_class_ledger(counts)
        output.append(
            {
                "index": index,
                "tensor": record["descriptor"]["tensor"],
                "trits": int(codes.size),
                "counts": [int(value) for value in counts],
                "entropy_bits_per_trit": ledger.entropy_bits_per_trit,
                "payload_bits": ledger.conservative_payload_bits,
                "payload_bytes": ledger.payload_bytes,
                "header_bytes": ledger.header_bytes,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--storage-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-storage-budget-v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-ternary-type-class-rate-v1.json"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-chunks", type=int, default=32)
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_chunks <= 0:
        raise ValueError("workers and batch size must be positive")
    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_ternary_type_class_rate_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    if not native_decoder_available():
        raise SystemExit("infrastructure-invalid: native uniform base-3 decoder is absent")

    manifest_path = args.artifact / "codes.cq3d/manifest.json"
    manifest_content = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Qwen3.8 mapped code manifest changed")
    manifest = json.loads(manifest_content)
    records = manifest.get("chunks", [])
    mapped_sources = manifest.get("identity", {}).get("mapped_sources", [])
    if (
        manifest.get("complete") is not True
        or len(records) != EXPECTED_CHUNKS
        or sum(int(record["trits"]) for record in records) != EXPECTED_MAPPED_TRITS
        or len(mapped_sources) != EXPECTED_MAPPED_TENSORS
        or len({source["name"] for source in mapped_sources}) != EXPECTED_MAPPED_TENSORS
    ):
        raise RuntimeError("Qwen3.8 mapped code coverage changed")
    tasks = [
        (str(args.artifact / "codes.cq3d/chunks"), records[start : start + args.batch_chunks])
        for start in range(0, len(records), args.batch_chunks)
    ]
    decoded_records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for batch in executor.map(decode_batch, tasks, chunksize=1):
            decoded_records.extend(batch)
    if [record["index"] for record in decoded_records] != list(range(EXPECTED_CHUNKS)):
        raise RuntimeError("decoded Qwen3.8 chunk order or coverage changed")

    tensor_accumulators = defaultdict(
        lambda: {"chunks": 0, "trits": 0, "counts": np.zeros(3, dtype=np.int64), "payload_bits": 0, "encoded_bytes": 0}
    )
    global_counts = np.zeros(3, dtype=np.int64)
    total_payload_bits = 0
    total_encoded_bytes = 0
    for record in decoded_records:
        target = tensor_accumulators[record["tensor"]]
        counts = np.asarray(record["counts"], dtype=np.int64)
        target["chunks"] += 1
        target["trits"] += int(record["trits"])
        target["counts"] += counts
        target["payload_bits"] += int(record["payload_bits"])
        target["encoded_bytes"] += int(record["header_bytes"]) + int(record["payload_bytes"])
        global_counts += counts
        total_payload_bits += int(record["payload_bits"])
        total_encoded_bytes += int(record["header_bytes"]) + int(record["payload_bytes"])
    if int(global_counts.sum()) != EXPECTED_MAPPED_TRITS:
        raise RuntimeError("authenticated Qwen3.8 trit denominator changed")

    storage_content = args.storage_receipt.read_bytes()
    if hashlib.sha256(storage_content).hexdigest() != EXPECTED_STORAGE_SHA256:
        raise RuntimeError("Qwen3.8 storage receipt changed")
    storage = json.loads(storage_content)
    base = storage["strategies"]["base3_group128_codebook4"]
    tensors, source = inventory_from_safetensors(args.model_dir)
    if (
        source["index_sha256"] != prereg["scope"]["index_sha256"]
        or len(tensors) != int(storage["inventory"]["tensor_count"])
    ):
        raise RuntimeError("Qwen3.8 checkpoint inventory changed")
    row_storage = row_normalized_scale_ledger(tensors, base)
    by_name = {tensor.name: tensor for tensor in tensors}
    uniform_strategy = StorageStrategy("base3_group128_codebook4", "base3", 128, 4, True)
    mapped_source_by_name = {source_record["name"]: source_record for source_record in mapped_sources}
    tensor_receipts = []
    current_mapped_sections = 0
    type_class_mapped_sections = 0
    for name in sorted(tensor_accumulators):
        accumulator = tensor_accumulators[name]
        tensor = by_name.get(name)
        mapped_source = mapped_source_by_name.get(name)
        if tensor is None or mapped_source is None or tensor.parameters != int(accumulator["trits"]):
            raise RuntimeError(f"mapped Qwen3.8 tensor identity changed: {name}")
        current_bytes = math.ceil(tensor_code_bits(tensor, uniform_strategy) / 8)
        current_aligned = align_up(current_bytes)
        candidate_aligned = align_up(int(accumulator["encoded_bytes"]))
        current_mapped_sections += current_aligned
        type_class_mapped_sections += candidate_aligned
        counts = accumulator["counts"]
        tensor_receipts.append(
            {
                "tensor": name,
                "shape": list(tensor.shape),
                "chunks": int(accumulator["chunks"]),
                "trits": int(accumulator["trits"]),
                "counts": [int(value) for value in counts],
                "entropy_bits_per_trit": ternary_type_class_ledger(counts).entropy_bits_per_trit,
                "conservative_payload_bits": int(accumulator["payload_bits"]),
                "encoded_bytes_before_tensor_alignment": int(accumulator["encoded_bytes"]),
                "encoded_section_aligned_bytes": candidate_aligned,
                "uniform_base3_section_aligned_bytes": current_aligned,
            }
        )
    if len(tensor_receipts) != EXPECTED_MAPPED_TENSORS:
        raise RuntimeError("Qwen3.8 tensor entropy coverage changed")

    projected_complete_bytes = (
        int(row_storage["bytes"]["complete_bundle"])
        - current_mapped_sections
        + type_class_mapped_sections
    )
    total_parameters = int(row_storage["parameters"]["total"])
    projected_bpw = projected_complete_bytes * 8 / total_parameters
    uniform_mapped_bits = sum(int(record["trits"]) * math.log2(3) for record in decoded_records)
    effective_candidate_bits = total_encoded_bytes * 8
    saving_bits_per_mapped_trit = (
        uniform_mapped_bits - effective_candidate_bits
    ) / EXPECTED_MAPPED_TRITS
    global_entropy = ternary_type_class_ledger(global_counts).entropy_bits_per_trit
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "observed_authenticated_chunks": len(decoded_records),
        "observed_authenticated_tensors": len(tensor_receipts),
        "observed_authenticated_mapped_trits": int(global_counts.sum()),
        "observed_projected_complete_bundle_bpw": projected_bpw,
        "observed_effective_saving_bits_per_mapped_trit": saving_bits_per_mapped_trit,
    }
    gate["passed"] = bool(
        len(decoded_records) == int(gate_spec["required_chunks"])
        and len(tensor_receipts) == int(gate_spec["required_mapped_tensors"])
        and int(global_counts.sum()) == int(gate_spec["required_mapped_trits"])
        and projected_bpw <= float(gate_spec["maximum_projected_complete_bundle_bpw"])
        and saving_bits_per_mapped_trit >= float(gate_spec["minimum_effective_saving_bits_per_mapped_trit"])
    )
    receipt = {
        "schema": "qwen38_27b_ternary_type_class_rate_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "preregistration_sha256": sha256_file(prereg_path),
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "storage_receipt": str(args.storage_receipt),
            "storage_receipt_sha256": hashlib.sha256(storage_content).hexdigest(),
        },
        "authenticated_codes": {
            "chunks": len(decoded_records),
            "tensors": len(tensor_receipts),
            "trits": int(global_counts.sum()),
            "counts": [int(value) for value in global_counts],
            "zero_fraction": int(global_counts[1]) / int(global_counts.sum()),
            "global_empirical_entropy_bits_per_trit": global_entropy,
        },
        "rate": {
            "conservative_type_class_payload_bits": total_payload_bits,
            "chunk_headers_and_byte_aligned_payload_bytes": total_encoded_bytes,
            "effective_bits_per_mapped_trit_including_chunk_headers": effective_candidate_bits / EXPECTED_MAPPED_TRITS,
            "uniform_log2_3_bits_per_mapped_trit": math.log2(3),
            "effective_saving_bits_per_mapped_trit": saving_bits_per_mapped_trit,
        },
        "storage_projection": {
            "parent_row_normalized_complete_bundle_bytes": int(row_storage["bytes"]["complete_bundle"]),
            "parent_row_normalized_complete_bundle_bpw": float(row_storage["effective_bits_per_total_parameter"]),
            "uniform_mapped_code_sections_aligned_bytes": current_mapped_sections,
            "type_class_mapped_code_sections_aligned_bytes": type_class_mapped_sections,
            "supplemental_ternary_codes_remain_conservative_uniform_base3": True,
            "projected_complete_bundle_bytes": projected_complete_bytes,
            "projected_complete_bundle_bpw": projected_bpw,
        },
        "tensor_records": tensor_receipts,
        "gate": gate,
        "claim_boundary": {
            "all_existing_mapped_Qwen38_27B_codes_authenticated": True,
            "finite_block_achievable_rate_upper_bound": True,
            "supplemental_ternary_codes_conservatively_uniform": True,
            "type_class_artifact_exported": False,
            "native_type_class_decoder_benchmarked": False,
            "model_weights_or_outputs_changed": False,
            "capability_preservation_newly_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
