#!/usr/bin/env python3
"""Materialize exact fixed-support ternary codes for the selected Qwen3.8 hash family."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
from safetensors import safe_open
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_fixed_support_codec import (
    exact_hash_residual_component_bits_per_weight,
    pack_fixed_support_matrix,
    unpack_fixed_support_matrix,
)
from clark_quantize.qwen38_hash_residual import (
    fit_hash_ternary_residual,
    quantize_hash_coefficients_int4,
)
from clark_quantize.qwen38_hash_residual_cpu import pack_signed_int4_rows


MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
HEADER = struct.Struct("<8sIIIIHHIHHQQQI")


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
            raise RuntimeError(f"frozen fixed-support source changed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument(
        "--parent-receipt",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-hash-residual-screen-v1.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/workspace/quant-runs/qwen38-27b-fixed-support-codec-v1"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/quant-state/qwen38-27b-fixed-support-codec-v1.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("fixed-support export requires positive threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    runtime = Path(__file__).resolve().parent
    prereg_path = runtime / "qwen38_27b_fixed_support_codec_preregistration_v1.json"
    prereg = json.loads(prereg_path.read_text())
    validate_frozen_sources(prereg)
    parent_prereg_path = runtime / "qwen38_27b_hash_residual_screen_preregistration_v1.json"
    parent_prereg = json.loads(parent_prereg_path.read_text())
    parent = json.loads(args.parent_receipt.read_text())
    if parent.get("schema") != "qwen38_27b_hash_residual_screen_receipt_v1":
        raise RuntimeError("fixed-support parent receipt schema changed")
    if not parent.get("gate", {}).get("passed"):
        raise RuntimeError("fixed-support export requires a passing hash-residual parent")
    if parent.get("preregistration_sha256") != sha256_file(parent_prereg_path):
        raise RuntimeError("fixed-support parent preregistration changed")

    selected = parent["gate"]["selected_candidate"]
    candidates = {
        candidate["name"]: candidate
        for candidate in parent_prereg["candidate_family"]["candidates"]
    }
    if selected not in candidates:
        raise RuntimeError("fixed-support parent candidate is unknown")
    candidate = candidates[selected]
    family = parent_prereg["candidate_family"]
    group_size = int(family["group_size"])
    support_size = int(candidate["nonzero_per_group"])
    ratio = int(candidate["input_to_sketch_ratio"])

    download = json.loads((args.model_dir / "CLARK_DOWNLOAD_RECEIPT.json").read_text())
    if download != {
        "bytes": 55_563_006_776,
        "index_sha256": INDEX_SHA256,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "schema": "qwen38_model_download_v1",
        "shards": 18,
    }:
        raise RuntimeError("Qwen3.8-27B checkpoint identity changed")
    index_content = (args.model_dir / "model.safetensors.index.json").read_bytes()
    if hashlib.sha256(index_content).hexdigest() != INDEX_SHA256:
        raise RuntimeError("Qwen3.8-27B tensor index changed")
    weight_map = json.loads(index_content)["weight_map"]
    parent_records = {
        (record["tensor"], record["row_start"], record["row_stop"]): record
        for record in parent["records"]
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total_parameters = 0
    total_bits = 0
    parent_weighted_rate = 0.0
    for tensor_spec in parent_prereg["scope"]["tensors"]:
        name = tensor_spec["name"]
        shard = weight_map.get(name)
        if shard != tensor_spec["shard"]:
            raise RuntimeError(f"fixed-support tensor shard changed: {name}")
        start, stop = tensor_spec["row_span"]
        with safe_open(args.model_dir / shard, framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            if list(view.get_shape()) != tensor_spec["shape"]:
                raise RuntimeError(f"fixed-support tensor shape changed: {name}")
            source = view[start:stop].contiguous().reshape(stop - start, -1).float()
        if source.shape[1] % ratio:
            raise RuntimeError("fixed-support selected sketch ratio no longer divides input")
        sketch_dim = source.shape[1] // ratio
        fit = fit_hash_ternary_residual(
            source,
            sketch_dim=sketch_dim,
            hashes_per_input=int(family["hashes_per_input"]),
            seed=int(family["seed"]),
            group=group_size,
            nonzero_per_group=support_size,
            iterations=int(family["alternating_iterations"]),
        )
        parent_record = parent_records[(name, start, stop)]["candidates"][selected]
        if not math.isclose(
            fit.exported_squared_error,
            float(parent_record["squared_error"]),
            rel_tol=2e-6,
            abs_tol=1e-7,
        ):
            raise RuntimeError("fixed-support replay changed the parent error")
        codes = fit.ternary_codes.cpu().numpy()
        encoded = pack_fixed_support_matrix(
            codes, group_size=group_size, nonzero_per_group=support_size
        )
        if not np.array_equal(unpack_fixed_support_matrix(encoded), codes):
            raise RuntimeError("fixed-support payload failed exact round trip")
        coefficient_codes, coefficient_scales, _ = quantize_hash_coefficients_int4(
            fit.hash_coefficients
        )
        packed_coefficients = pack_signed_int4_rows(coefficient_codes.cpu().numpy())
        rows, columns = codes.shape
        groups_per_row = columns // group_size
        name_hash = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        header = HEADER.pack(
            b"CQ3HFS1\0",
            1,
            rows,
            columns,
            group_size,
            support_size,
            int(family["hashes_per_input"]),
            sketch_dim,
            4,
            4,
            int(family["seed"]),
            encoded.bit_length,
            name_hash,
            0,
        )
        artifact = header + encoded.payload
        safe_name = name.replace(".", "-") + f"-rows-{start}-{stop}.cq3hfs"
        artifact_path = args.artifact_dir / safe_name
        atomic_bytes(artifact_path, artifact)
        structural_bits = {
            "tensor_header": len(header) * 8,
            "fixed_support_ternary_payload_on_disk": len(encoded.payload) * 8,
            "row_anchor_FP16": rows * 16,
            "group_scale_indices_INT4": rows * groups_per_row * 4,
            "tensor_ratio_codebook_16xFP16": 16 * 16,
            "hash_coefficients_INT4": packed_coefficients.nbytes * 8,
            "hash_coefficient_row_scales_FP16": coefficient_scales.numel() * 16,
        }
        bits = sum(structural_bits.values())
        parameters = codes.size
        projected_without_header_codebook = exact_hash_residual_component_bits_per_weight(
            input_dim=columns,
            sketch_dim=sketch_dim,
            nonzero_per_group=support_size,
            group_size=group_size,
        )
        parent_rate = float(parent_record["projected_component_bits_per_weight"])
        record = {
            "tensor": name,
            "row_start": start,
            "row_stop": stop,
            "rows": rows,
            "columns": columns,
            "parameters": parameters,
            "sketch_dim": sketch_dim,
            "group_width_bits": encoded.group_width_bits,
            "ternary_payload_bit_length": encoded.bit_length,
            "ternary_payload_bytes": len(encoded.payload),
            "artifact": str(artifact_path),
            "artifact_bytes": len(artifact),
            "artifact_sha256": sha256_file(artifact_path),
            "coefficient_codes_sha256": hashlib.sha256(packed_coefficients.tobytes()).hexdigest(),
            "structural_bits": structural_bits,
            "structural_projected_bits_per_weight": bits / parameters,
            "exact_component_bits_per_weight_without_header_codebook": projected_without_header_codebook,
            "parent_entropy_projected_bits_per_weight": parent_rate,
            "bits_per_weight_reduction_vs_parent_projection": parent_rate - bits / parameters,
            "exact_round_trip": True,
            "parent_exported_squared_error_replayed": True,
        }
        records.append(record)
        total_parameters += parameters
        total_bits += bits
        parent_weighted_rate += parent_rate * parameters

    exact_rate = total_bits / total_parameters
    parent_rate = parent_weighted_rate / total_parameters
    reduction = parent_rate - exact_rate
    gate_spec = prereg["gate"]
    gate = {
        **gate_spec,
        "selected_candidate": selected,
        "observed_tensor_spans": len(records),
        "observed_parameters": total_parameters,
        "observed_structural_projected_bits_per_weight": exact_rate,
        "observed_parent_entropy_projected_bits_per_weight": parent_rate,
        "observed_bits_per_weight_reduction": reduction,
        "observed_all_payloads_round_trip": all(record["exact_round_trip"] for record in records),
    }
    gate["passed"] = bool(
        len(records) == int(gate_spec["required_tensor_spans"])
        and total_parameters == int(gate_spec["required_parameters"])
        and gate["observed_all_payloads_round_trip"]
        and exact_rate <= float(gate_spec["maximum_structural_projected_bits_per_weight"])
        and reduction >= float(gate_spec["minimum_bits_per_weight_reduction_vs_parent_projection"])
    )
    receipt = {
        "schema": "qwen38_27b_fixed_support_codec_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "complete": True,
        "terminal": True,
        "preregistration_sha256": sha256_file(prereg_path),
        "parent_receipt_sha256": sha256_file(args.parent_receipt),
        "records": records,
        "gate": gate,
        "claim_boundary": {
            "real_Qwen38_27B_weights_measured": True,
            "selected_parent_candidate_replayed": True,
            "exact_fixed_support_ternary_payloads_materialized": True,
            "all_structural_side_channel_bytes_counted": True,
            "row_normalized_scale_quality_proven": False,
            "complete_model_storage_proven": False,
            "capability_preservation_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
