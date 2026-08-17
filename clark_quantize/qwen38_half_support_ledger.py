"""Conservative whole-checkpoint ledger for half-support ternary plus A R.

For any even group width ``n``, exactly ``n/2`` nonzeros can be represented by
an ``n``-bit support mask and ``n/2`` support-ordered signs.  The runtime form
therefore costs exactly 1.5 raw bits per weight without combinatorial decode.
This generalizes mask64 to even tail groups such as Qwen vision width 4304.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from .qwen38_storage_budget import (
    TensorSpec,
    align_up,
    inventory_from_safetensors,
    layout_metadata_bytes,
)


GROUP_SIZE = 128
MINIMUM_COMPRESSED_COLUMNS = 128
TENSOR_HEADER_BYTES = 64


def half_support_group_bytes(width: int) -> int:
    if width <= 0 or width > GROUP_SIZE or width % 2:
        raise ValueError("half-support groups require a positive even width <= 128")
    return (width + width // 2 + 7) // 8


def mask_payload_bytes(*, rows: int, columns: int) -> int:
    if rows <= 0 or columns <= 0 or columns % 2:
        raise ValueError("half-support mask matrices require positive even dimensions")
    full_groups, tail = divmod(columns, GROUP_SIZE)
    row_bytes = full_groups * half_support_group_bytes(GROUP_SIZE)
    if tail:
        row_bytes += half_support_group_bytes(tail)
    return rows * row_bytes


def _compressed_tensor_ledger(tensor: TensorSpec) -> dict:
    rows = tensor.rows
    columns = tensor.columns
    groups = math.ceil(columns / GROUP_SIZE)
    sketch_dim = groups
    bucket_bits = max(1, math.ceil(math.log2(sketch_dim)))
    tail_width = columns % GROUP_SIZE
    components = {
        "half_support_mask_and_sign_codes": mask_payload_bytes(
            rows=rows, columns=columns
        ),
        "scale4_indices_row_anchors_and_codebook": (
            (rows * groups * 4 + 7) // 8 + rows * 2 + 16 * 2
        ),
        "signed_int4_hash_coefficients_and_row_scales": (
            rows * ((sketch_dim + 1) // 2) + rows * 2
        ),
        "bitpacked_learned_route": (
            columns * (bucket_bits + 1) + 7
        ) // 8,
        "tensor_header": TENSOR_HEADER_BYTES,
    }
    aligned = {name: align_up(size) for name, size in components.items()}
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "source_dtype": tensor.dtype,
        "parameters": tensor.parameters,
        "policy": "half_support_mask_plus_learned_hash",
        "group_size": GROUP_SIZE,
        "full_groups_per_row": columns // GROUP_SIZE,
        "tail_width": tail_width,
        "tail_nonzero_count": tail_width // 2,
        "sketch_dim": sketch_dim,
        "routing_bucket_bits": bucket_bits,
        "raw_mask_bits_per_weight": (
            components["half_support_mask_and_sign_codes"] * 8
            / tensor.parameters
        ),
        "payload_bytes": components,
        "aligned_section_bytes": aligned,
        "complete_tensor_bytes": sum(aligned.values()),
    }


def _retained_tensor_ledger(tensor: TensorSpec, reason: str) -> dict:
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "source_dtype": tensor.dtype,
        "parameters": tensor.parameters,
        "policy": "retained_exact",
        "reason": reason,
        "payload_bytes": tensor.retained_bytes,
        "complete_tensor_bytes": align_up(tensor.retained_bytes),
    }


def tensor_half_support_ledger(tensor: TensorSpec) -> dict:
    if not tensor.quantized:
        return _retained_tensor_ledger(tensor, "rank below two")
    if tensor.columns < MINIMUM_COMPRESSED_COLUMNS:
        return _retained_tensor_ledger(
            tensor, "input width below compression break-even policy"
        )
    if tensor.columns % 2:
        return _retained_tensor_ledger(
            tensor, "odd input width lacks an exact half-support state"
        )
    return _compressed_tensor_ledger(tensor)


def half_support_checkpoint_ledger(
    tensors: Sequence[TensorSpec],
    *,
    auxiliary_bytes: int,
) -> dict:
    if not tensors or auxiliary_bytes < 0:
        raise ValueError("non-empty inventory and non-negative auxiliary bytes required")
    records = [tensor_half_support_ledger(tensor) for tensor in tensors]
    total_parameters = sum(record["parameters"] for record in records)
    compressed = [
        record for record in records
        if record["policy"] == "half_support_mask_plus_learned_hash"
    ]
    retained = [record for record in records if record["policy"] == "retained_exact"]
    compressed_bytes = sum(record["complete_tensor_bytes"] for record in compressed)
    retained_bytes = sum(record["complete_tensor_bytes"] for record in retained)
    metadata_bytes = layout_metadata_bytes(tensors)
    complete_bytes = (
        compressed_bytes + retained_bytes + metadata_bytes + auxiliary_bytes
    )
    compressed_parameters = sum(record["parameters"] for record in compressed)
    retained_parameters = sum(record["parameters"] for record in retained)
    if compressed_parameters + retained_parameters != total_parameters:
        raise AssertionError("half-support ledger parameter denominator changed")
    tail_records = [record for record in compressed if record["tail_width"]]
    return {
        "schema": "qwen38_half_support_checkpoint_ledger_v1",
        "policy": {
            "compressed": (
                "rank-two-or-higher tensors with even flattened input width >= 128"
            ),
            "retained": "rank-one, short, or odd-width tensors in source dtype",
            "route_sharing_assumed": False,
            "alignment": "every component of every tensor independently aligned to 64 bytes",
            "tail_representation": (
                "even tail n uses n support bits and n/2 support-ordered sign bits"
            ),
        },
        "parameters": {
            "total": total_parameters,
            "compressed": compressed_parameters,
            "retained_exact": retained_parameters,
        },
        "tensors": {
            "total": len(records),
            "compressed": len(compressed),
            "retained_exact": len(retained),
            "compressed_with_even_tail": len(tail_records),
        },
        "bytes": {
            "compressed_aligned_sections": compressed_bytes,
            "retained_exact_aligned_sections": retained_bytes,
            "tensor_table_and_layout_metadata": metadata_bytes,
            "auxiliary_model_files": auxiliary_bytes,
            "complete_bundle": complete_bytes,
        },
        "effective_bits_per_total_parameter": complete_bytes * 8 / total_parameters,
        "compressed_component_bits_per_compressed_parameter": (
            compressed_bytes * 8 / compressed_parameters
        ),
        "records": records,
        "claim_boundary": {
            "projected_layout_only": True,
            "real_tensor_inventory_required": True,
            "tail_codec_or_kernel_implemented": False,
            "artifact_exported": False,
            "real_weight_approximation_quality_proven": False,
            "capability_preservation_proven": False,
            "token_throughput_proven": False,
            "scientific_breakthrough_proven": False,
        },
    }


def build_half_support_checkpoint_receipt(model_dir: Path) -> dict:
    tensors, source = inventory_from_safetensors(Path(model_dir))
    return {
        "model_dir": str(Path(model_dir).resolve()),
        "source": source,
        "ledger": half_support_checkpoint_ledger(
            tensors, auxiliary_bytes=source["auxiliary_bytes"]
        ),
    }
