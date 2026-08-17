"""Complete structural byte ledger for a fixed-support Qwen3.8 hash residual."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .qwen38_fixed_support_codec import fixed_support_width_bits
from .qwen38_storage_budget import (
    LAYOUT_ALIGNMENT,
    TensorSpec,
    align_up,
    layout_metadata_bytes,
)


@dataclass(frozen=True)
class HashResidualStorageStrategy:
    name: str
    group_size: int
    nonzero_per_group: int
    input_to_sketch_ratio: int
    hashes_per_input: int = 4
    coefficient_bits: int = 4
    scale_index_bits: int = 4
    tensor_codec_header_bytes: int = 64
    tensor_scale_codebook_entries: int = 16

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.group_size <= 0
            or not 0 <= self.nonzero_per_group <= self.group_size
            or self.input_to_sketch_ratio <= 0
            or self.hashes_per_input <= 0
            or self.coefficient_bits != 4
            or self.scale_index_bits != 4
            or self.tensor_codec_header_bytes < 0
            or self.tensor_scale_codebook_entries != 16
        ):
            raise ValueError("invalid hash-residual storage strategy")

    @property
    def group_width_bits(self) -> int:
        return fixed_support_width_bits(self.group_size, self.nonzero_per_group)


def tensor_hash_eligible(tensor: TensorSpec, strategy: HashResidualStorageStrategy) -> bool:
    return bool(
        tensor.quantized
        and tensor.columns % strategy.group_size == 0
        and tensor.columns % strategy.input_to_sketch_ratio == 0
        and tensor.columns // strategy.input_to_sketch_ratio <= 65535
    )


def hash_tensor_storage_ledger(
    tensor: TensorSpec,
    strategy: HashResidualStorageStrategy,
) -> dict:
    if not tensor_hash_eligible(tensor, strategy):
        raise ValueError("tensor is not eligible for the fixed-support hash representation")
    groups_per_row = tensor.columns // strategy.group_size
    sketch_dim = tensor.columns // strategy.input_to_sketch_ratio
    ternary_bits = tensor.rows * groups_per_row * strategy.group_width_bits
    ternary_bytes = math.ceil(ternary_bits / 8)
    row_anchor_bytes = tensor.rows * 2
    scale_index_bytes = math.ceil(
        tensor.rows * groups_per_row * strategy.scale_index_bits / 8
    )
    scale_codebook_bytes = strategy.tensor_scale_codebook_entries * 2
    coefficient_bytes = tensor.rows * math.ceil(
        sketch_dim * strategy.coefficient_bits / 8
    )
    coefficient_scale_bytes = tensor.rows * 2
    payload_bytes = (
        strategy.tensor_codec_header_bytes
        + ternary_bytes
        + row_anchor_bytes
        + scale_index_bytes
        + scale_codebook_bytes
        + coefficient_bytes
        + coefficient_scale_bytes
    )
    aligned_bytes = align_up(payload_bytes, LAYOUT_ALIGNMENT)
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "parameters": tensor.parameters,
        "rows": tensor.rows,
        "columns": tensor.columns,
        "groups_per_row": groups_per_row,
        "sketch_dim": sketch_dim,
        "group_width_bits": strategy.group_width_bits,
        "bytes": {
            "tensor_codec_header": strategy.tensor_codec_header_bytes,
            "fixed_support_ternary_payload": ternary_bytes,
            "row_anchors_FP16": row_anchor_bytes,
            "group_scale_indices_INT4": scale_index_bytes,
            "tensor_ratio_codebook_16xFP16": scale_codebook_bytes,
            "hash_coefficients_INT4": coefficient_bytes,
            "hash_coefficient_row_scales_FP16": coefficient_scale_bytes,
            "payload": payload_bytes,
            "aligned_section": aligned_bytes,
            "alignment_padding": aligned_bytes - payload_bytes,
        },
        "structural_bits_per_weight": aligned_bytes * 8 / tensor.parameters,
    }


def complete_hash_storage_ledger(
    tensors: Sequence[TensorSpec],
    strategy: HashResidualStorageStrategy,
    *,
    auxiliary_bytes: int = 0,
) -> dict:
    if not tensors or auxiliary_bytes < 0:
        raise ValueError("complete hash ledger requires tensors and nonnegative auxiliary bytes")
    eligible = []
    retained = []
    projection_shapes: dict[tuple[int, int], dict] = {}
    for tensor in tensors:
        if tensor_hash_eligible(tensor, strategy):
            record = hash_tensor_storage_ledger(tensor, strategy)
            eligible.append(record)
            key = (tensor.columns, record["sketch_dim"])
            projection_shapes.setdefault(
                key,
                {
                    "input_dim": tensor.columns,
                    "sketch_dim": record["sketch_dim"],
                    "tensor_count": 0,
                    "mapping_bytes": (
                        tensor.columns * strategy.hashes_per_input * (2 + 1)
                    ),
                    "per_thread_scratch_bytes": record["sketch_dim"] * 4,
                },
            )["tensor_count"] += 1
        else:
            retained.append(
                {
                    "name": tensor.name,
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype,
                    "parameters": tensor.parameters,
                    "payload_bytes": tensor.retained_bytes,
                    "aligned_section_bytes": align_up(tensor.retained_bytes),
                    "reason": (
                        "rank_below_2"
                        if not tensor.quantized
                        else "flattened_input_not_group_and_ratio_aligned"
                    ),
                }
            )
    total_parameters = sum(tensor.parameters for tensor in tensors)
    eligible_parameters = sum(record["parameters"] for record in eligible)
    retained_parameters = total_parameters - eligible_parameters
    eligible_payload = sum(record["bytes"]["payload"] for record in eligible)
    eligible_aligned = sum(record["bytes"]["aligned_section"] for record in eligible)
    retained_payload = sum(record["payload_bytes"] for record in retained)
    retained_aligned = sum(record["aligned_section_bytes"] for record in retained)
    metadata = layout_metadata_bytes(tensors)
    complete_bytes = eligible_aligned + retained_aligned + metadata + auxiliary_bytes
    active_mapping_bytes = sum(record["mapping_bytes"] for record in projection_shapes.values())
    return {
        "strategy": {
            "name": strategy.name,
            "group_size": strategy.group_size,
            "nonzero_per_group": strategy.nonzero_per_group,
            "group_width_bits": strategy.group_width_bits,
            "input_to_sketch_ratio": strategy.input_to_sketch_ratio,
            "hashes_per_input": strategy.hashes_per_input,
            "coefficient_bits": strategy.coefficient_bits,
            "scale_index_bits": strategy.scale_index_bits,
        },
        "parameters": {
            "total": total_parameters,
            "fixed_support_hash": eligible_parameters,
            "retained_exact": retained_parameters,
        },
        "tensors": {
            "total": len(tensors),
            "fixed_support_hash": len(eligible),
            "retained_exact": len(retained),
        },
        "bytes": {
            "fixed_support_hash_payload": eligible_payload,
            "fixed_support_hash_sections_aligned": eligible_aligned,
            "retained_exact_payload": retained_payload,
            "retained_exact_sections_aligned": retained_aligned,
            "tensor_table_and_layout_metadata": metadata,
            "auxiliary_model_files": auxiliary_bytes,
            "complete_bundle": complete_bytes,
            "alignment_padding": (
                eligible_aligned + retained_aligned - eligible_payload - retained_payload
            ),
        },
        "effective_bits_per_total_parameter": complete_bytes * 8 / total_parameters,
        "eligible_payload_bits_per_eligible_parameter": (
            eligible_payload * 8 / eligible_parameters if eligible_parameters else None
        ),
        "active_compute_metadata_excluded_from_model_storage": {
            "unique_projection_shapes": len(projection_shapes),
            "mapping_bytes": active_mapping_bytes,
            "projection_shapes": list(projection_shapes.values()),
            "maximum_per_thread_sketch_scratch_bytes": max(
                (record["per_thread_scratch_bytes"] for record in projection_shapes.values()),
                default=0,
            ),
        },
        "eligible_tensor_records": eligible,
        "retained_tensor_records": retained,
    }
