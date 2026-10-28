"""Exact storage accounting for row-normalized Qwen3.8-27B scales."""
from __future__ import annotations

import math
from typing import Sequence

from .qwen38_storage_budget import (
    TensorSpec,
    align_up,
)


def row_normalized_scale_ledger(
    tensors: Sequence[TensorSpec],
    base_codebook4_ledger: dict,
) -> dict:
    """Add one FP16 row anchor to the existing group-128 codebook4 layout.

    The supplied ledger must be the complete base-3/group-128/4-bit strategy.
    Recomputing each tensor's aligned scale section avoids treating raw payload
    bytes as if section padding were free.
    """
    strategy = base_codebook4_ledger.get("strategy", {})
    if strategy != {
        "name": "base3_group128_codebook4",
        "codec": "base3",
        "group_size": 128,
        "group_semantics": "fixed",
        "scale_bits": 4,
        "scale_codebook": True,
    }:
        raise ValueError("expected the frozen base3 group128 codebook4 ledger")
    quantized = [tensor for tensor in tensors if tensor.quantized]
    retained = [tensor for tensor in tensors if not tensor.quantized]
    expected = base_codebook4_ledger
    if (
        len(tensors) != expected["tensors"]["total"]
        or len(quantized) != expected["tensors"]["ternary"]
        or len(retained) != expected["tensors"]["retained_exact"]
        or sum(tensor.parameters for tensor in tensors) != expected["parameters"]["total"]
    ):
        raise ValueError("tensor inventory does not match the base storage ledger")

    scale_count = 0
    index_bytes = 0
    anchor_bytes = 0
    codebook_bytes = 0
    scale_section_bytes = 0
    for tensor in quantized:
        count = tensor.rows * math.ceil(tensor.columns / 128)
        indices = (count + 1) // 2
        anchors = tensor.rows * 2
        codebook = 16 * 2
        scale_count += count
        index_bytes += indices
        anchor_bytes += anchors
        codebook_bytes += codebook
        scale_section_bytes += align_up(indices + anchors + codebook)
    if (
        scale_count != expected["scales"]["count"]
        or index_bytes != expected["scales"]["index_bytes"]
        or codebook_bytes != expected["scales"]["fp16_codebook_bytes"]
    ):
        raise ValueError("recomputed group-scale payload differs from the base ledger")

    source_bytes = expected["bytes"]
    scale_payload = index_bytes + anchor_bytes + codebook_bytes
    complete_bundle = (
        source_bytes["complete_bundle"]
        - source_bytes["scale_sections_aligned"]
        + scale_section_bytes
    )
    payload = (
        source_bytes["ternary_code_payload"]
        + scale_payload
        + source_bytes["retained_exact_payload"]
    )
    aligned_weight_bytes = (
        source_bytes["ternary_code_sections_aligned"]
        + scale_section_bytes
        + source_bytes["retained_exact_sections_aligned"]
    )
    total_parameters = expected["parameters"]["total"]
    return {
        "strategy": {
            "name": "base3_group128_row_normalized_codebook4",
            "codec": "base3",
            "group_size": 128,
            "group_semantics": "fixed",
            "scale_bits": 4,
            "scale_codebook": True,
            "fp16_row_anchor": True,
        },
        "parameters": dict(expected["parameters"]),
        "tensors": dict(expected["tensors"]),
        "scales": {
            "count": scale_count,
            "index_bytes": index_bytes,
            "fp16_row_anchor_count": anchor_bytes // 2,
            "fp16_row_anchor_bytes": anchor_bytes,
            "fp16_codebook_bytes": codebook_bytes,
        },
        "bytes": {
            "ternary_code_payload": source_bytes["ternary_code_payload"],
            "ternary_code_sections_aligned": source_bytes[
                "ternary_code_sections_aligned"
            ],
            "scale_payload_anchors_and_codebooks": scale_payload,
            "scale_sections_aligned": scale_section_bytes,
            "retained_exact_payload": source_bytes["retained_exact_payload"],
            "retained_exact_sections_aligned": source_bytes[
                "retained_exact_sections_aligned"
            ],
            "alignment_padding": aligned_weight_bytes - payload,
            "tensor_table_and_layout_metadata": source_bytes[
                "tensor_table_and_layout_metadata"
            ],
            "auxiliary_model_files": source_bytes["auxiliary_model_files"],
            "complete_bundle": complete_bundle,
        },
        "effective_bits_per_total_parameter": complete_bundle * 8 / total_parameters,
        "ternary_code_payload_bits_per_ternary_parameter": expected[
            "ternary_code_payload_bits_per_ternary_parameter"
        ],
    }
