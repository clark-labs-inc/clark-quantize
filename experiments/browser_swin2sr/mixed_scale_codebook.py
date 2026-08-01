"""Post-training mechanism probe for category-specific scale codebooks."""

from __future__ import annotations

import math

import numpy as np
import torch

from rewrite_onnx_ternary import recover_ternary_groups
from rotation_control import block_hadamard
from scale_codebook import (
    aligned_bytes,
    group_nonzero_counts,
    quantize_scale_codebook,
)


def mixed_scale_quantized_dense_state(
    materialization: dict,
    runtime_state: dict[str, torch.Tensor],
    dense_state: dict[str, torch.Tensor],
    bits_by_category: dict[str, int],
) -> tuple[dict[str, torch.Tensor], dict]:
    categories = {
        contract["category"] for contract in materialization["surfaces"].values()
    }
    if set(bits_by_category) != categories:
        raise ValueError("mixed scale map must cover every surface category")
    if any(bits < 1 or bits > 8 for bits in bits_by_category.values()):
        raise ValueError("mixed scale widths must be between one and eight bits")
    candidate = {name: value.detach().clone() for name, value in dense_state.items()}
    totals = {
        "quantized_values": 0,
        "code_bytes": 0,
        "scale_payload_bytes": 0,
        "scale_storage_bytes": 0,
        "scale_alignment_bytes": 0,
        "squared_error": 0.0,
        "squared_reference": 0.0,
    }
    by_category = {}
    for name, contract in materialization["surfaces"].items():
        source = runtime_state[name].detach().cpu().numpy()
        group_size = int(contract["group_size"])
        codes, scales, _recovery_error = recover_ternary_groups(source, group_size)
        bits = int(bits_by_category[contract["category"]])
        codebook = quantize_scale_codebook(
            scales,
            group_nonzero_counts(codes, group_size),
            bits,
        )
        expanded = np.repeat(codebook.decoded, group_size)[: codes.size]
        reconstructed = codes.astype(np.float32) * expanded
        reference = codes.astype(np.float32) * np.repeat(scales, group_size)[: codes.size]
        delta = reconstructed - reference
        runtime_tensor = torch.from_numpy(reconstructed.reshape(source.shape))
        if contract["rotation"] == "hadamard":
            runtime_tensor = block_hadamard(
                runtime_tensor, int(contract["rotation_block"])
            )
        candidate[name] = runtime_tensor.to(dtype=dense_state[name].dtype)
        code_bytes = ((codes.size + 19) // 20) * 4
        scale_bytes, alignment = aligned_bytes([
            codebook.codebook_bytes,
            codebook.index_bytes,
        ])
        totals["quantized_values"] += codes.size
        totals["code_bytes"] += code_bytes
        totals["scale_payload_bytes"] += (
            codebook.codebook_bytes + codebook.index_bytes
        )
        totals["scale_storage_bytes"] += scale_bytes
        totals["scale_alignment_bytes"] += alignment
        totals["squared_error"] += float(np.dot(delta, delta))
        totals["squared_reference"] += float(np.dot(reference, reference))
        category = by_category.setdefault(contract["category"], {
            "requested_bits": bits,
            "values": 0,
            "groups": 0,
            "scale_storage_bytes": 0,
        })
        category["values"] += int(codes.size)
        category["groups"] += int(scales.size)
        category["scale_storage_bytes"] += scale_bytes
    values = totals["quantized_values"]
    storage = {
        "codec": "category_mixed_weighted_kmeans_fp16_centers",
        "bits_by_category": dict(sorted(bits_by_category.items())),
        "quantized_values": values,
        "code_bytes": totals["code_bytes"],
        "scale_payload_bytes": totals["scale_payload_bytes"],
        "scale_alignment_bytes": totals["scale_alignment_bytes"],
        "scale_storage_bytes": totals["scale_storage_bytes"],
        "packed_code_bits_per_quantized_weight": totals["code_bytes"] * 8 / values,
        "scale_bits_per_quantized_weight": (
            totals["scale_storage_bytes"] * 8 / values
        ),
        "packed_trunk_bits_per_quantized_weight": (
            (totals["code_bytes"] + totals["scale_storage_bytes"]) * 8 / values
        ),
        "relative_weight_rmse": math.sqrt(
            totals["squared_error"] / totals["squared_reference"]
        ),
        "by_category": dict(sorted(by_category.items())),
    }
    return candidate, storage
