"""Deterministic packed codebooks for positive per-group ternary scales."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rewrite_onnx_ternary import recover_ternary_groups
from rotation_control import block_hadamard


@dataclass(frozen=True)
class ScaleCodebook:
    bit_width: int
    centers: np.ndarray
    indices: np.ndarray
    decoded: np.ndarray
    index_bytes: int
    codebook_bytes: int


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    if cumulative[-1] <= 0:
        cumulative = np.arange(1, values.size + 1, dtype=np.float64)
    targets = quantiles * cumulative[-1]
    return ordered[np.searchsorted(cumulative, targets, side="left").clip(0, values.size - 1)]


def quantize_scale_codebook(
    scales: np.ndarray,
    nonzero_weights: np.ndarray,
    bit_width: int,
    *,
    iterations: int = 64,
) -> ScaleCodebook:
    """Fit a deterministic weighted 1-D codebook and round centers to FP16."""

    values = np.asarray(scales, dtype=np.float32).reshape(-1)
    weights = np.asarray(nonzero_weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.shape != weights.shape:
        raise ValueError("scale values and weights must be non-empty and paired")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("ternary scales must be finite and non-negative")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("scale weights must be finite and non-negative")
    if bit_width < 1 or bit_width > 8:
        raise ValueError("scale codebook width must be between one and eight bits")
    level_count = min(1 << bit_width, values.size)
    quantiles = (np.arange(level_count, dtype=np.float64) + 0.5) / level_count
    centers = _weighted_quantiles(values, weights, quantiles).astype(np.float64)
    for _ in range(iterations):
        indices = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        updated = centers.copy()
        for level in range(level_count):
            members = indices == level
            member_weight = weights[members]
            if members.any() and member_weight.sum() > 0:
                updated[level] = np.average(values[members], weights=member_weight)
        updated.sort()
        if np.array_equal(updated, centers):
            break
        centers = updated
    stored_centers = centers.astype(np.float16)
    stored_centers = np.unique(stored_centers)
    decoded_centers = stored_centers.astype(np.float32)
    indices = np.abs(values[:, None] - decoded_centers[None, :]).argmin(axis=1)
    actual_bits = 0 if decoded_centers.size == 1 else math.ceil(math.log2(decoded_centers.size))
    index_bytes = (values.size * actual_bits + 7) // 8
    return ScaleCodebook(
        bit_width=actual_bits,
        centers=stored_centers,
        indices=indices.astype(np.uint8),
        decoded=decoded_centers[indices],
        index_bytes=index_bytes,
        codebook_bytes=int(stored_centers.nbytes),
    )


def group_nonzero_counts(codes: np.ndarray, group_size: int) -> np.ndarray:
    flat = np.asarray(codes, dtype=np.int8).reshape(-1)
    return np.asarray([
        np.count_nonzero(flat[start : start + group_size])
        for start in range(0, flat.size, group_size)
    ], dtype=np.float64)


def aligned_bytes(payload_sizes: list[int], alignment: int = 4) -> tuple[int, int]:
    offset = 0
    payload = 0
    for size in payload_sizes:
        if size < 0:
            raise ValueError("payload sizes must be non-negative")
        offset += (-offset) % alignment
        offset += size
        payload += size
    return offset, offset - payload


def fp16_scale_dense_state(
    materialization: dict,
    runtime_state: dict[str, torch.Tensor],
    dense_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Rebuild the current browser baseline with its exact FP16 scale storage."""

    candidate = {name: value.detach().clone() for name, value in dense_state.items()}
    quantized_values = 0
    code_bytes = 0
    scale_storage_bytes = 0
    squared_error = 0.0
    squared_reference = 0.0
    for name, contract in materialization["surfaces"].items():
        source = runtime_state[name].detach().cpu().numpy()
        group_size = int(contract["group_size"])
        codes, scales, _recovery_error = recover_ternary_groups(source, group_size)
        decoded = scales.astype(np.float16).astype(np.float32)
        expanded = np.repeat(decoded, group_size)[: codes.size]
        reconstructed = codes.astype(np.float32) * expanded
        reference = codes.astype(np.float32) * np.repeat(scales, group_size)[: codes.size]
        delta = reconstructed - reference
        squared_error += float(np.dot(delta, delta))
        squared_reference += float(np.dot(reference, reference))
        runtime_tensor = torch.from_numpy(reconstructed.reshape(source.shape))
        if contract["rotation"] == "hadamard":
            runtime_tensor = block_hadamard(
                runtime_tensor,
                int(contract["rotation_block"]),
            )
        candidate[name] = runtime_tensor.to(dtype=dense_state[name].dtype)
        code_bytes += ((codes.size + 19) // 20) * 4
        scale_storage_bytes += int(decoded.size * 2)
        quantized_values += codes.size
    return candidate, {
        "codec": "fp16_per_group",
        "requested_scale_index_bits": 16,
        "quantized_values": quantized_values,
        "code_bytes": code_bytes,
        "scale_payload_bytes": scale_storage_bytes,
        "scale_alignment_bytes": 0,
        "scale_storage_bytes": scale_storage_bytes,
        "packed_code_bits_per_quantized_weight": code_bytes * 8 / quantized_values,
        "scale_bits_per_quantized_weight": scale_storage_bytes * 8 / quantized_values,
        "packed_trunk_bits_per_quantized_weight": (
            (code_bytes + scale_storage_bytes) * 8 / quantized_values
        ),
        "relative_weight_rmse": math.sqrt(squared_error / squared_reference),
    }


def scale_quantized_dense_state(
    materialization: dict,
    runtime_state: dict[str, torch.Tensor],
    dense_state: dict[str, torch.Tensor],
    bit_width: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Rebuild a plain-model state after codebook quantizing runtime scales."""

    candidate = {name: value.detach().clone() for name, value in dense_state.items()}
    tensor_records = []
    quantized_values = 0
    code_bytes = 0
    scale_payload_bytes = 0
    scale_storage_bytes = 0
    scale_alignment_bytes = 0
    squared_error = 0.0
    squared_reference = 0.0
    for name, contract in materialization["surfaces"].items():
        source = runtime_state[name].detach().cpu().numpy()
        group_size = int(contract["group_size"])
        codes, scales, recovery_error = recover_ternary_groups(source, group_size)
        nonzero = group_nonzero_counts(codes, group_size)
        codebook = quantize_scale_codebook(scales, nonzero, bit_width)
        expanded = np.repeat(codebook.decoded, group_size)[: codes.size]
        reconstructed = codes.astype(np.float32) * expanded
        reference = codes.astype(np.float32) * np.repeat(scales, group_size)[: codes.size]
        delta = reconstructed - reference
        squared_error += float(np.dot(delta, delta))
        squared_reference += float(np.dot(reference, reference))
        runtime_tensor = torch.from_numpy(reconstructed.reshape(source.shape))
        if contract["rotation"] == "hadamard":
            runtime_tensor = block_hadamard(
                runtime_tensor,
                int(contract["rotation_block"]),
            )
        candidate[name] = runtime_tensor.to(dtype=dense_state[name].dtype)
        tensor_code_bytes = ((codes.size + 19) // 20) * 4
        scale_bytes, alignment = aligned_bytes([
            codebook.codebook_bytes,
            codebook.index_bytes,
        ])
        code_bytes += tensor_code_bytes
        scale_payload_bytes += codebook.codebook_bytes + codebook.index_bytes
        scale_storage_bytes += scale_bytes
        scale_alignment_bytes += alignment
        quantized_values += codes.size
        tensor_records.append({
            "name": name,
            "values": int(codes.size),
            "groups": int(scales.size),
            "requested_bits": bit_width,
            "stored_bits": codebook.bit_width,
            "codebook_levels": int(codebook.centers.size),
            "codebook_bytes": codebook.codebook_bytes,
            "index_bytes": codebook.index_bytes,
            "aligned_scale_bytes": scale_bytes,
            "hard_ternary_recovery_max_error": recovery_error,
        })
    return candidate, {
        "codec": f"weighted_kmeans_fp16_centers_q{bit_width}",
        "requested_scale_index_bits": bit_width,
        "quantized_values": quantized_values,
        "code_bytes": code_bytes,
        "scale_payload_bytes": scale_payload_bytes,
        "scale_alignment_bytes": scale_alignment_bytes,
        "scale_storage_bytes": scale_storage_bytes,
        "packed_code_bits_per_quantized_weight": code_bytes * 8 / quantized_values,
        "scale_bits_per_quantized_weight": scale_storage_bytes * 8 / quantized_values,
        "packed_trunk_bits_per_quantized_weight": (
            (code_bytes + scale_storage_bytes) * 8 / quantized_values
        ),
        "relative_weight_rmse": math.sqrt(squared_error / squared_reference),
        "tensors": tensor_records,
    }
