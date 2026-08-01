#!/usr/bin/env python3
"""Inventory PrismML Bonsai Image's packed transformer without executing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHUNK_BYTES = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def retained_category(name: str) -> str:
    for prefix in (
        "double_stream_modulation_img",
        "double_stream_modulation_txt",
        "single_stream_modulation",
        "context_embedder",
        "time_guidance_embed",
        "x_embedder",
        "norm_out",
        "proj_out",
    ):
        if name.startswith(prefix):
            return prefix
    if ".norm_" in name:
        return "attention_norms"
    return "other"


def packed_code_histogram(tensor: torch.Tensor) -> list[int]:
    array = tensor.detach().cpu().numpy().reshape(-1)
    counts = np.zeros(4, dtype=np.int64)
    for start in range(0, array.size, CHUNK_BYTES):
        chunk = array[start : start + CHUNK_BYTES]
        for shift in (0, 2, 4, 6):
            counts += np.bincount((chunk >> shift) & 0x3, minlength=4)
    return [int(value) for value in counts]


def entropy_bits(counts: list[int]) -> float:
    total = sum(counts)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts
        if count
    )


def inspect(checkpoint: Path, expected_sha256: str | None) -> dict[str, Any]:
    checkpoint_sha256 = sha256(checkpoint)
    if expected_sha256 and checkpoint_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {checkpoint_sha256} != {expected_sha256}"
        )

    state = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(state, dict):
        raise TypeError(f"expected state dictionary, got {type(state).__name__}")

    prefixes = sorted(key[: -len(".W_q")] for key in state if key.endswith(".W_q"))
    packed_histogram = [0, 0, 0, 0]
    quantized_parameters = 0
    packed_weight_bytes = 0
    scale_bytes = 0
    zero_bytes = 0
    metadata_bytes = 0
    scale_elements = 0
    scales_are_bf16_exact = True
    scales_are_fp16_exact = True
    zeros_are_negative_scales = True
    per_linear: list[dict[str, Any]] = []

    auxiliary_keys: set[str] = set()
    for prefix in prefixes:
        keys = {
            suffix: f"{prefix}.{suffix}"
            for suffix in ("W_q", "scales", "zeros", "metadata", "orig_shape")
        }
        missing = [key for key in keys.values() if key not in state]
        if missing:
            raise ValueError(f"{prefix} is missing packed tensors: {missing}")
        auxiliary_keys.update(keys.values())

        packed = state[keys["W_q"]]
        scales = state[keys["scales"]]
        zeros = state[keys["zeros"]]
        shape = [int(value) for value in state[keys["orig_shape"]].tolist()]
        logical_parameters = product(shape)
        histogram = packed_code_histogram(packed)
        if sum(histogram) != logical_parameters:
            raise ValueError(
                f"{prefix} contains {sum(histogram)} codes for {logical_parameters} weights"
            )
        for index, count in enumerate(histogram):
            packed_histogram[index] += count

        scale_bf16_exact = torch.equal(scales, scales.to(torch.bfloat16).float())
        scale_fp16_exact = torch.equal(scales, scales.to(torch.float16).float())
        zero_relation = torch.equal(zeros, -scales)
        scales_are_bf16_exact &= scale_bf16_exact
        scales_are_fp16_exact &= scale_fp16_exact
        zeros_are_negative_scales &= zero_relation

        quantized_parameters += logical_parameters
        packed_weight_bytes += tensor_bytes(packed)
        scale_bytes += tensor_bytes(scales)
        zero_bytes += tensor_bytes(zeros)
        metadata_bytes += tensor_bytes(state[keys["metadata"]])
        metadata_bytes += tensor_bytes(state[keys["orig_shape"]])
        scale_elements += scales.numel()
        per_linear.append(
            {
                "prefix": prefix,
                "original_shape": shape,
                "logical_parameters": logical_parameters,
                "packed_shape": list(packed.shape),
                "packed_bytes": tensor_bytes(packed),
                "scale_elements": scales.numel(),
                "scale_dtype": str(scales.dtype).removeprefix("torch."),
                "scale_min": float(scales.min()),
                "scale_max": float(scales.max()),
                "scale_bf16_exact": scale_bf16_exact,
                "scale_fp16_exact": scale_fp16_exact,
                "zeros_equal_negative_scales": zero_relation,
                "code_histogram": histogram,
            }
        )

    retained_bytes: Counter[str] = Counter()
    retained_parameters: Counter[str] = Counter()
    retained_tensors: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    tensor_storage_bytes = 0
    retained_parameter_count = 0
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        size = tensor_bytes(value)
        tensor_storage_bytes += size
        dtype_bytes[str(value.dtype).removeprefix("torch.")] += size
        if name in auxiliary_keys:
            continue
        category = retained_category(name)
        retained_bytes[category] += size
        retained_parameters[category] += value.numel()
        retained_tensors[category] += 1
        retained_parameter_count += value.numel()

    logical_parameters = quantized_parameters + retained_parameter_count
    actual_low_bit_bytes = packed_weight_bytes + scale_bytes + zero_bytes + metadata_bytes
    int2_derived_zero_bf16_scale_bytes = (
        packed_weight_bytes + scale_elements * 2 + metadata_bytes
    )
    conceptual_entropy_bf16_scale_bits = (
        quantized_parameters * math.log2(3) + scale_elements * 16
    )
    high_precision_bytes = sum(retained_bytes.values())

    return {
        "format": "prism-bonsai-image-packed-weight-inventory-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_file_bytes": checkpoint.stat().st_size,
        "safe_load": {"mmap": True, "weights_only": True},
        "tensor_count": sum(torch.is_tensor(value) for value in state.values()),
        "tensor_storage_bytes": tensor_storage_bytes,
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "quantized_linears": len(prefixes),
        "quantized_logical_parameters": quantized_parameters,
        "retained_logical_parameters": retained_parameter_count,
        "logical_parameters": logical_parameters,
        "retained_parameter_fraction": retained_parameter_count / logical_parameters,
        "packed_codes": {
            "histogram": packed_histogram,
            "fractions": [count / sum(packed_histogram) for count in packed_histogram],
            "shannon_entropy_bits_per_weight": entropy_bits(packed_histogram),
            "unused_code_3": packed_histogram[3] == 0,
        },
        "scale_and_zero_storage": {
            "group_size": 128,
            "scale_elements": scale_elements,
            "scale_bytes": scale_bytes,
            "zero_bytes": zero_bytes,
            "metadata_bytes": metadata_bytes,
            "scales_are_bf16_exact": scales_are_bf16_exact,
            "scales_are_fp16_exact": scales_are_fp16_exact,
            "zeros_are_negative_scales": zeros_are_negative_scales,
        },
        "storage": {
            "packed_weight_bytes": packed_weight_bytes,
            "actual_low_bit_bytes": actual_low_bit_bytes,
            "retained_high_precision_bytes": high_precision_bytes,
            "actual_tensor_bits_per_logical_parameter": 8
            * tensor_storage_bytes
            / logical_parameters,
            "actual_file_bits_per_logical_parameter": 8
            * checkpoint.stat().st_size
            / logical_parameters,
            "int2_derived_zero_bf16_scale_tensor_bytes": (
                int2_derived_zero_bf16_scale_bytes + high_precision_bytes
            ),
            "int2_derived_zero_bf16_scale_bits_per_logical_parameter": 8
            * (int2_derived_zero_bf16_scale_bytes + high_precision_bytes)
            / logical_parameters,
            "conceptual_entropy_bf16_scale_tensor_bytes": math.ceil(
                conceptual_entropy_bf16_scale_bits / 8
            )
            + high_precision_bytes,
            "conceptual_entropy_bf16_scale_bits_per_logical_parameter": (
                conceptual_entropy_bf16_scale_bits + high_precision_bytes * 8
            )
            / logical_parameters,
        },
        "retained_high_precision": {
            category: {
                "bytes": retained_bytes[category],
                "logical_parameters": retained_parameters[category],
                "tensor_count": retained_tensors[category],
            }
            for category in sorted(retained_bytes)
        },
        "per_linear": per_linear,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect(args.checkpoint, args.expected_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "checkpoint_sha256",
        "quantized_linears",
        "logical_parameters",
        "retained_parameter_fraction",
        "packed_codes",
        "scale_and_zero_storage",
        "storage",
    )}, indent=2))


if __name__ == "__main__":
    main()
