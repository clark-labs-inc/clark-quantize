"""Exact storage accounting for Clark-packed model artifacts."""

from __future__ import annotations

import math


def build_artifact_ledger(
    *,
    mode: str,
    binary_bytes: int,
    parameter_values: int,
    state_values: int,
    quantized_values: int,
    code_bytes: int,
    scale_bytes: int,
    raw_bytes: int,
    offset_bytes: int = 0,
) -> dict:
    if parameter_values <= 0 or state_values <= 0:
        raise ValueError("model and state value counts must be positive")
    if quantized_values <= 0:
        raise ValueError("quantized value count must be positive")
    accounted = code_bytes + scale_bytes + offset_bytes + raw_bytes
    if accounted > binary_bytes:
        raise ValueError("tensor payload exceeds binary size")
    ideal_code_bits = math.log2(3) if mode == "ternary" else 1.0
    return {
        "parameter_values": parameter_values,
        "state_values": state_values,
        "quantized_values": quantized_values,
        "quantized_fraction_of_parameters": quantized_values / parameter_values,
        "code_bytes": code_bytes,
        "scale_bytes": scale_bytes,
        "offset_bytes": offset_bytes,
        "raw_tensor_bytes": raw_bytes,
        "alignment_bytes": binary_bytes - accounted,
        "binary_bytes": binary_bytes,
        "ideal_code_bits_per_quantized_weight": ideal_code_bits,
        "packed_code_bits_per_quantized_weight": code_bytes * 8 / quantized_values,
        "scale_bits_per_quantized_weight": scale_bytes * 8 / quantized_values,
        "packed_trunk_bits_per_quantized_weight": (
            (code_bytes + scale_bytes + offset_bytes) * 8 / quantized_values
        ),
        "whole_model_data_bits_per_parameter": (
            binary_bytes * 8 / parameter_values
        ),
    }
