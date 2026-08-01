#!/usr/bin/env python3
"""Prototype a portable ONNX bridge for base-3 packed ternary weights.

ONNX Runtime's MatMulNBits kernel currently provides optimized 4-bit and
8-bit paths, not ternary weights. This bridge keeps five trits in each byte
and reconstructs the weight tensor from ordinary ONNX operators. It is a
portable correctness and artifact-size fallback, not the final fast kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TRITS_PER_BYTE = 5
TRIT_VALUES = 3 ** TRITS_PER_BYTE


def pack_trits_5(codes: np.ndarray) -> tuple[np.ndarray, int]:
    flat = np.asarray(codes, dtype=np.int8).reshape(-1)
    if not np.isin(flat, (-1, 0, 1)).all():
        raise ValueError("ternary codes must contain only -1, 0, and 1")
    pad = (-flat.size) % TRITS_PER_BYTE
    digits = (flat + 1).astype(np.uint8)
    if pad:
        digits = np.pad(digits, (0, pad), constant_values=1)
    digits = digits.reshape(-1, TRITS_PER_BYTE)
    powers = (3 ** np.arange(TRITS_PER_BYTE, dtype=np.uint16))[None, :]
    packed = np.sum(digits.astype(np.uint16) * powers, axis=1).astype(np.uint8)
    return packed, pad


def trit_lookup() -> np.ndarray:
    packed = np.arange(TRIT_VALUES, dtype=np.uint16)[:, None]
    powers = (3 ** np.arange(TRITS_PER_BYTE, dtype=np.uint16))[None, :]
    return ((packed // powers) % 3).astype(np.float32) - 1.0


def _int64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def build_packed_linear_model(
    *,
    codes: np.ndarray,
    scales: np.ndarray,
    bias: np.ndarray,
    out_features: int,
    in_features: int,
    group_size: int,
    output: Path,
) -> dict[str, int | float]:
    if out_features <= 0 or in_features <= 0 or group_size <= 0:
        raise ValueError("matrix dimensions and group_size must be positive")
    value_count = out_features * in_features
    flat_codes = np.asarray(codes, dtype=np.int8).reshape(-1)
    flat_scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    flat_bias = np.asarray(bias, dtype=np.float32).reshape(-1)
    expected_scales = (value_count + group_size - 1) // group_size
    if flat_codes.size != value_count:
        raise ValueError("code count does not match matrix shape")
    if flat_scales.size != expected_scales:
        raise ValueError("scale count does not match matrix and group size")
    if flat_bias.size != out_features:
        raise ValueError("bias count does not match out_features")

    packed, pad = pack_trits_5(flat_codes)
    expanded_scale_shape = [flat_scales.size, group_size]
    initializers = [
        numpy_helper.from_array(packed, "packed_codes"),
        numpy_helper.from_array(trit_lookup(), "trit_lookup"),
        numpy_helper.from_array(flat_scales.astype(np.float16), "group_scales_f16"),
        numpy_helper.from_array(flat_bias, "bias"),
        _int64("flat_shape", [-1]),
        _int64("weight_shape", [out_features, in_features]),
        _int64("scale_unsqueeze_axes", [1]),
        _int64("expanded_scale_shape", expanded_scale_shape),
        _int64("slice_starts", [0]),
        _int64("slice_ends", [value_count]),
        _int64("slice_axes", [0]),
        _int64("slice_steps", [1]),
    ]
    nodes = [
        helper.make_node("Cast", ["packed_codes"], ["code_indices"], to=TensorProto.INT32),
        helper.make_node("Gather", ["trit_lookup", "code_indices"], ["decoded_groups"]),
        helper.make_node("Reshape", ["decoded_groups", "flat_shape"], ["decoded_padded"]),
        helper.make_node(
            "Slice",
            ["decoded_padded", "slice_starts", "slice_ends", "slice_axes", "slice_steps"],
            ["decoded_codes"],
        ),
        helper.make_node("Cast", ["group_scales_f16"], ["group_scales"], to=TensorProto.FLOAT),
        helper.make_node(
            "Unsqueeze",
            ["group_scales", "scale_unsqueeze_axes"],
            ["group_scales_2d"],
        ),
        helper.make_node(
            "Expand",
            ["group_scales_2d", "expanded_scale_shape"],
            ["expanded_scales_2d"],
        ),
        helper.make_node(
            "Reshape", ["expanded_scales_2d", "flat_shape"], ["expanded_scales_padded"]
        ),
        helper.make_node(
            "Slice",
            [
                "expanded_scales_padded",
                "slice_starts",
                "slice_ends",
                "slice_axes",
                "slice_steps",
            ],
            ["expanded_scales"],
        ),
        helper.make_node("Mul", ["decoded_codes", "expanded_scales"], ["weight_flat"]),
        helper.make_node("Reshape", ["weight_flat", "weight_shape"], ["weight_out_in"]),
        helper.make_node("Transpose", ["weight_out_in"], ["weight_in_out"], perm=[1, 0]),
        helper.make_node("MatMul", ["input", "weight_in_out"], ["projection"]),
        helper.make_node("Add", ["projection", "bias"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "packed_ternary_linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", in_features])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", out_features])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="clark-labs-backdropkit",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output)
    model_bytes = output.stat().st_size
    return {
        "values": value_count,
        "packed_code_bytes": int(packed.nbytes),
        "scale_bytes": int(flat_scales.astype(np.float16).nbytes),
        "model_bytes": model_bytes,
        "weight_bits_per_value": (
            packed.nbytes * 8 + flat_scales.astype(np.float16).nbytes * 8
        )
        / value_count,
        "pad": pad,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--in-features", type=int, default=120)
    parser.add_argument("--out-features", type=int, default=60)
    parser.add_argument("--group-size", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    value_count = args.in_features * args.out_features
    scale_count = (value_count + args.group_size - 1) // args.group_size
    codes = rng.integers(-1, 2, size=value_count, dtype=np.int8)
    scales = rng.uniform(0.005, 0.08, size=scale_count).astype(np.float32)
    bias = rng.uniform(-0.02, 0.02, size=args.out_features).astype(np.float32)
    receipt = build_packed_linear_model(
        codes=codes,
        scales=scales,
        bias=bias,
        out_features=args.out_features,
        in_features=args.in_features,
        group_size=args.group_size,
        output=args.out,
    )
    input_values = rng.normal(size=(3, args.in_features)).astype(np.float32)
    artifact_scales = scales.astype(np.float16).astype(np.float32)
    expanded_scales = np.repeat(artifact_scales, args.group_size)[:value_count]
    weight = (codes.astype(np.float32) * expanded_scales).reshape(
        args.out_features, args.in_features
    )
    expected = input_values @ weight.T + bias
    fixture_path = args.out.with_suffix(".json")
    fixture_path.write_text(
        json.dumps(
            {
                "model": str(args.out),
                "input": input_values.reshape(-1).tolist(),
                "input_shape": list(input_values.shape),
                "expected": expected.reshape(-1).tolist(),
                "output_shape": list(expected.shape),
                "receipt": receipt,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print(json.dumps({**receipt, "fixture": str(fixture_path)}, indent=2))


if __name__ == "__main__":
    main()
