#!/usr/bin/env python3
"""Rewrite hard-ternary ONNX initializers into portable base-3 subgraphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_packed_bridge import pack_trits_5, trit_lookup


def _int64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def recover_ternary_groups(
    values: np.ndarray,
    group_size: int,
    *,
    relative_tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    codes = np.zeros(flat.size, dtype=np.int8)
    scales = np.zeros((flat.size + group_size - 1) // group_size, dtype=np.float32)
    max_error = 0.0
    for group_index, start in enumerate(range(0, flat.size, group_size)):
        group = flat[start : start + group_size]
        scale = float(np.max(np.abs(group)))
        scales[group_index] = scale
        if scale == 0.0:
            continue
        recovered = np.rint(group / scale).astype(np.int8)
        if not np.isin(recovered, (-1, 0, 1)).all():
            raise ValueError(f"group {group_index} is not ternary")
        reconstructed = recovered.astype(np.float32) * scale
        error = float(np.max(np.abs(group - reconstructed)))
        max_error = max(max_error, error)
        if error > max(relative_tolerance * scale, np.finfo(np.float32).eps):
            raise ValueError(
                f"group {group_index} is not hard ternary: error {error} at scale {scale}"
            )
        codes[start : start + group.size] = recovered
    return codes, scales, max_error


def _decode_nodes(
    *,
    prefix: str,
    output_name: str,
    shape: list[int],
    packed_name: str,
    scales_name: str,
    value_count: int,
    scale_count: int,
    group_size: int,
    opset_version: int,
    transpose_output: bool,
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    names = {
        key: f"{prefix}_{key}"
        for key in (
            "indices",
            "decoded_groups",
            "decoded_padded",
            "decoded_codes",
            "scales_f32",
            "scales_2d",
            "scales_expanded_2d",
            "scales_padded",
            "scales_expanded",
            "weight_flat",
            "weight_source",
            "flat_shape",
            "weight_shape",
            "unsqueeze_axes",
            "scale_shape",
            "slice_starts",
            "slice_ends",
            "slice_axes",
            "slice_steps",
        )
    }
    initializers = [
        _int64(names["flat_shape"], [-1]),
        _int64(names["weight_shape"], shape),
        _int64(names["scale_shape"], [scale_count, group_size]),
        _int64(names["slice_starts"], [0]),
        _int64(names["slice_ends"], [value_count]),
        _int64(names["slice_axes"], [0]),
        _int64(names["slice_steps"], [1]),
    ]
    if opset_version >= 13:
        initializers.append(_int64(names["unsqueeze_axes"], [1]))
        unsqueeze = helper.make_node(
            "Unsqueeze",
            [names["scales_f32"], names["unsqueeze_axes"]],
            [names["scales_2d"]],
        )
    else:
        unsqueeze = helper.make_node(
            "Unsqueeze",
            [names["scales_f32"]],
            [names["scales_2d"]],
            axes=[1],
        )
    final_output = names["weight_source"] if transpose_output else output_name
    nodes = [
        helper.make_node("Cast", [packed_name], [names["indices"]], to=TensorProto.INT32),
        helper.make_node(
            "Gather",
            ["clark_trit_lookup", names["indices"]],
            [names["decoded_groups"]],
        ),
        helper.make_node(
            "Reshape",
            [names["decoded_groups"], names["flat_shape"]],
            [names["decoded_padded"]],
        ),
        helper.make_node(
            "Slice",
            [
                names["decoded_padded"],
                names["slice_starts"],
                names["slice_ends"],
                names["slice_axes"],
                names["slice_steps"],
            ],
            [names["decoded_codes"]],
        ),
        helper.make_node(
            "Cast", [scales_name], [names["scales_f32"]], to=TensorProto.FLOAT
        ),
        unsqueeze,
        helper.make_node(
            "Expand",
            [names["scales_2d"], names["scale_shape"]],
            [names["scales_expanded_2d"]],
        ),
        helper.make_node(
            "Reshape",
            [names["scales_expanded_2d"], names["flat_shape"]],
            [names["scales_padded"]],
        ),
        helper.make_node(
            "Slice",
            [
                names["scales_padded"],
                names["slice_starts"],
                names["slice_ends"],
                names["slice_axes"],
                names["slice_steps"],
            ],
            [names["scales_expanded"]],
        ),
        helper.make_node(
            "Mul",
            [names["decoded_codes"], names["scales_expanded"]],
            [names["weight_flat"]],
        ),
        helper.make_node(
            "Reshape", [names["weight_flat"], names["weight_shape"]], [final_output]
        ),
    ]
    if transpose_output:
        nodes.append(
            helper.make_node(
                "Transpose", [names["weight_source"]], [output_name], perm=[1, 0]
            )
        )
    return nodes, initializers


def rewrite_ternary_initializers(
    model: onnx.ModelProto,
    tensor_names: set[str],
    tensor_group_sizes: dict[str, int],
    *,
    source_values: dict[str, np.ndarray] | None = None,
    transpose_outputs: set[str] | None = None,
) -> dict[str, object]:
    if "clark_trit_lookup" in {item.name for item in model.graph.initializer}:
        raise ValueError("model already contains Clark ternary lookup")
    available = {item.name: item for item in model.graph.initializer}
    missing = tensor_names - available.keys()
    if missing:
        raise ValueError(f"selected initializers are missing: {sorted(missing)}")
    if set(tensor_group_sizes) != tensor_names:
        raise ValueError("tensor group sizes must exactly match selected initializers")
    if any(group_size <= 0 for group_size in tensor_group_sizes.values()):
        raise ValueError("tensor group sizes must be positive")
    selected = [item for item in model.graph.initializer if item.name in tensor_names]
    if not selected:
        raise ValueError("no ternary initializers selected")
    default_opsets = [
        item.version for item in model.opset_import if item.domain in ("", "ai.onnx")
    ]
    if len(default_opsets) != 1 or default_opsets[0] < 10:
        raise ValueError("model must have exactly one default ONNX opset of at least 10")
    opset_version = default_opsets[0]
    source_values = source_values or {}
    transpose_outputs = transpose_outputs or set()
    if source_values.keys() - tensor_names:
        raise ValueError("source_values contains an unselected initializer")
    if transpose_outputs - tensor_names:
        raise ValueError("transpose_outputs contains an unselected initializer")

    replacement_nodes: list[onnx.NodeProto] = []
    replacement_initializers = [
        numpy_helper.from_array(trit_lookup(), "clark_trit_lookup")
    ]
    records = []
    for index, initializer in enumerate(selected):
        group_size = tensor_group_sizes[initializer.name]
        stored_values = numpy_helper.to_array(initializer)
        values = np.asarray(source_values.get(initializer.name, stored_values))
        if values.dtype not in (np.float32, np.float64):
            raise ValueError(f"selected initializer must be float: {initializer.name}")
        transpose_output = initializer.name in transpose_outputs
        expected_shape = values.T.shape if transpose_output else values.shape
        if expected_shape != stored_values.shape:
            raise ValueError(
                f"source layout does not reconstruct initializer: {initializer.name}"
            )
        codes, scales, recovery_error = recover_ternary_groups(values, group_size)
        packed, pad = pack_trits_5(codes)
        prefix = f"clark_ternary_{index}"
        packed_name = f"{prefix}_packed"
        scales_name = f"{prefix}_scales_f16"
        replacement_initializers.extend(
            [
                numpy_helper.from_array(packed, packed_name),
                numpy_helper.from_array(scales.astype(np.float16), scales_name),
            ]
        )
        nodes, constants = _decode_nodes(
            prefix=prefix,
            output_name=initializer.name,
            shape=list(values.shape),
            packed_name=packed_name,
            scales_name=scales_name,
            value_count=values.size,
            scale_count=scales.size,
            group_size=group_size,
            opset_version=opset_version,
            transpose_output=transpose_output,
        )
        replacement_nodes.extend(nodes)
        replacement_initializers.extend(constants)
        records.append(
            {
                "name": initializer.name,
                "shape": list(values.shape),
                "output_shape": list(stored_values.shape),
                "output_transform": "transpose" if transpose_output else "none",
                "values": int(values.size),
                "packed_code_bytes": int(packed.nbytes),
                "scale_bytes": int(scales.astype(np.float16).nbytes),
                "pad": pad,
                "hard_ternary_recovery_max_error": recovery_error,
            }
        )

    kept = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in tensor_names
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept + replacement_initializers)
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(replacement_nodes + original_nodes)
    onnx.checker.check_model(model)
    code_bytes = sum(record["packed_code_bytes"] for record in records)
    scale_bytes = sum(record["scale_bytes"] for record in records)
    values = sum(record["values"] for record in records)
    return {
        "format": "clark-portable-onnx-ternary-v2",
        "tensor_group_sizes": {
            name: tensor_group_sizes[name]
            for name in sorted(tensor_group_sizes)
        },
        "opset_version": opset_version,
        "tensors": records,
        "tensor_count": len(records),
        "values": values,
        "packed_code_bytes": code_bytes,
        "scale_bytes": scale_bytes,
        "weight_bits_per_value_including_scales": (code_bytes + scale_bytes)
        * 8
        / values,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tensor-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor_names = set(json.loads(args.tensor_list.read_text()))
    model = onnx.load_model(args.model)
    ledger = rewrite_ternary_initializers(
        model,
        tensor_names,
        {name: args.group_size for name in tensor_names},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.out)
    ledger["model_bytes"] = args.out.stat().st_size
    ledger_path = args.out.with_suffix(".ledger.json")
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({**ledger, "ledger": str(ledger_path)}, indent=2))


if __name__ == "__main__":
    main()
