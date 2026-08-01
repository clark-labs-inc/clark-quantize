#!/usr/bin/env python3
"""Build a complete portable ONNX artifact from a hard-ternary Swin2SR state."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from transformers import Swin2SRForImageSuperResolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from materialize_surface_states import load_materialized_state
from mixed_precision_artifact import (
    PrecisionContract,
    validate_precision_contract,
)
from rewrite_onnx_ternary import rewrite_ternary_initializers
from surface_schedule import SCOPES, surface_targets


MODEL_ID = "caidas/swin2SR-lightweight-x2-64"
MODEL_REVISION = "07e8e61492ce6daf4f34c651622960faf110c17f"
BASE_ONNX_MODEL = "Xenova/swin2SR-lightweight-x2-64"
BASE_ONNX_REVISION = "54ccc3f9ed678912a9c47dc14f6c478855380a2b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_deterministic_gzip(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, output.open("wb") as output_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=output_file,
            mtime=0,
        ) as compressed:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                compressed.write(chunk)


def _array_key(values: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    contiguous = np.ascontiguousarray(values.astype(np.float32))
    return contiguous.shape, hashlib.sha256(contiguous.tobytes()).digest()


def map_state_to_onnx(
    teacher_state: dict[str, torch.Tensor],
    initializers: list[onnx.TensorProto],
) -> dict[str, tuple[str, str]]:
    index: dict[tuple[tuple[int, ...], bytes], list[str]] = {}
    for initializer in initializers:
        values = numpy_helper.to_array(initializer)
        if not np.issubdtype(values.dtype, np.floating):
            continue
        index.setdefault(_array_key(values), []).append(initializer.name)

    mapping = {}
    for state_name, tensor in teacher_state.items():
        values = tensor.detach().cpu().numpy().astype(np.float32)
        candidates = [
            (initializer_name, "direct")
            for initializer_name in index.get(_array_key(values), [])
        ]
        if values.ndim == 2:
            candidates.extend(
                (initializer_name, "transpose")
                for initializer_name in index.get(_array_key(values.T), [])
            )
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            raise ValueError(
                f"teacher tensor must map uniquely into base ONNX: "
                f"{state_name} -> {candidates}"
            )
        mapping[state_name] = candidates[0]
    initializer_names = {initializer.name for initializer in initializers}
    mapped_names = {initializer_name for initializer_name, _ in mapping.values()}
    if mapped_names != initializer_names:
        missing = sorted(initializer_names - mapped_names)
        raise ValueError(f"base ONNX has unmapped initializers: {missing}")
    return mapping


def replace_initializers(
    model: onnx.ModelProto,
    candidate_state: dict[str, torch.Tensor],
    mapping: dict[str, tuple[str, str]],
) -> None:
    replacements = {}
    for state_name, (initializer_name, layout) in mapping.items():
        if state_name not in candidate_state:
            raise ValueError(f"candidate state is missing tensor: {state_name}")
        values = candidate_state[state_name].detach().cpu().numpy().astype(np.float32)
        if layout == "transpose":
            values = values.T
        replacements[initializer_name] = numpy_helper.from_array(values, initializer_name)
    for initializer in model.graph.initializer:
        initializer.CopyFrom(replacements[initializer.name])

def inject_hadamard_inputs(
    model: onnx.ModelProto,
    tensor_names: set[str],
    block_size: int,
) -> int:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a positive power of two")
    hadamard = np.asarray([[1.0]], dtype=np.float32)
    while hadamard.shape[0] < block_size:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    hadamard /= np.sqrt(block_size)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(hadamard, "clark_hadamard_matrix"),
            numpy_helper.from_array(
                np.asarray([-1, block_size], dtype=np.int64),
                "clark_hadamard_flat_shape",
            ),
        ]
    )
    rewritten_nodes = []
    transformed = set()
    for index, node in enumerate(model.graph.node):
        if node.op_type != "MatMul" or len(node.input) < 2 or node.input[1] not in tensor_names:
            rewritten_nodes.append(node)
            continue
        weight_name = node.input[1]
        if weight_name in transformed:
            raise ValueError(f"selected weight has multiple MatMul consumers: {weight_name}")
        transformed.add(weight_name)
        source_input = node.input[0]
        prefix = f"clark_hadamard_{index}"
        shape = f"{prefix}_shape"
        blocks = f"{prefix}_blocks"
        transformed_blocks = f"{prefix}_transformed_blocks"
        transformed_input = f"{prefix}_input"
        rewritten_nodes.extend(
            [
                onnx.helper.make_node("Shape", [source_input], [shape]),
                onnx.helper.make_node(
                    "Reshape", [source_input, "clark_hadamard_flat_shape"], [blocks]
                ),
                onnx.helper.make_node(
                    "MatMul",
                    [blocks, "clark_hadamard_matrix"],
                    [transformed_blocks],
                ),
                onnx.helper.make_node(
                    "Reshape", [transformed_blocks, shape], [transformed_input]
                ),
            ]
        )
        node.input[0] = transformed_input
        rewritten_nodes.append(node)
    missing = tensor_names - transformed
    if missing:
        raise ValueError(f"selected weights have no MatMul consumer: {sorted(missing)}")
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    return len(transformed)


def load_runtime_state(
    teacher: Swin2SRForImageSuperResolution,
    materialization_path: Path,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, object],
    dict[str, int],
    dict,
]:
    materialization, state = load_materialized_state(
        materialization_path,
        "runtime_state",
    )
    teacher.load_state_dict(state, strict=True)
    quantization = materialization["quantization"]
    if quantization.get("mode") != "ternary":
        raise ValueError("portable materialization must be ternary")
    return (
        state,
        quantization,
        {
            name: int(contract["group_size"])
            for name, contract in materialization["surfaces"].items()
        },
        materialization,
    )


def portable_precision_contract(
    materialization: dict,
    state: dict[str, torch.Tensor],
    expected_state_names: set[str],
    selected_state_names: set[str],
) -> PrecisionContract:
    contract = validate_precision_contract(
        materialization,
        state,
        expected_state_names,
    )
    retained_state_names = set(contract.fp16_weights) | set(contract.int7_weights)
    if selected_state_names | retained_state_names != expected_state_names:
        raise ValueError(
            "materialized surface contract does not match selected ONNX scope"
        )
    if selected_state_names & retained_state_names:
        raise ValueError("portable ternary and retained-precision surfaces overlap")
    return contract


def build_artifact(
    *,
    materialization_path: Path,
    base_onnx_path: Path,
    output_path: Path,
    scope: str,
) -> dict[str, object]:
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    teacher_state = {
        name: tensor.detach().clone() for name, tensor in teacher.state_dict().items()
    }
    payload, quantization, runtime_group_sizes, materialization = load_runtime_state(
        teacher,
        materialization_path,
    )
    if scope != quantization.get("scope"):
        raise ValueError(
            f"requested scope {scope} != materialized scope "
            f"{quantization.get('scope')}"
        )
    model = onnx.load_model(base_onnx_path)
    mapping = map_state_to_onnx(teacher_state, list(model.graph.initializer))
    replace_initializers(model, payload, mapping)
    targets = surface_targets(
        teacher,
        scope,
        quantization.get("schedule_profile", "joint_io_v1"),
    )
    expected_state_names = {f"{target.name}.weight" for target in targets}
    selected_state_names = set(runtime_group_sizes)
    precision_contract = portable_precision_contract(
        materialization,
        payload,
        expected_state_names,
        selected_state_names,
    )
    rotated_state_names = {
        f"{target.name}.weight"
        for target in targets
        if target.category == "core_linear"
        and f"{target.name}.weight" in selected_state_names
    }
    selected_onnx_names = {
        mapping[state_name][0] for state_name in selected_state_names
    }
    selected_sources = {
        mapping[state_name][0]: payload[state_name].detach().cpu().numpy().astype(np.float32)
        for state_name in selected_state_names
    }
    selected_group_sizes = {
        mapping[state_name][0]: runtime_group_sizes[state_name]
        for state_name in selected_state_names
    }
    transpose_outputs = {
        mapping[state_name][0]
        for state_name in selected_state_names
        if mapping[state_name][1] == "transpose"
    }
    rotation = quantization.get("rotation", "none")
    rotation_block = int(quantization.get("rotation_block", 4))
    hadamard_nodes = 0
    if rotation == "hadamard":
        hadamard_nodes = inject_hadamard_inputs(
            model,
            {
                mapping[state_name][0]
                for state_name in rotated_state_names
            },
            rotation_block,
        )
    ledger = rewrite_ternary_initializers(
        model,
        selected_onnx_names,
        selected_group_sizes,
        source_values=selected_sources,
        transpose_outputs=transpose_outputs,
    )
    onnx.checker.check_model(model, full_check=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output_path)
    package_path = output_path.with_suffix(output_path.suffix + ".gz")
    write_deterministic_gzip(output_path, package_path)
    ledger.update(
        {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "base_onnx_model": BASE_ONNX_MODEL,
            "base_onnx_revision": BASE_ONNX_REVISION,
            "base_onnx_sha256": sha256_file(base_onnx_path),
            "materialization_manifest": str(materialization_path),
            "materialization_manifest_sha256": sha256_file(
                materialization_path
            ),
            "runtime_state_file_sha256": materialization[
                "runtime_state"
            ]["file_sha256"],
            "runtime_state_tensor_sha256": materialization[
                "runtime_state"
            ]["tensor_sha256"],
            "hard_state_sha256": materialization[
                "dense_state"
            ]["file_sha256"],
            "scope": scope,
            "method": quantization.get("method", "optimal_mse"),
            "input_transform": (
                {
                    "kind": "tensor_local_block_hadamard",
                    "block_size": rotation_block,
                    "tensors": len(rotated_state_names),
                }
                if rotation == "hadamard"
                else {"kind": "none"}
            ),
            "hadamard_matmul_inputs": hadamard_nodes,
            "retained_precision_islands": precision_contract.islands,
            "retained_precision_island": (
                precision_contract.islands[0]
                if len(precision_contract.islands) == 1
                else None
            ),
            "portable_dense_island_values": {
                "fp16": sum(
                    payload[name].numel()
                    for name in precision_contract.fp16_weights
                ),
                "int7": sum(
                    payload[name].numel()
                    for name in precision_contract.int7_weights
                ),
            },
            "onnx_checker": "passed",
            "model_bytes": output_path.stat().st_size,
            "model_sha256": sha256_file(output_path),
            "runtime_package": str(package_path),
            "runtime_package_compression": "gzip",
            "runtime_package_bytes": package_path.stat().st_size,
            "runtime_package_sha256": sha256_file(package_path),
        }
    )
    ledger_path = output_path.with_suffix(".ledger.json")
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    return {**ledger, "ledger": str(ledger_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("materialization", type=Path)
    parser.add_argument("--base-onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, default="core")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_artifact(
        materialization_path=args.materialization,
        base_onnx_path=args.base_onnx,
        output_path=args.out,
        scope=args.scope,
    )
    print(json.dumps({
        "ledger": receipt["ledger"],
        "model": receipt["runtime_package"],
        "scope": receipt["scope"],
        "tensor_count": receipt["tensor_count"],
        "values": receipt["values"],
        "weight_bits_per_value_including_scales": (
            receipt["weight_bits_per_value_including_scales"]
        ),
        "model_bytes": receipt["model_bytes"],
        "runtime_package_bytes": receipt["runtime_package_bytes"],
        "onnx_checker": receipt["onnx_checker"],
    }, indent=2))


if __name__ == "__main__":
    main()
