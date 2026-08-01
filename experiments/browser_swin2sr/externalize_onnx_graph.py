#!/usr/bin/env python3
"""Build a weightless ONNX graph and an exact state-to-external-data layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from transformers import Swin2SRForImageSuperResolution

from build_portable_onnx_artifact import (
    MODEL_ID,
    MODEL_REVISION,
    map_state_to_onnx,
    sha256_file,
)


FORMAT = "backdropkit-swin2sr-onnx-external-layout-v1"


def external_fields(initializer: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def build_external_graph(
    base_onnx: Path,
    graph_out: Path,
    external_out: Path,
    layout_out: Path,
) -> dict:
    model = onnx.load_model(base_onnx)
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    teacher_state = {
        name: value.detach().cpu() for name, value in teacher.state_dict().items()
    }
    mapping = map_state_to_onnx(teacher_state, list(model.graph.initializer))
    onnx.checker.check_model(model, full_check=True)
    graph_out.parent.mkdir(parents=True, exist_ok=True)
    external_out.parent.mkdir(parents=True, exist_ok=True)
    if graph_out.exists() or external_out.exists() or layout_out.exists():
        raise FileExistsError("external ONNX graph outputs must not already exist")
    onnx.save_model(
        model,
        graph_out,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_out.name,
        size_threshold=0,
        convert_attribute=False,
    )
    externalized = onnx.load_model(graph_out, load_external_data=False)
    by_initializer = {tensor.name: tensor for tensor in externalized.graph.initializer}
    entries = []
    covered = []
    for state_name, (initializer_name, tensor_layout) in sorted(mapping.items()):
        initializer = by_initializer[initializer_name]
        fields = external_fields(initializer)
        if fields.get("location") != external_out.name:
            raise RuntimeError(f"initializer was not externalized: {initializer_name}")
        offset = int(fields.get("offset", -1))
        length = int(fields.get("length", -1))
        if offset < 0 or length <= 0:
            raise RuntimeError(f"initializer external range is invalid: {initializer_name}")
        entries.append({
            "state_name": state_name,
            "initializer_name": initializer_name,
            "layout": tensor_layout,
            "state_shape": list(teacher_state[state_name].shape),
            "initializer_shape": list(initializer.dims),
            "data_type": int(initializer.data_type),
            "offset": offset,
            "length": length,
        })
        covered.append((offset, offset + length, initializer_name))
    covered.sort()
    if covered[0][0] != 0 or covered[-1][1] != external_out.stat().st_size:
        raise RuntimeError("external tensor ranges do not cover the full data file")
    for left, right in zip(covered, covered[1:]):
        if left[1] != right[0]:
            raise RuntimeError("external tensor ranges overlap or have a gap")
    layout = {
        "format": FORMAT,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_onnx": str(base_onnx.resolve()),
        "base_onnx_sha256": sha256_file(base_onnx),
        "graph": str(graph_out.resolve()),
        "graph_bytes": graph_out.stat().st_size,
        "graph_sha256": sha256_file(graph_out),
        "external_data_name": external_out.name,
        "external_data_bytes": external_out.stat().st_size,
        "external_data_sha256": sha256_file(external_out),
        "initializers": len(entries),
        "entries": entries,
    }
    layout_out.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n")
    return layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_onnx", type=Path)
    parser.add_argument("--graph-out", type=Path, required=True)
    parser.add_argument("--external-out", type=Path, required=True)
    parser.add_argument("--layout-out", type=Path, required=True)
    args = parser.parse_args()
    layout = build_external_graph(
        args.base_onnx,
        args.graph_out,
        args.external_out,
        args.layout_out,
    )
    print(json.dumps({
        "graph": layout["graph"],
        "graph_bytes": layout["graph_bytes"],
        "graph_sha256": layout["graph_sha256"],
        "external_data_bytes": layout["external_data_bytes"],
        "initializers": layout["initializers"],
    }, indent=2))


if __name__ == "__main__":
    main()
