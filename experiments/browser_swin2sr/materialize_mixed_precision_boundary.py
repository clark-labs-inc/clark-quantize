#!/usr/bin/env python3
"""Merge a precision-boundary state with its canonical browser runtime state."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch

from full_surface_provenance import file_sha256
from materialize_surface_states import (
    RETAINED_PRECISION_FORMAT,
    load_materialized_state,
    state_receipt,
    state_sha256,
)


BOUNDARY_FORMAT = (
    "backdropkit-swin2sr-rstb-precision-boundary-materialization-v1"
)
FORMAT = "backdropkit-swin2sr-mixed-precision-runtime-materialization-v1"
SUPPORTED_ARMS = {
    "int7_per_output": {
        "precision": "int7",
        "storage_dtype": "packed_int7",
        "bit_width": 7,
        "granularity": "per_output",
    }
}


def legacy_boundary_tensor_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_boundary(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    receipt = json.loads(path.read_text())
    if receipt.get("format") != BOUNDARY_FORMAT:
        raise ValueError("unsupported precision-boundary materialization")
    record = receipt["dense_state"]
    state_path = Path(record["path"])
    if file_sha256(state_path) != record["file_sha256"]:
        raise ValueError("precision-boundary dense-state identity mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    accepted_hashes = {
        state_sha256(state),
        legacy_boundary_tensor_sha256(state),
    }
    if record["tensor_sha256"] not in accepted_hashes:
        raise ValueError("precision-boundary tensor identity mismatch")
    return receipt, state


def materialize(boundary_path: Path, base_path: Path, out: Path) -> dict:
    boundary, dense_state = load_boundary(boundary_path)
    base, base_dense = load_materialized_state(base_path, "dense_state")
    _, base_runtime = load_materialized_state(base_path, "runtime_state")
    if base.get("format") != RETAINED_PRECISION_FORMAT:
        raise ValueError("mixed runtime base must be a retained-precision state")
    if dense_state.keys() != base_dense.keys() or dense_state.keys() != base_runtime.keys():
        raise RuntimeError("mixed-precision state dictionaries differ")
    changed = [
        name for name in dense_state
        if not torch.equal(dense_state[name], base_dense[name])
    ]
    if set(changed) != set(boundary["changed_tensors"]):
        raise RuntimeError("mixed-precision state changed undeclared tensors")
    arm = boundary["arm"]
    if arm not in SUPPORTED_ARMS:
        raise ValueError(f"unsupported mixed-precision boundary arm: {arm}")
    surfaces = deepcopy(base["surfaces"])
    for name in changed:
        contract = surfaces.pop(name, None)
        if contract is None:
            raise RuntimeError(f"boundary tensor is not a quantized surface: {name}")
        if contract["category"] != "residual_body_conv":
            raise RuntimeError(f"boundary tensor is not an RSTB convolution: {name}")
        if contract["rotation"] != "none":
            raise RuntimeError(f"boundary RSTB tensor has a runtime transform: {name}")
    runtime_state = {
        name: value.detach().cpu().clone()
        for name, value in base_runtime.items()
    }
    for name in changed:
        runtime_state[name] = dense_state[name].detach().cpu().clone()
    precision_islands = [
        deepcopy(island)
        for island in base.get("retained_precision_islands", [])
        if island["name"] != "rstb_fusion_fp16"
    ]
    precision_islands.append({
        "name": "rstb_fusion_int7",
        **SUPPORTED_ARMS[arm],
        "weight_values": sum(dense_state[name].numel() for name in changed),
        "weights": changed,
        "data_bytes": int(boundary["storage"]["data_bytes"]),
        "incremental_data_bytes": int(
            boundary["storage"]["incremental_data_bytes"]
        ),
        "source_materialization": str(boundary_path.resolve()),
        "source_materialization_sha256": file_sha256(boundary_path),
    })
    out.mkdir(parents=True, exist_ok=True)
    dense_path = out / "dense-hard.pt"
    runtime_path = out / "runtime-hard.pt"
    torch.save(dense_state, dense_path)
    torch.save(runtime_state, runtime_path)
    receipt = {
        "format": RETAINED_PRECISION_FORMAT,
        "mixed_precision_format": FORMAT,
        "interpretation": (
            "The canonical ternary browser runtime state is preserved except "
            "for the declared RSTB int7-per-output island. Image-I/O weights "
            "remain exact FP16 roundtrips."
        ),
        "base_materialization": str(base_path.resolve()),
        "base_materialization_sha256": file_sha256(base_path),
        "boundary_materialization": str(boundary_path.resolve()),
        "boundary_materialization_sha256": file_sha256(boundary_path),
        "all_target_surfaces_materialized_hard": True,
        "quantization": base["quantization"],
        "surface_training": base["surface_training"],
        "surfaces": surfaces,
        "retained_precision_islands": precision_islands,
        "dense_state": state_receipt(dense_path, dense_state),
        "runtime_state": state_receipt(runtime_path, runtime_state),
    }
    manifest = out / "materialization.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    return {**receipt, "manifest": str(manifest.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("boundary", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(args.boundary, args.base, args.out)
    print(json.dumps({
        "manifest": receipt["manifest"],
        "mixed_precision_format": receipt["mixed_precision_format"],
        "surfaces": len(receipt["surfaces"]),
        "retained_precision_islands": receipt["retained_precision_islands"],
        "dense_state": receipt["dense_state"],
        "runtime_state": receipt["runtime_state"],
    }, indent=2))


if __name__ == "__main__":
    main()
