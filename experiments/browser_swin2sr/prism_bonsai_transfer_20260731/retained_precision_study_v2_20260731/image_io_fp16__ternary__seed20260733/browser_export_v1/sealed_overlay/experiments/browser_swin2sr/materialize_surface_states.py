#!/usr/bin/env python3
"""Materialize one canonical dense and browser-runtime Swin2SR hard state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from surface_artifact_state import (
    hard_surface_state_dict,
    runtime_surface_state_dict,
)


FORMAT = "backdropkit-swin2sr-surface-materialization-v1"
RETAINED_PRECISION_FORMAT = (
    "backdropkit-swin2sr-retained-precision-materialization-v1"
)
SUPPORTED_FORMATS = {FORMAT, RETAINED_PRECISION_FORMAT}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-device", required=True)
    return parser.parse_args()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_receipt(path: Path, state: dict[str, torch.Tensor]) -> dict:
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "tensor_sha256": state_sha256(state),
        "tensors": len(state),
        "values": sum(tensor.numel() for tensor in state.values()),
    }


def write_materialization(
    checkpoint: Path,
    out: Path,
    source_device: str,
) -> dict:
    source, surfaces, payload = load_checkpoint_student(
        checkpoint,
        source_device,
    )
    runtime_state = runtime_surface_state_dict(source, surfaces)
    dense_state = hard_surface_state_dict(source, surfaces)
    out.mkdir(parents=True, exist_ok=True)
    dense_path = out / "dense-hard.pt"
    runtime_path = out / "runtime-hard.pt"
    torch.save(dense_state, dense_path)
    torch.save(runtime_state, runtime_path)
    surface_contract = {
        f"{surface.name}.weight": {
            "category": surface.category,
            "method": surface.quantizer.method,
            "group_size": surface.quantizer.group_size,
            "rotation": surface.quantizer.rotation,
            "rotation_block": surface.quantizer.rotation_block,
        }
        for surface in surfaces
    }
    receipt = {
        "format": FORMAT,
        "interpretation": (
            "Continuous masters were projected exactly once on source_device. "
            "dense_state is the plain-model evaluation identity; runtime_state "
            "preserves rotated core weights for packed/ONNX browser export."
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_device": source_device,
        "surface_step": int(payload["surface_step"]),
        "all_target_surfaces_materialized_hard": True,
        "quantization": payload["quantization"],
        "surface_training": payload["surface_training"],
        "surfaces": surface_contract,
        "dense_state": state_receipt(dense_path, dense_state),
        "runtime_state": state_receipt(runtime_path, runtime_state),
    }
    manifest_path = out / "materialization.json"
    manifest_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {**receipt, "manifest": str(manifest_path.resolve())}


def load_materialized_state(
    manifest_path: Path,
    kind: str,
) -> tuple[dict, dict[str, torch.Tensor]]:
    if kind not in {"dense_state", "runtime_state"}:
        raise ValueError(f"unsupported materialized state kind: {kind}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") not in SUPPORTED_FORMATS:
        raise ValueError("unsupported materialization manifest")
    record = manifest[kind]
    path = Path(record["path"])
    if not path.exists() or file_sha256(path) != record["file_sha256"]:
        raise ValueError(f"{kind} file identity mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not all(
        torch.is_tensor(value) for value in state.values()
    ):
        raise ValueError(f"{kind} is not a tensor state dictionary")
    if state_sha256(state) != record["tensor_sha256"]:
        raise ValueError(f"{kind} tensor identity mismatch")
    if len(state) != int(record["tensors"]):
        raise ValueError(f"{kind} tensor count mismatch")
    if sum(value.numel() for value in state.values()) != int(record["values"]):
        raise ValueError(f"{kind} value count mismatch")
    return manifest, state


def main() -> None:
    args = parse_args()
    receipt = write_materialization(
        args.checkpoint,
        args.out,
        args.source_device,
    )
    print(json.dumps({
        "manifest": receipt["manifest"],
        "source_device": receipt["source_device"],
        "surface_step": receipt["surface_step"],
        "surfaces": len(receipt["surfaces"]),
        "dense_state": receipt["dense_state"]["path"],
        "runtime_state": receipt["runtime_state"]["path"],
    }, indent=2))


if __name__ == "__main__":
    main()
