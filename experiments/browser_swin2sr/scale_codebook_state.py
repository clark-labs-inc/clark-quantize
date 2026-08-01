"""Checkpoint and materialization contracts for scale-codebook recovery."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from full_surface_provenance import file_sha256
from materialize_surface_states import state_receipt, state_sha256
from scale_codebook_qat import add_per_surface_scale_codebooks, add_scale_codebooks
from scale_codebook_parent import load_scale_codebook_parent
from surface_artifact_state import hard_surface_state_dict, runtime_surface_state_dict


CHECKPOINT_FORMAT = "backdropkit-swin2sr-scale-codebook-checkpoint-v1"
MATERIALIZATION_FORMAT = "backdropkit-swin2sr-scale-codebook-materialization-v1"


def load_scale_codebook_student(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported scale-codebook checkpoint")
    parent = Path(payload["parent_checkpoint"])
    if file_sha256(parent) != payload["parent_checkpoint_sha256"]:
        raise ValueError("scale-codebook parent checkpoint identity changed")
    student, surfaces, parent_payload = load_scale_codebook_parent(parent, device)
    for surface in surfaces:
        surface.quantizer.amount.fill_(1.0)
    config = payload["scale_codebook"]
    if config.get("allocation_scope", "category") == "per_surface":
        codebooks = add_per_surface_scale_codebooks(
            surfaces,
            target_bits_by_surface=config["target_bits_by_surface"],
        )
    else:
        codebooks = add_scale_codebooks(
            surfaces,
            target_bits_by_category=config["target_bits_by_category"],
        )
    student.load_state_dict(payload["model"], strict=True)
    return student, surfaces, codebooks, payload, parent_payload


def materialize_scale_codebook_checkpoint(
    checkpoint: Path,
    out: Path,
    source_device: str,
) -> dict:
    student, surfaces, codebooks, payload, _parent = load_scale_codebook_student(
        checkpoint, source_device
    )
    for codebook in codebooks:
        codebook.projection.amount.fill_(1.0)
    runtime_state = runtime_surface_state_dict(student, surfaces)
    dense_state = hard_surface_state_dict(student, surfaces)
    out.mkdir(parents=True, exist_ok=True)
    runtime_path = out / "runtime-hard.pt"
    dense_path = out / "dense-hard.pt"
    torch.save(runtime_state, runtime_path)
    torch.save(dense_state, dense_path)
    receipt = {
        "format": MATERIALIZATION_FORMAT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_device": source_device,
        "scale_step": int(payload["scale_step"]),
        "parent_checkpoint": payload["parent_checkpoint"],
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "quantization": payload["quantization"],
        "surface_training": payload["surface_training"],
        "scale_codebook": payload["scale_codebook"],
        "surfaces": payload["surfaces"],
        "runtime_state": state_receipt(runtime_path, runtime_state),
        "dense_state": state_receipt(dense_path, dense_state),
        "all_scale_codebooks_materialized_hard": True,
    }
    manifest = out / "materialization.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    return {**receipt, "manifest": str(manifest.resolve())}


def load_scale_codebook_materialized_state(
    manifest_path: Path,
    kind: str,
) -> tuple[dict, dict[str, torch.Tensor]]:
    if kind not in {"dense_state", "runtime_state"}:
        raise ValueError(f"unsupported scale-codebook state kind: {kind}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != MATERIALIZATION_FORMAT:
        raise ValueError("unsupported scale-codebook materialization manifest")
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


def model_tensor_sha256(model: torch.nn.Module) -> str:
    return state_sha256({
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    })
