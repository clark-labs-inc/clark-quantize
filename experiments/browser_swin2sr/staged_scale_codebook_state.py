"""Checkpoint and materialization contracts for chained scale collapse."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from full_surface_provenance import file_sha256
from materialize_surface_states import state_receipt, state_sha256
from scale_codebook_state import (
    MATERIALIZATION_FORMAT,
    load_scale_codebook_student,
)
from staged_scale_codebook import add_staged_collapse, freeze_hard_bridge
from surface_artifact_state import hard_surface_state_dict, runtime_surface_state_dict


STAGED_CHECKPOINT_FORMAT = "backdropkit-swin2sr-staged-scale-codebook-checkpoint-v1"


def load_staged_scale_codebook_student(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != STAGED_CHECKPOINT_FORMAT:
        raise ValueError("unsupported staged scale-codebook checkpoint")
    bridge = Path(payload["bridge_checkpoint"])
    if file_sha256(bridge) != payload["bridge_checkpoint_sha256"]:
        raise ValueError("staged scale-codebook bridge identity changed")
    (
        student,
        surfaces,
        bridge_codebooks,
        bridge_payload,
        parent_payload,
    ) = load_scale_codebook_student(bridge, device)
    freeze_hard_bridge(bridge_codebooks)
    collapse = add_staged_collapse(
        surfaces,
        bridge_codebooks,
        target_bits_by_category=payload["staged_scale_codebook"][
            "target_bits_by_category"
        ],
    )
    student.load_state_dict(payload["model"], strict=True)
    return (
        student,
        surfaces,
        bridge_codebooks,
        collapse,
        payload,
        bridge_payload,
        parent_payload,
    )


def materialize_staged_scale_codebook_checkpoint(
    checkpoint: Path,
    out: Path,
    source_device: str,
) -> dict:
    (
        student,
        surfaces,
        bridge_codebooks,
        collapse,
        payload,
        bridge_payload,
        _parent,
    ) = load_staged_scale_codebook_student(checkpoint, source_device)
    for codebook in bridge_codebooks:
        codebook.projection.amount.fill_(1.0)
    for codebook in collapse:
        codebook.projection.amount.fill_(1.0)
    runtime_state = runtime_surface_state_dict(student, surfaces)
    dense_state = hard_surface_state_dict(student, surfaces)
    out.mkdir(parents=True, exist_ok=True)
    runtime_path = out / "runtime-hard.pt"
    dense_path = out / "dense-hard.pt"
    torch.save(runtime_state, runtime_path)
    torch.save(dense_state, dense_path)
    final_scale_contract = {
        "format": "backdropkit-swin2sr-staged-final-scale-codebook-v1",
        "target_bits_by_category": payload["staged_scale_codebook"][
            "target_bits_by_category"
        ],
        "bridge_checkpoint": payload["bridge_checkpoint"],
        "bridge_checkpoint_sha256": payload["bridge_checkpoint_sha256"],
        "bridge_target_bits_by_category": bridge_payload["scale_codebook"][
            "target_bits_by_category"
        ],
    }
    receipt = {
        "format": MATERIALIZATION_FORMAT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_device": source_device,
        "scale_step": int(payload["collapse_step"]),
        "parent_checkpoint": bridge_payload["parent_checkpoint"],
        "parent_checkpoint_sha256": bridge_payload["parent_checkpoint_sha256"],
        "quantization": payload["quantization"],
        "surface_training": payload["surface_training"],
        "scale_codebook": final_scale_contract,
        "scale_codebook_lineage": {
            "bridge": bridge_payload["scale_codebook"],
            "collapse": payload["staged_scale_codebook"],
        },
        "surfaces": payload["surfaces"],
        "runtime_state": state_receipt(runtime_path, runtime_state),
        "dense_state": state_receipt(dense_path, dense_state),
        "all_scale_codebooks_materialized_hard": True,
    }
    manifest = out / "materialization.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    return {**receipt, "manifest": str(manifest.resolve())}


def staged_model_tensor_sha256(model: torch.nn.Module) -> str:
    return state_sha256({
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    })
