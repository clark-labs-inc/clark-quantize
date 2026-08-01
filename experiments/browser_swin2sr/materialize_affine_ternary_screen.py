#!/usr/bin/env python3
"""Materialize the preregistered group-60 linear affine-ternary screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from affine_ternary import fit_affine_ternary_codes
from analyze_affine_ternary_screen import current_projection, stored_affine_values
from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from materialize_surface_states import state_sha256
from rotation_control import block_hadamard
from surface_artifact_state import hard_surface_state_dict


FORMAT = "backdropkit-swin2sr-affine-ternary-screen-materialization-v1"
ELIGIBLE_CATEGORIES = frozenset(("core_linear", "position_bias_linear"))
ELIGIBLE_GROUP_SIZE = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def is_eligible(surface) -> bool:
    return (
        surface.category in ELIGIBLE_CATEGORIES
        and int(surface.quantizer.group_size) == ELIGIBLE_GROUP_SIZE
    )


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


@torch.no_grad()
def materialize(checkpoint: Path, out: Path) -> dict:
    student, surfaces, payload = load_checkpoint_student(checkpoint, "cpu")
    base_state = hard_surface_state_dict(student, surfaces)
    state = {name: value.clone() for name, value in base_state.items()}
    code_digest = hashlib.sha256()
    changed = []
    shift_bytes = 0
    for surface in surfaces:
        if not is_eligible(surface):
            continue
        master = surface.module.parametrizations.weight.original.detach().cpu().float()
        current = current_projection(surface, master)
        affine = fit_affine_ternary_codes(
            master,
            current.codes,
            current.group_size,
            current.pad,
        )
        if not torch.equal(affine.codes, current.codes):
            raise RuntimeError(f"ternary code map changed: {surface.name}")
        code_digest.update(surface.name.encode())
        code_digest.update(b"\0")
        code_digest.update(tensor_bytes(current.codes))
        affine_rotated = stored_affine_values(affine)
        effective = (
            block_hadamard(affine_rotated, surface.quantizer.rotation_block)
            if surface.quantizer.rotation == "hadamard"
            else affine_rotated
        )
        state_name = f"{surface.name}.weight"
        base_weight = base_state[state_name]
        if effective.shape != base_weight.shape:
            raise RuntimeError(f"shape changed: {surface.name}")
        state[state_name] = effective.cpu()
        groups = int(current.scales.numel())
        shift_bytes += groups * 2
        changed.append({
            "surface": surface.name,
            "category": surface.category,
            "values": master.numel(),
            "groups": groups,
            "group_size": current.group_size,
            "rotation": surface.quantizer.rotation,
            "max_abs_delta_from_hard": float(
                (effective.cpu() - base_weight).abs().max().item()
            ),
            "mean_abs_delta_from_hard": float(
                (effective.cpu() - base_weight).abs().mean().item()
            ),
        })

    selected_names = {f"{item['surface']}.weight" for item in changed}
    for name, value in state.items():
        if name not in selected_names and not torch.equal(value, base_state[name]):
            raise RuntimeError(f"non-selected tensor changed: {name}")
    if len(changed) != 189:
        raise RuntimeError(f"expected 189 selected surfaces, got {len(changed)}")
    if sum(item["values"] for item in changed) != 783856:
        raise RuntimeError("selected value count changed")
    if shift_bytes != 26208:
        raise RuntimeError(f"selected shift bytes changed: {shift_bytes}")

    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "affine-group60-linears.pt"
    torch.save(state, state_path)
    receipt = {
        "format": FORMAT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "surface_step": int(payload["surface_step"]),
        "intervention": (
            "Fixed existing ternary codes; added FP16-roundtripped affine shifts "
            "only to group-60 core and position-bias linear surfaces."
        ),
        "data_access": {
            "training": False,
            "development": False,
            "final_heldout": False,
        },
        "selected_surfaces": len(changed),
        "selected_values": sum(item["values"] for item in changed),
        "selected_groups": sum(item["groups"] for item in changed),
        "incremental_shift_bytes": shift_bytes,
        "ternary_codes_sha256": code_digest.hexdigest(),
        "changed": changed,
        "state": {
            "path": str(state_path.resolve()),
            "file_sha256": file_sha256(state_path),
            "tensor_sha256": state_sha256(state),
            "tensors": len(state),
            "values": sum(value.numel() for value in state.values()),
        },
    }
    receipt_path = out / "materialization.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    args = parse_args()
    receipt = materialize(args.checkpoint, args.out)
    print(json.dumps({
        "state": receipt["state"],
        "selected_surfaces": receipt["selected_surfaces"],
        "selected_values": receipt["selected_values"],
        "incremental_shift_bytes": receipt["incremental_shift_bytes"],
        "ternary_codes_sha256": receipt["ternary_codes_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
