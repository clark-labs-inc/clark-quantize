#!/usr/bin/env python3
"""Measure ternary movement signatures without turning them into objectives."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clark_air.quant.optimal_ternary import quantize_ternary_mse
from full_surface_provenance import file_sha256
from quantizer_controls import quantize_ternary_absmean
from surface_model_state import pretrained
from surface_quantization import (
    add_surface_parametrizations,
    apply_surface_group_sizes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def codes(weight: torch.Tensor, quantizer) -> torch.Tensor:
    if quantizer.method == "absmean":
        packed = quantize_ternary_absmean(weight, quantizer.group_size)
    elif quantizer.method == "optimal_mse":
        packed = quantize_ternary_mse(weight, quantizer.group_size)
    else:
        raise ValueError(f"unsupported signature quantizer: {quantizer.method}")
    return packed.codes[: weight.numel()].reshape(weight.shape)


def signature_counts(
    base: torch.Tensor,
    master: torch.Tensor,
    base_codes: torch.Tensor,
    current_codes: torch.Tensor,
) -> dict[str, float | int]:
    nonzero = current_codes != 0
    base_sign = torch.sign(base)
    current_sign = torch.sign(current_codes)
    inherited = nonzero & (current_sign == base_sign)
    direct_flips = (
        (base_codes != 0)
        & nonzero
        & (torch.sign(base_codes) != current_sign)
    )
    delta = (master.float() - base.float()).square().sum().item()
    base_energy = base.float().square().sum().item()
    return {
        "values": current_codes.numel(),
        "changed_codes": (current_codes != base_codes).sum().item(),
        "zero_codes": (current_codes == 0).sum().item(),
        "current_nonzero": nonzero.sum().item(),
        "sign_inherited": inherited.sum().item(),
        "direct_sign_flips": direct_flips.sum().item(),
        "master_delta_sq": delta,
        "base_energy": base_energy,
    }


def summarize(counts: dict[str, float | int]) -> dict[str, float | int]:
    values = int(counts["values"])
    nonzero = int(counts["current_nonzero"])
    return {
        **counts,
        "code_churn": int(counts["changed_codes"]) / values,
        "zero_fraction": int(counts["zero_codes"]) / values,
        "sign_inheritance": (
            int(counts["sign_inherited"]) / nonzero
            if nonzero
            else 1.0
        ),
        "direct_sign_flip_fraction": (
            int(counts["direct_sign_flips"]) / values
        ),
        "relative_master_movement": (
            (float(counts["master_delta_sq"]) / counts["base_energy"]) ** 0.5
            if counts["base_energy"]
            else 0.0
        ),
    }


def merge(target: dict, source: dict) -> None:
    for name, value in source.items():
        target[name] += value


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    quantization = payload["quantization"]
    model = pretrained("cpu")
    surfaces = add_surface_parametrizations(
        model,
        scope="all_matrix_and_conv",
        group_size=int(quantization["group_size"]),
        method=quantization["method"],
        core_rotation=quantization.get("rotation", "none"),
        rotation_block=int(quantization.get("rotation_block", 4)),
        conv_method=quantization.get("conv_method"),
        conv_grouping=quantization.get("conv_grouping", "flat"),
        schedule_profile=quantization.get("schedule_profile", "joint_io_v1"),
    )
    apply_surface_group_sizes(
        surfaces,
        quantization.get("surface_group_sizes"),
    )
    base_masters = {
        surface.name: (
            surface.module.parametrizations.weight.original.detach().clone()
        )
        for surface in surfaces
    }
    model.load_state_dict(payload["model"], strict=True)
    overall = defaultdict(float)
    categories = defaultdict(lambda: defaultdict(float))
    per_surface = []
    with torch.inference_mode():
        for surface in surfaces:
            base = base_masters[surface.name]
            master = surface.module.parametrizations.weight.original.detach()
            initial_codes = codes(base, surface.quantizer)
            current_codes = codes(master, surface.quantizer)
            counts = signature_counts(
                base,
                master,
                initial_codes,
                current_codes,
            )
            merge(overall, counts)
            merge(categories[surface.category], counts)
            per_surface.append({
                "name": surface.name,
                "category": surface.category,
                "method": surface.quantizer.method,
                "group_size": surface.quantizer.group_size,
                **summarize(counts),
            })
    receipt = {
        "format": "backdropkit-swin2sr-ternary-signatures-v1",
        "diagnostic_only": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "surface_step": payload["surface_step"],
        "overall": summarize(overall),
        "categories": {
            category: summarize(counts)
            for category, counts in sorted(categories.items())
        },
        "surfaces": per_surface,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "surface_step": receipt["surface_step"],
        "overall": receipt["overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
