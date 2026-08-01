#!/usr/bin/env python3
"""Build a dense evaluation state from one or more packed ternary branches.

This is a capacity ablation, not an artifact exporter. Each branch is a true
groupwise ternary matrix, but multiple branches consume multiple 1.58-bit code
streams. The receipt reports both per-branch entropy and deployed bits per
parameter of the original model so the experiment cannot be mistaken for a
single-stream result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from transformers import Swin2SRForImageSuperResolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clark_air.quant.optimal_ternary import TernaryQuantized, quantize_ternary_mse
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terms", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def is_core_projection(name: str, module: torch.nn.Module) -> bool:
    return (
        isinstance(module, torch.nn.Linear)
        and "continuous_position_bias_mlp" not in name
        and (
            ".attention." in name
            or ".intermediate." in name
            or ".output.dense" in name
        )
    )


def residual_ternary_decomposition(
    weight: torch.Tensor,
    terms: int,
    group_size: int,
) -> tuple[list[TernaryQuantized], torch.Tensor]:
    if terms < 1:
        raise ValueError("terms must be positive")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    target = weight.detach().float()
    approximation = torch.zeros_like(target)
    branches = []
    for _ in range(terms):
        branch = quantize_ternary_mse(
            target - approximation,
            group_size=group_size,
        )
        branches.append(branch)
        approximation.add_(branch.dequantize())
    return branches, approximation.to(dtype=weight.dtype)


def build_state(
    model: torch.nn.Module,
    terms: int,
    group_size: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    selected_values = 0
    packed_code_bytes = 0
    scale_bytes = 0
    squared_error = 0.0
    module_records = []
    for name, module in model.named_modules():
        if not is_core_projection(name, module):
            continue
        branches, approximation = residual_ternary_decomposition(
            module.weight,
            terms,
            group_size,
        )
        values = module.weight.numel()
        selected_values += values
        state[f"{name}.weight"] = approximation.cpu()
        module_error = (
            module.weight.detach().float() - approximation.detach().float()
        ).square().sum().item()
        squared_error += module_error
        module_code_bytes = sum(
            branch.codes.numel() // 5 + int(branch.codes.numel() % 5 != 0)
            for branch in branches
        )
        module_scale_bytes = sum(branch.scales.numel() * 2 for branch in branches)
        packed_code_bytes += module_code_bytes
        scale_bytes += module_scale_bytes
        module_records.append({
            "name": name,
            "shape": list(module.weight.shape),
            "values": values,
            "terms": terms,
            "mse": module_error / values,
            "packed_code_bytes": module_code_bytes,
            "fp16_scale_bytes": module_scale_bytes,
        })
    if not module_records:
        raise ValueError("model has no selected Swin2SR core projections")
    receipt = {
        "format": "backdropkit-swin2sr-residual-ternary-ablation-v1",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "terms": terms,
        "group_size": group_size,
        "selected_modules": len(module_records),
        "selected_values": selected_values,
        "ideal_entropy_bits_per_branch_weight": math.log2(3),
        "packed_code_bytes": packed_code_bytes,
        "fp16_scale_bytes": scale_bytes,
        "codes_plus_scales_bytes": packed_code_bytes + scale_bytes,
        "deployed_bits_per_original_selected_weight": (
            (packed_code_bytes + scale_bytes) * 8 / selected_values
        ),
        "weight_mse": squared_error / selected_values,
        "modules": module_records,
        "scope": (
            "capacity ablation; multiple branches are not a single-stream "
            "1.58-bit result"
        ),
    }
    return state, receipt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).eval()
    state, receipt = build_state(model, args.terms, args.group_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    receipt["dense_evaluation_state"] = str(args.out)
    receipt["dense_evaluation_state_sha256"] = sha256(args.out)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "modules"}, indent=2))


if __name__ == "__main__":
    main()
