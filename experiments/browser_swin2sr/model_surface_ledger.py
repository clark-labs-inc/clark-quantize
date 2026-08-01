#!/usr/bin/env python3
"""Account for every Swin2SR parameter and projected ternary deployment cost."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from transformers import Swin2SRForImageSuperResolution

from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


TERNARY_CATEGORIES = (
    "core_linear",
    "position_bias_linear",
    "residual_body_conv",
    "image_io_conv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def parameter_categories(model: torch.nn.Module) -> dict[str, list[dict]]:
    module_by_parameter = {}
    for module_name, module in model.named_modules():
        for parameter_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            module_by_parameter[full_name] = (module_name, module, parameter_name)
    categories = defaultdict(list)
    for name, parameter in model.named_parameters():
        module_name, module, parameter_name = module_by_parameter[name]
        if parameter_name != "weight":
            category = "non_weight_parameter"
        elif isinstance(module, torch.nn.Linear):
            if "continuous_position_bias_mlp" in module_name:
                category = "position_bias_linear"
            elif (
                ".attention." in module_name
                or ".intermediate." in module_name
                or ".output.dense" in module_name
            ):
                category = "core_linear"
            else:
                category = "other_weight"
        elif isinstance(module, torch.nn.Conv2d):
            if module_name in {"swin2sr.first_convolution", "upsample.conv"}:
                category = "image_io_conv"
            else:
                category = "residual_body_conv"
        else:
            category = "other_weight"
        categories[category].append({
            "name": name,
            "shape": list(parameter.shape),
            "values": parameter.numel(),
        })
    return dict(categories)


def projected_storage(
    categories: dict[str, list[dict]],
    selected_categories: set[str],
    group_size: int,
    other_parameter_bits: int,
) -> dict:
    selected_values = 0
    code_bits = 0
    scale_bits = 0
    protected_weight_values = 0
    other_parameter_values = 0
    for category, tensors in categories.items():
        for tensor in tensors:
            values = tensor["values"]
            if category in selected_categories:
                selected_values += values
                code_bits += math.ceil(values / 20) * 32
                scale_bits += math.ceil(values / group_size) * 16
            elif category == "non_weight_parameter":
                other_parameter_values += values
            else:
                protected_weight_values += values
    total_values = selected_values + protected_weight_values + other_parameter_values
    protected_weight_bits = protected_weight_values * 16
    other_bits = other_parameter_values * other_parameter_bits
    total_bits = code_bits + scale_bits + protected_weight_bits + other_bits
    return {
        "group_size": group_size,
        "selected_categories": sorted(selected_categories),
        "ternary_values": selected_values,
        "ternary_fraction": selected_values / total_values,
        "packed_code_bits": code_bits,
        "scale_bits": scale_bits,
        "protected_fp16_weight_bits": protected_weight_bits,
        "other_parameter_bits": other_bits,
        "other_parameter_precision": other_parameter_bits,
        "total_bits": total_bits,
        "projected_bytes": math.ceil(total_bits / 8),
        "whole_model_bits_per_parameter": total_bits / total_values,
    }


def build_ledger(model: torch.nn.Module) -> dict:
    categories = parameter_categories(model)
    counts = {
        category: {
            "tensors": len(tensors),
            "values": sum(tensor["values"] for tensor in tensors),
        }
        for category, tensors in sorted(categories.items())
    }
    total_parameters = sum(item["values"] for item in counts.values())
    policies = {}
    selections = {
        "current_core": {"core_linear"},
        "core_plus_residual": {"core_linear", "residual_body_conv"},
        "all_matrix_and_conv": set(TERNARY_CATEGORIES),
    }
    for name, selected in selections.items():
        policies[name] = [
            projected_storage(categories, selected, group_size, other_bits)
            for group_size in (60, 120, 240)
            for other_bits in (16, 8)
        ]
    return {
        "format": "backdropkit-swin2sr-surface-ledger-v1",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameters": total_parameters,
        "categories": counts,
        "policies": policies,
    }


def main() -> None:
    args = parse_args()
    model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    ledger = build_ledger(model)
    payload = json.dumps(ledger, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
