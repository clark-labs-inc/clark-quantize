"""Optimizer construction for sensitive full-surface Swin2SR training."""

from __future__ import annotations

import torch

from clark_quantize.bonsai_qat8 import BonsaiQAT8


FULL_SURFACE_OPTIMIZERS = (
    "adamw",
    "bonsai_qat8_muon",
)
DEFAULT_FULL_SURFACE_OPTIMIZER = "bonsai_qat8_muon"


def _validate_groups(
    groups: dict[str, list[torch.nn.Parameter]],
) -> None:
    expected = {
        "matrix_masters",
        "convolution_masters",
        "corrections",
    }
    if set(groups) != expected:
        raise ValueError(
            f"full-surface optimizer groups must be {sorted(expected)}"
        )
    identities = [
        id(parameter)
        for parameters in groups.values()
        for parameter in parameters
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("full-surface optimizer groups overlap")


def build_full_surface_optimizer(
    groups: dict[str, list[torch.nn.Parameter]],
    *,
    optimizer_name: str = DEFAULT_FULL_SURFACE_OPTIMIZER,
    learning_rate: float,
    convolution_learning_rate_multiplier: float = 0.25,
    correction_learning_rate_multiplier: float = 0.10,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    if optimizer_name not in FULL_SURFACE_OPTIMIZERS:
        raise ValueError(f"unsupported full-surface optimizer: {optimizer_name}")
    if learning_rate <= 0:
        raise ValueError("surface learning rate must be positive")
    if convolution_learning_rate_multiplier <= 0:
        raise ValueError("convolution learning-rate multiplier must be positive")
    if correction_learning_rate_multiplier <= 0:
        raise ValueError("correction learning-rate multiplier must be positive")
    _validate_groups(groups)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            [
                {
                    "params": groups["matrix_masters"],
                    "role": "matrix_adamw",
                    "lr": learning_rate,
                },
                {
                    "params": groups["convolution_masters"],
                    "role": "convolution_adamw",
                    "lr": (
                        learning_rate
                        * convolution_learning_rate_multiplier
                    ),
                },
                {
                    "params": groups["corrections"],
                    "role": "correction_adamw",
                    "lr": learning_rate * correction_learning_rate_multiplier,
                },
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    return BonsaiQAT8([
        {
            "params": groups["matrix_masters"],
            "role": "matrix_qat8",
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.95),
            "use_muon": True,
            "muon_every": 16,
            "muon_steps": 2,
        },
        {
            "params": groups["convolution_masters"],
            "role": "adamw",
            "lr": learning_rate * convolution_learning_rate_multiplier,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.95),
        },
        {
            "params": groups["corrections"],
            "role": "adamw",
            "lr": learning_rate * correction_learning_rate_multiplier,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.95),
        },
    ])


def optimizer_config(
    *,
    optimizer_name: str,
    learning_rate: float,
    convolution_learning_rate_multiplier: float,
    correction_learning_rate_multiplier: float,
    reset_from_base: bool,
    weight_decay: float = 1e-4,
) -> dict:
    config = {
        "name": optimizer_name,
        "learning_rate": learning_rate,
        "convolution_learning_rate_multiplier": (
            convolution_learning_rate_multiplier
        ),
        "correction_learning_rate_multiplier": (
            correction_learning_rate_multiplier
        ),
        "weight_decay": weight_decay,
        "reset_from_base": reset_from_base,
    }
    if optimizer_name == "adamw":
        config.update({
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        })
    else:
        config.update({
            "matrix_betas": [0.9, 0.95],
            "muon_every": 16,
            "muon_steps": 2,
        })
    return config
