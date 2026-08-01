"""Matched optimizer controls for the Swin2SR ternary-QAT study."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from clark_quantize.bonsai_muon import BonsaiMuon
from clark_quantize.bonsai_qat8 import BonsaiQAT8
from clark_quantize.qat_optimizer_variants import (
    FactoredAdafactorQAT,
    GroupMirrorSGD,
)
from clark_quantize.ternary_kd import BonsaiMuonFast


OPTIMIZERS = (
    "adamw",
    "adamw_prox",
    "bonsai_qat8",
    "bonsai_qat8_muon",
    "bonsai_muon",
    "bonsai_muon_fast",
    "mirror_sgd",
    "mirror_sgd_adagrad",
    "factored_adafactor",
)


def unique_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    result = []
    for parameter in parameters:
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(parameter)
    return result


def build_optimizer(
    name: str,
    masters: Sequence[torch.nn.Parameter],
    other_parameters: Sequence[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
    group_size: int = 60,
    muon_every: int = 16,
    muon_steps: int = 2,
) -> torch.optim.Optimizer:
    if name not in OPTIMIZERS:
        raise ValueError(f"unsupported optimizer: {name}")
    masters = unique_parameters(masters)
    master_ids = {id(parameter) for parameter in masters}
    others = [
        parameter
        for parameter in unique_parameters(other_parameters)
        if id(parameter) not in master_ids
    ]
    if name in ("adamw", "adamw_prox"):
        return torch.optim.AdamW(
            [*masters, *others],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name in ("mirror_sgd", "mirror_sgd_adagrad"):
        return GroupMirrorSGD(
            [*masters, *others],
            lr=learning_rate,
            weight_decay=weight_decay,
            group_size=group_size,
            group_adagrad=name == "mirror_sgd_adagrad",
        )
    if name == "factored_adafactor":
        return FactoredAdafactorQAT(
            [*masters, *others],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    groups = []
    if name in ("bonsai_muon", "bonsai_muon_fast"):
        if masters:
            groups.append(
                {
                    "params": masters,
                    "role": "muon",
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "momentum": 0.95,
                    "nesterov": True,
                    "ns_steps": 5,
                    "rms_match": True,
                    "state_dtype": torch.float32,
                    "ns_dtype": torch.float32,
                }
            )
        if others:
            groups.append(
                {
                    "params": others,
                    "role": "adamw",
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                }
            )
        optimizer_type = BonsaiMuon if name == "bonsai_muon" else BonsaiMuonFast
        return optimizer_type(groups)

    if masters:
        groups.append(
            {
                "params": masters,
                "role": "matrix_qat8",
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.95),
                "use_muon": name == "bonsai_qat8_muon",
                "muon_every": muon_every,
                "muon_steps": muon_steps,
            }
        )
    if others:
        groups.append(
            {
                "params": others,
                "role": "adamw",
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.95),
            }
        )
    return BonsaiQAT8(groups)


def configure_resumed_optimizer(
    optimizer: torch.optim.Optimizer,
    checkpoint_config: dict | None,
    requested_config: dict,
) -> None:
    previous_name = (checkpoint_config or {}).get("name", "adamw")
    requested_name = requested_config["name"]
    if previous_name != requested_name:
        raise ValueError(
            f"resume optimizer mismatch: checkpoint={previous_name} "
            f"requested={requested_name}"
        )
    for group in optimizer.param_groups:
        group["lr"] = requested_config["learning_rate"]
        group["weight_decay"] = requested_config["weight_decay"]
        if group.get("role") == "matrix_qat8":
            group["use_muon"] = requested_name == "bonsai_qat8_muon"
            group["muon_every"] = requested_config["muon_every"]
            group["muon_steps"] = requested_config["muon_steps"]


def scheduled_prox_rho(
    optimizer_name: str,
    progress: float,
    *,
    hard_fraction: float = 0.28,
    maximum: float = 0.05,
) -> float:
    if (
        optimizer_name != "adamw_prox"
        or progress < hard_fraction
        or maximum <= 0
    ):
        return 0.0
    recovery = min(1.0, max(0.0, (progress - hard_fraction) / (1.0 - hard_fraction)))
    if recovery < 1 / 3:
        return maximum * 0.1
    if recovery < 2 / 3:
        return maximum * 0.4
    return maximum


@torch.no_grad()
def apply_proximal_pull(
    master_quantizers: Sequence[tuple[torch.nn.Parameter, torch.nn.Module]],
    rho: float,
) -> None:
    if rho <= 0:
        return
    for master, quantizer in master_quantizers:
        projected = quantizer.project(master).detach()
        master.lerp_(projected.to(device=master.device, dtype=master.dtype), rho)


def attach_bonsai_callbacks(
    master_quantizers: Sequence[tuple[torch.nn.Parameter, torch.nn.Module]],
) -> None:
    for master, quantizer in master_quantizers:
        master._bonsai_make_quantized = lambda master=master, quantizer=quantizer: (
            quantizer.project(master)
        )
