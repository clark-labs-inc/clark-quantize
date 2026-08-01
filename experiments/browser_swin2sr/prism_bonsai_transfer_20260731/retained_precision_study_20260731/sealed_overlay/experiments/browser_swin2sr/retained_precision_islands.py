"""Exact retained-FP16 island contracts for the Swin2SR ternary model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn.utils import parametrize


@dataclass(frozen=True)
class IslandSpec:
    name: str
    modules: tuple[tuple[str, int], ...]

    @property
    def weight_values(self) -> int:
        return sum(values for _name, values in self.modules)


IMAGE_IO_FP16 = IslandSpec(
    name="image_io_fp16",
    modules=(
        ("swin2sr.first_convolution", 1_620),
        ("upsample.conv", 6_480),
    ),
)

RSTB_FUSION_FP16 = IslandSpec(
    name="rstb_fusion_fp16",
    modules=tuple(
        (f"swin2sr.encoder.stages.{stage}.conv", 32_400)
        for stage in range(4)
    ),
)

ISLAND_SPECS = {
    spec.name: spec
    for spec in (IMAGE_IO_FP16, RSTB_FUSION_FP16)
}


class Fp16RoundTripSTE(torch.nn.Module):
    """Use deployment-faithful FP16 values with a straight-through gradient."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        rounded = weight.to(torch.float16).to(weight.dtype)
        return weight + (rounded - weight).detach()


def packed_trit_bytes(values: int) -> int:
    """Clark's format packs twenty ternary values in one u32."""

    if values < 0:
        raise ValueError("weight values must be non-negative")
    return math.ceil(values / 20) * 4


def fp16_roundtrip(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().to(torch.float16).to(torch.float32)


def initialize_trainable_fp16_island(model, surfaces, spec: IslandSpec) -> dict:
    """Replace selected ternary projections with zero-discontinuity FP16 STEs."""

    resolved = _resolved_surfaces(surfaces, spec)
    for surface in surfaces:
        surface.quantizer.amount.fill_(1.0)
    hard_weights = {
        surface.name: surface.module.weight.detach().clone()
        for surface in resolved
    }
    model.requires_grad_(False)
    initialized = []
    for surface in resolved:
        module = surface.module
        master = module.parametrizations.weight.original
        master.data.copy_(hard_weights[surface.name])
        parametrize.remove_parametrizations(
            module,
            "weight",
            leave_parametrized=False,
        )
        parametrize.register_parametrization(
            module,
            "weight",
            Fp16RoundTripSTE(),
        )
        trainable = module.parametrizations.weight.original
        trainable.requires_grad_(True)
        initial = module.weight.detach()
        reference = hard_weights[surface.name].to(torch.float16).to(initial.dtype)
        maximum = float((initial - reference).abs().max().item())
        if maximum != 0.0:
            raise RuntimeError(
                f"FP16 island initialization is not an exact hard-weight roundtrip: "
                f"{surface.name}: {maximum}"
            )
        initialized.append({
            "name": surface.name,
            "values": trainable.numel(),
            "initial_fp16_vs_hard_fp16_max_abs": maximum,
        })
    parameters = [
        surface.module.parametrizations.weight.original
        for surface in resolved
    ]
    trainable_values = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_values != spec.weight_values:
        raise RuntimeError(
            f"unexpected trainable island values: {trainable_values} "
            f"!= {spec.weight_values}"
        )
    return {
        "precision": "fp16",
        "parameters": parameters,
        "initialized": initialized,
        "trainable_values": trainable_values,
    }


def initialize_trainable_ternary_island(model, surfaces, spec: IslandSpec) -> dict:
    """Train the same hard-started tensors while retaining ternary projection."""

    resolved = _resolved_surfaces(surfaces, spec)
    for surface in surfaces:
        surface.quantizer.amount.fill_(1.0)
    hard_weights = {
        surface.name: surface.module.weight.detach().clone()
        for surface in resolved
    }
    model.requires_grad_(False)
    initialized = []
    for surface in resolved:
        master = surface.module.parametrizations.weight.original
        master.data.copy_(hard_weights[surface.name])
        master.requires_grad_(True)
        maximum = float(
            (surface.module.weight.detach() - hard_weights[surface.name])
            .abs().max().item()
        )
        if maximum != 0.0:
            raise RuntimeError(
                f"ternary control initialization changed the hard weight: "
                f"{surface.name}: {maximum}"
            )
        initialized.append({
            "name": surface.name,
            "values": master.numel(),
            "initial_hard_ternary_max_abs": maximum,
        })
    parameters = [
        surface.module.parametrizations.weight.original
        for surface in resolved
    ]
    trainable_values = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_values != spec.weight_values:
        raise RuntimeError(
            f"unexpected trainable control values: {trainable_values} "
            f"!= {spec.weight_values}"
        )
    return {
        "precision": "ternary",
        "parameters": parameters,
        "initialized": initialized,
        "trainable_values": trainable_values,
    }


def initialize_trainable_island(model, surfaces, spec: IslandSpec, precision: str) -> dict:
    if precision == "fp16":
        return initialize_trainable_fp16_island(model, surfaces, spec)
    if precision == "ternary":
        return initialize_trainable_ternary_island(model, surfaces, spec)
    raise ValueError(f"unsupported retained-island precision: {precision}")


def _resolved_surfaces(surfaces, spec: IslandSpec):
    by_name = {surface.name: surface for surface in surfaces}
    if len(by_name) != len(surfaces):
        raise ValueError("surface names are not unique")
    resolved = []
    for name, expected_values in spec.modules:
        surface = by_name.get(name)
        if surface is None:
            raise ValueError(f"retained island surface is missing: {name}")
        if surface.category not in {"image_io_conv", "residual_body_conv"}:
            raise ValueError(f"retained island is not a convolution: {name}")
        if surface.quantizer.rotation != "none":
            raise ValueError(f"retained convolution has an unexpected rotation: {name}")
        master = surface.module.parametrizations.weight.original
        if master.numel() != expected_values:
            raise ValueError(
                f"retained island shape changed: {name}: "
                f"{master.numel()} != {expected_values}"
            )
        resolved.append(surface)
    return resolved


def storage_projection(surfaces, spec: IslandSpec) -> dict:
    resolved = _resolved_surfaces(surfaces, spec)
    modules = []
    code_bytes = 0
    scale_bytes = 0
    for surface in resolved:
        values = surface.module.parametrizations.weight.original.numel()
        group_size = int(surface.quantizer.group_size)
        if group_size <= 0:
            raise ValueError(f"invalid group size for {surface.name}")
        tensor_code_bytes = packed_trit_bytes(values)
        tensor_scale_bytes = math.ceil(values / group_size) * 2
        code_bytes += tensor_code_bytes
        scale_bytes += tensor_scale_bytes
        modules.append({
            "name": surface.name,
            "values": values,
            "ternary_group_size": group_size,
            "ternary_code_bytes": tensor_code_bytes,
            "ternary_scale_bytes": tensor_scale_bytes,
        })
    fp16_bytes = spec.weight_values * 2
    return {
        "island": spec.name,
        "weight_values": spec.weight_values,
        "modules": modules,
        "current_ternary_code_bytes": code_bytes,
        "current_fp16_scale_bytes": scale_bytes,
        "retained_fp16_weight_bytes": fp16_bytes,
        "projected_data_byte_delta": fp16_bytes - code_bytes - scale_bytes,
    }


def materialize_fp16_island(
    baseline_state: dict[str, torch.Tensor],
    surfaces,
    spec: IslandSpec,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Replace only declared hard-ternary weights with FP16-rounded masters."""

    resolved = _resolved_surfaces(surfaces, spec)
    result = {name: value.detach().cpu().clone() for name, value in baseline_state.items()}
    changes = []
    for surface in resolved:
        key = f"{surface.name}.weight"
        if key not in baseline_state:
            raise ValueError(f"canonical dense state is missing {key}")
        master = surface.module.parametrizations.weight.original.detach().cpu().float()
        retained = fp16_roundtrip(master)
        if retained.shape != baseline_state[key].shape:
            raise ValueError(f"retained island shape differs from canonical state: {key}")
        difference = retained - baseline_state[key].detach().cpu().float()
        result[key] = retained
        changes.append({
            "name": surface.name,
            "values": retained.numel(),
            "max_abs_delta_from_hard_ternary": float(difference.abs().max().item()),
            "mean_abs_delta_from_hard_ternary": float(difference.abs().mean().item()),
            "fp16_roundtrip_exact": bool(
                torch.equal(retained, retained.to(torch.float16).to(torch.float32))
            ),
        })
    receipt = storage_projection(surfaces, spec)
    receipt["changed_weights"] = changes
    receipt["changed_tensor_count"] = len(changes)
    return result, receipt
