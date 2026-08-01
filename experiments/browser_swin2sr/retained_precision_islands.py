"""Exact mixed-precision island contracts for the Swin2SR ternary model."""

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


def symmetric_int7_per_output_components(
    weight: torch.Tensor,
    scale_method: str = "max",
    scale_blend: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed int7 codes and FP16-rounded per-output scales."""

    if weight.ndim != 4 or tuple(weight.shape[2:]) != (3, 3):
        raise ValueError("int7 retained weights must be 3x3 convolutions")
    rows = weight.reshape(weight.shape[0], -1)
    limit = 63
    maximum = rows.abs().amax(dim=1)
    maximum_scales = maximum / limit
    scales = maximum_scales
    if scale_method == "mse_iterative":
        if not 0 <= scale_blend <= 1:
            raise ValueError("int7 scale blend must be between zero and one")
        for _iteration in range(4):
            safe = torch.where(scales > 0, scales, torch.ones_like(scales))
            codes = torch.round(rows / safe[:, None]).clamp(-limit, limit)
            denominator = codes.square().sum(dim=1)
            proposed = (rows * codes).sum(dim=1) / denominator.clamp_min(1)
            scales = torch.minimum(proposed.clamp_min(0), maximum_scales)
        scales = torch.lerp(maximum_scales, scales, scale_blend)
    elif scale_method != "max":
        raise ValueError(f"unsupported int7 scale method: {scale_method}")
    scales = scales.to(torch.float16).to(weight.dtype)
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round(rows / safe[:, None]).clamp(-limit, limit)
    return codes.to(torch.int8), scales


def symmetric_int7_per_output(
    weight: torch.Tensor,
    scale_method: str = "max",
    scale_blend: float = 1.0,
) -> torch.Tensor:
    """Deployment-exact signed int7 with one FP16 scale per output channel."""

    codes, scales = symmetric_int7_per_output_components(
        weight,
        scale_method,
        scale_blend,
    )
    rows = codes.to(weight.dtype) * scales[:, None]
    return rows.reshape_as(weight)


@torch.no_grad()
def boost_int7_boundary_gradients(
    parameters: list[torch.nn.Parameter],
    boost: float,
    temperature: float,
) -> dict:
    """Amplify gradients near signed-int7 code-transition boundaries."""

    if boost < 0:
        raise ValueError("int7 boundary boost must be non-negative")
    if temperature <= 0:
        raise ValueError("int7 boundary temperature must be positive")
    gains = []
    modified = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if parameter.ndim != 4 or tuple(parameter.shape[2:]) != (3, 3):
            raise ValueError("int7 boundary parameters must be 3x3 convolutions")
        rows = parameter.detach().reshape(parameter.shape[0], -1)
        scales = (rows.abs().amax(dim=1) / 63).to(torch.float16).to(rows.dtype)
        safe = torch.where(scales > 0, scales, torch.ones_like(scales))
        normalized = (rows / safe[:, None]).clamp(-63, 63)
        fractional = (normalized - torch.round(normalized)).abs()
        distance = (0.5 - fractional).abs()
        gain = 1.0 + boost * torch.exp(-distance / temperature)
        parameter.grad.mul_(gain.reshape_as(parameter.grad))
        gains.append(gain)
        modified += parameter.numel()
    if not gains:
        return {
            "parameters": 0,
            "mean_gain": 1.0,
            "max_gain": 1.0,
        }
    combined = torch.cat([gain.reshape(-1) for gain in gains])
    return {
        "parameters": modified,
        "mean_gain": float(combined.mean().item()),
        "max_gain": float(combined.max().item()),
    }


class SymmetricInt7PerOutputSTE(torch.nn.Module):
    """Use the packed-runtime int7 boundary with a master-weight gradient."""

    def __init__(
        self,
        scale_method: str = "max",
        scale_blend: float = 1.0,
    ) -> None:
        super().__init__()
        if scale_method not in {"max", "mse_iterative"}:
            raise ValueError(f"unsupported int7 scale method: {scale_method}")
        if not 0 <= scale_blend <= 1:
            raise ValueError("int7 scale blend must be between zero and one")
        self.scale_method = scale_method
        self.scale_blend = scale_blend

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        quantized = symmetric_int7_per_output(
            weight,
            self.scale_method,
            self.scale_blend,
        )
        return weight + (quantized - weight).detach()


def packed_trit_bytes(values: int) -> int:
    """Clark's format packs twenty ternary values in one u32."""

    if values < 0:
        raise ValueError("weight values must be non-negative")
    return math.ceil(values / 20) * 4


def fp16_roundtrip(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().to(torch.float16).to(torch.float32)


def retained_precision_configs(payload: dict) -> list[dict]:
    """Return the effective retained-precision islands in lineage order."""

    candidates = list(payload.get("retained_precision_history", []))
    current = payload.get("retained_fp16_training")
    if current is not None:
        candidates.append(current)
    ordered: list[dict] = []
    positions: dict[str, int] = {}
    for candidate in candidates:
        config = dict(candidate)
        island = config.get("island")
        if island not in ISLAND_SPECS:
            raise ValueError(f"unknown retained-precision island: {island}")
        if config.get("precision") not in {"fp16", "int7", "ternary"}:
            raise ValueError(
                f"unsupported retained-precision value for {island}: "
                f"{config.get('precision')}"
            )
        if island in positions:
            ordered[positions[island]] = config
        else:
            positions[island] = len(ordered)
            ordered.append(config)
    return ordered


def configure_fp16_island_for_loading(surfaces, spec: IslandSpec) -> dict:
    """Recreate an FP16 deployment island before loading its state dict."""

    resolved = _resolved_surfaces(surfaces, spec)
    configured = []
    for surface in resolved:
        module = surface.module
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
        configured.append({
            "name": surface.name,
            "values": module.parametrizations.weight.original.numel(),
        })
    return {
        "island": spec.name,
        "precision": "fp16",
        "configured": configured,
        "weight_values": sum(item["values"] for item in configured),
    }


def configure_int7_island_for_loading(
    surfaces,
    spec: IslandSpec,
    scale_method: str = "max",
    scale_blend: float = 1.0,
) -> dict:
    """Recreate the signed-int7 deployment island before checkpoint loading."""

    resolved = _resolved_surfaces(surfaces, spec)
    configured = []
    for surface in resolved:
        module = surface.module
        parametrize.remove_parametrizations(
            module,
            "weight",
            leave_parametrized=False,
        )
        parametrize.register_parametrization(
            module,
            "weight",
            SymmetricInt7PerOutputSTE(scale_method, scale_blend),
        )
        configured.append({
            "name": surface.name,
            "values": module.parametrizations.weight.original.numel(),
        })
    return {
        "island": spec.name,
        "precision": "int7",
        "bit_width": 7,
        "granularity": "per_output",
        "scale_method": scale_method,
        "scale_blend": scale_blend,
        "configured": configured,
        "weight_values": sum(item["values"] for item in configured),
    }


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


def initialize_trainable_int7_island(
    model,
    surfaces,
    spec: IslandSpec,
    scale_method: str = "max",
    scale_blend: float = 1.0,
) -> dict:
    """Start QAT at the exact packed int7 deployment boundary."""

    resolved = _resolved_surfaces(surfaces, spec)
    for surface in surfaces:
        surface.quantizer.amount.fill_(1.0)
    parent_weights = {
        surface.name: surface.module.weight.detach().clone()
        for surface in resolved
    }
    model.requires_grad_(False)
    initialized = []
    for surface in resolved:
        module = surface.module
        parent = parent_weights[surface.name]
        parametrize.remove_parametrizations(
            module,
            "weight",
            leave_parametrized=False,
        )
        module.weight.data.copy_(parent)
        parametrize.register_parametrization(
            module,
            "weight",
            SymmetricInt7PerOutputSTE(scale_method, scale_blend),
        )
        trainable = module.parametrizations.weight.original
        trainable.requires_grad_(True)
        expected = symmetric_int7_per_output(
            parent,
            scale_method,
            scale_blend,
        )
        maximum = float((module.weight.detach() - expected).abs().max().item())
        if maximum != 0.0:
            raise RuntimeError(
                f"int7 island initialization is not exact: {surface.name}: "
                f"{maximum}"
            )
        initialized.append({
            "name": surface.name,
            "values": trainable.numel(),
            "initial_int7_recovery_max_abs": maximum,
            "initial_int7_vs_parent_max_abs": float(
                (expected - parent).abs().max().item()
            ),
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
            f"unexpected trainable int7 island values: {trainable_values} "
            f"!= {spec.weight_values}"
        )
    return {
        "precision": "int7",
        "bit_width": 7,
        "granularity": "per_output",
        "scale_method": scale_method,
        "scale_blend": scale_blend,
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
        difference = surface.module.weight.detach() - hard_weights[surface.name]
        maximum = float(difference.abs().max().item())
        reference_scale = max(
            1.0,
            float(hard_weights[surface.name].abs().max().item()),
        )
        tolerance = float(
            4 * torch.finfo(hard_weights[surface.name].dtype).eps * reference_scale
        )
        if maximum > tolerance:
            raise RuntimeError(
                f"ternary control initialization changed the hard weight: "
                f"{surface.name}: {maximum} > {tolerance}"
            )
        initialized.append({
            "name": surface.name,
            "values": master.numel(),
            "initial_hard_ternary_max_abs": maximum,
            "initial_hard_ternary_tolerance": tolerance,
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


def initialize_trainable_island(
    model,
    surfaces,
    spec: IslandSpec,
    precision: str,
    int7_scale_method: str = "max",
    int7_scale_blend: float = 1.0,
) -> dict:
    if precision == "fp16":
        return initialize_trainable_fp16_island(model, surfaces, spec)
    if precision == "ternary":
        return initialize_trainable_ternary_island(model, surfaces, spec)
    if precision == "int7":
        return initialize_trainable_int7_island(
            model,
            surfaces,
            spec,
            int7_scale_method,
            int7_scale_blend,
        )
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
    int7_code_bytes = 0
    int7_scale_bytes = 0
    for surface in resolved:
        values = surface.module.parametrizations.weight.original.numel()
        group_size = int(surface.quantizer.group_size)
        if group_size <= 0:
            raise ValueError(f"invalid group size for {surface.name}")
        tensor_code_bytes = packed_trit_bytes(values)
        tensor_scale_bytes = math.ceil(values / group_size) * 2
        code_bytes += tensor_code_bytes
        scale_bytes += tensor_scale_bytes
        tensor_int7_code_bytes = math.ceil(values * 7 / 8)
        tensor_int7_scale_bytes = int(surface.module.weight.shape[0]) * 2
        int7_code_bytes += tensor_int7_code_bytes
        int7_scale_bytes += tensor_int7_scale_bytes
        modules.append({
            "name": surface.name,
            "values": values,
            "ternary_group_size": group_size,
            "ternary_code_bytes": tensor_code_bytes,
            "ternary_scale_bytes": tensor_scale_bytes,
            "int7_code_bytes": tensor_int7_code_bytes,
            "int7_scale_bytes": tensor_int7_scale_bytes,
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
        "retained_int7_code_bytes": int7_code_bytes,
        "retained_int7_scale_bytes": int7_scale_bytes,
        "retained_int7_data_bytes": int7_code_bytes + int7_scale_bytes,
        "projected_int7_data_byte_delta": (
            int7_code_bytes + int7_scale_bytes - code_bytes - scale_bytes
        ),
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
