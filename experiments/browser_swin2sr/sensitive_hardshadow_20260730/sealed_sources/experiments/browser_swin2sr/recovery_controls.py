"""Winner-only recovery controls for ternary Swin2SR.

These controls are deliberately separate from the active factorial race. They
are introduced only after the race chooses a base optimizer/quantizer recipe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

from clark_air.quant.optimal_ternary import _pad_flat, quantize_ternary_mse
from clark_quantize.bonsai_qat8 import BonsaiQAT8
from island_policy import island_parameters
from quantizer_controls import ProgressiveLowBit, quantize_ternary_absmean
from rotation_control import block_hadamard


class ScalarDecomposedLowBit(ProgressiveLowBit):
    """CompSRT-inspired two-scalar correction over groupwise ternary codes.

    The code stream remains ternary. One learned multiplier and one learned
    offset are shared by the tensor, adding two scalar parameters per selected
    matrix. The offset is expressed relative to each group's base scale.
    """

    def __init__(
        self,
        mode: str,
        method: str,
        group_size: int,
        start_fraction: float,
        end_fraction: float,
        initial_weight: torch.Tensor,
        rotation: str = "none",
        rotation_block: int = 4,
    ) -> None:
        if mode != "ternary" or method not in {"optimal_mse", "absmean"}:
            raise ValueError(
                "scalar decomposition requires groupwise ternary quantization"
            )
        super().__init__(
            mode,
            method,
            group_size,
            start_fraction,
            end_fraction,
            initial_weight,
            rotation,
            rotation_block,
        )
        self.log_scale_multiplier = torch.nn.Parameter(torch.zeros(()))
        self.offset_multiplier = torch.nn.Parameter(torch.zeros(()))

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        quantized = (
            quantize_ternary_mse(weight, group_size=self.group_size)
            if self.method == "optimal_mse"
            else quantize_ternary_absmean(weight, self.group_size)
        )
        padded, _pad = _pad_flat(weight, self.group_size)
        scale = quantized.scales.repeat_interleave(self.group_size)
        scale = scale.to(device=weight.device, dtype=weight.dtype)
        codes = quantized.codes.to(device=weight.device, dtype=weight.dtype)
        corrected_scale = scale * self.log_scale_multiplier.exp()
        corrected_offset = scale * self.offset_multiplier
        projected = codes * corrected_scale + corrected_offset
        return projected[: padded.numel() - quantized.pad].reshape_as(weight)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        projected = self.project(weight)
        amount = self.amount.to(device=weight.device, dtype=weight.dtype)
        master_ste = weight + amount * (projected.detach() - weight.detach())
        scalar_gradient = amount * (projected - projected.detach())
        return master_ste + scalar_gradient


def normalized_feature_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mode: str = "layernorm",
) -> torch.Tensor:
    if mode == "layernorm":
        student = F.layer_norm(student, (student.shape[-1],))
        teacher = F.layer_norm(teacher, (teacher.shape[-1],))
        return F.smooth_l1_loss(student, teacher)
    if mode == "l2":
        dimensions = tuple(range(1, student.ndim))
        student = student / student.norm(
            dim=dimensions,
            keepdim=True,
        ).clamp_min(1e-8)
        teacher = teacher / teacher.norm(
            dim=dimensions,
            keepdim=True,
        ).clamp_min(1e-8)
        # Averaging over every value after unit normalization makes the loss
        # vanish as feature resolution grows. Sum each vector distance and
        # average only across the batch; /2 yields cosine distance.
        return (student - teacher).square().sum(dim=dimensions).mean() / 2
    raise ValueError(f"unknown feature normalization: {mode}")


def localized_feature_loss(
    student_states,
    teacher_states,
    active_stage: int | None,
    mode: str = "layernorm",
) -> torch.Tensor:
    """Compare one encoder stage, or all stages for final joint recovery."""

    student_stages = list(student_states[1:])
    teacher_stages = list(teacher_states[1:])
    if len(student_stages) != len(teacher_stages) or not student_stages:
        raise ValueError("student and teacher stage states must align")
    if active_stage is None:
        indices = range(len(student_stages))
    elif 0 <= active_stage < len(student_stages):
        indices = (active_stage,)
    else:
        raise ValueError(f"active stage is out of range: {active_stage}")
    losses = [
        normalized_feature_loss(
            student_stages[index],
            teacher_stages[index],
            mode,
        )
        for index in indices
    ]
    return torch.stack(losses).mean()


def scalar_storage_bytes(selected_modules: int, scalar_bytes: int = 2) -> int:
    if selected_modules < 0 or scalar_bytes < 1:
        raise ValueError("storage dimensions must be non-negative and positive")
    return selected_modules * 2 * scalar_bytes


def recovery_loss_weights(recipe: str) -> dict[str, float]:
    recipes = {
        "balanced": {
            "teacher": 0.55,
            "ground_truth": 0.20,
            "edge": 0.15,
            "feature": 0.10,
        },
        "teacher_heavy": {
            "teacher": 0.72,
            "ground_truth": 0.08,
            "edge": 0.08,
            "feature": 0.12,
        },
    }
    if recipe not in recipes:
        raise ValueError(f"unknown recovery loss recipe: {recipe}")
    return recipes[recipe]


def add_recovery_parametrizations(
    model: torch.nn.Module,
    *,
    group_size: int,
    method: str,
    rotation: str,
    rotation_block: int,
    scalar_decomposition: bool,
) -> list[tuple[str, ProgressiveLowBit, torch.nn.Linear]]:
    quantizers = []
    quantizer_type = (
        ScalarDecomposedLowBit
        if scalar_decomposition
        else ProgressiveLowBit
    )
    for name, module in model.named_modules():
        if not (
            isinstance(module, torch.nn.Linear)
            and "continuous_position_bias_mlp" not in name
            and (
                ".attention." in name
                or ".intermediate." in name
                or ".output.dense" in name
            )
        ):
            continue
        schedule = (0.0, 0.12) if ".attention." in name else (0.08, 0.28)
        if rotation == "hadamard":
            module.weight.data.copy_(
                block_hadamard(module.weight.data, rotation_block)
            )
        quantizer = quantizer_type(
            "ternary",
            method,
            group_size,
            *schedule,
            module.weight,
            rotation,
            rotation_block,
        )
        quantizer.set_progress(1.0)
        parametrize.register_parametrization(module, "weight", quantizer)
        if module.bias is not None:
            module.register_parameter(
                "input_shift",
                torch.nn.Parameter(
                    torch.zeros(
                        module.in_features,
                        device=module.weight.device,
                    )
                ),
            )

        if rotation == "hadamard" or module.bias is not None:
            def prepare_inputs(target, inputs):
                values = inputs[0]
                active = target.parametrizations.weight[0]
                if active.rotation == "hadamard":
                    values = block_hadamard(values, active.rotation_block)
                if hasattr(target, "input_shift"):
                    values = values + target.input_shift
                return (values, *inputs[1:])

            module.register_forward_pre_hook(prepare_inputs)
        quantizers.append((name, quantizer, module))
    if not quantizers:
        raise ValueError("no Swin2SR core projections were selected")
    return quantizers


def recovery_parameter_groups(
    model: torch.nn.Module,
    quantizers,
    island_policy_name: str,
) -> dict[str, list[torch.nn.Parameter]]:
    model.requires_grad_(False)
    masters = []
    scalars = []
    corrections = []
    for _name, quantizer, module in quantizers:
        master = module.parametrizations.weight.original
        master.requires_grad_(True)
        masters.append(master)
        for parameter in quantizer.parameters():
            parameter.requires_grad_(True)
            scalars.append(parameter)
        if module.bias is not None:
            module.bias.requires_grad_(True)
            module.input_shift.requires_grad_(True)
            corrections.extend((module.bias, module.input_shift))

    named_parameters = dict(model.named_parameters())
    islands = {"normalization": [], "residual_body": []}
    for record in island_parameters(model, island_policy_name):
        parameter = named_parameters.get(record.name)
        if parameter is None:
            raise ValueError(f"island parameter is missing: {record.name}")
        parameter.requires_grad_(True)
        islands[record.role].append(parameter)

    seen = set()
    groups = {}
    for name, parameters in {
        "masters": masters,
        "scalars": scalars,
        "corrections": corrections,
        **islands,
    }.items():
        unique = []
        for parameter in parameters:
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(parameter)
        groups[name] = unique
    return groups


def build_recovery_optimizer(
    name: str,
    groups: dict[str, list[torch.nn.Parameter]],
    *,
    learning_rate: float,
    weight_decay: float,
    muon_every: int,
    muon_steps: int,
    scalar_lr_multiplier: float = 100.0,
) -> torch.optim.Optimizer:
    if name not in {"adamw", "bonsai_qat8", "bonsai_qat8_muon"}:
        raise ValueError(f"unsupported recovery optimizer: {name}")
    if learning_rate <= 0 or scalar_lr_multiplier <= 0:
        raise ValueError("recovery learning rates must be positive")
    multipliers = {
        "masters": 1.0,
        "scalars": scalar_lr_multiplier,
        "corrections": 1.0,
        "normalization": 0.10,
        "residual_body": 0.05,
    }
    parameter_groups = []
    for role, parameters in groups.items():
        if not parameters:
            continue
        group = {
            "params": parameters,
            "lr": learning_rate * multipliers[role],
            "weight_decay": weight_decay,
            "betas": (0.9, 0.95),
            "role": "matrix_qat8" if role == "masters" else "adamw",
        }
        if role == "masters":
            group.update({
                "use_muon": name == "bonsai_qat8_muon",
                "muon_every": muon_every,
                "muon_steps": muon_steps,
            })
        parameter_groups.append(group)
    if name == "adamw":
        return torch.optim.AdamW(parameter_groups)
    return BonsaiQAT8(parameter_groups)
