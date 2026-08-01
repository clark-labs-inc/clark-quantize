"""Dense and runtime state materialization for trained Swin2SR quantizers."""

from __future__ import annotations

import torch

from rotation_control import block_hadamard


def deployment_state_dict(
    student,
    quantizers,
    *,
    preserve_rotated_basis: bool,
) -> dict[str, torch.Tensor]:
    for _, quantizer, _ in quantizers:
        quantizer.amount.fill_(1.0)
    state = {
        name: value.detach().cpu()
        for name, value in student.state_dict().items()
        if ".parametrizations.weight." not in name
        and not name.endswith(".amount")
        and not name.endswith(".input_shift")
    }
    for name, quantizer, module in quantizers:
        weight = module.weight.detach()
        effective_weight = (
            block_hadamard(weight, quantizer.rotation_block)
            if quantizer.rotation == "hadamard" and not preserve_rotated_basis
            else weight
        )
        state[f"{name}.weight"] = effective_weight.cpu()
        if module.bias is not None:
            folded = torch.mv(weight, module.input_shift.detach()) + module.bias.detach()
            state[f"{name}.bias"] = folded.cpu()
    return state


def hard_state_dict(student, quantizers) -> dict[str, torch.Tensor]:
    return deployment_state_dict(
        student,
        quantizers,
        preserve_rotated_basis=False,
    )


def runtime_state_dict(student, quantizers) -> dict[str, torch.Tensor]:
    return deployment_state_dict(
        student,
        quantizers,
        preserve_rotated_basis=True,
    )
