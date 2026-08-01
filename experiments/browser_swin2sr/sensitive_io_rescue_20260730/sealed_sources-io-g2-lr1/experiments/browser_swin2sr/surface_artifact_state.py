"""Dense state materialization for mixed linear and convolutional surfaces."""

from __future__ import annotations

import torch

from rotation_control import block_hadamard


def deployment_surface_state_dict(
    model,
    surfaces,
    *,
    preserve_rotated_basis: bool,
) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".parametrizations.weight." not in name
        and not name.endswith(".amount")
        and not name.endswith(".input_shift")
    }
    for surface in surfaces:
        module = surface.module
        quantizer = surface.quantizer
        quantizer.amount.fill_(1.0)
        weight = module.weight.detach()
        effective = (
            block_hadamard(weight, quantizer.rotation_block)
            if quantizer.rotation == "hadamard" and not preserve_rotated_basis
            else weight
        )
        state[f"{surface.name}.weight"] = effective.cpu()
        if module.bias is not None:
            bias = module.bias.detach()
            if hasattr(module, "input_shift"):
                bias = torch.mv(weight, module.input_shift.detach()) + bias
            state[f"{surface.name}.bias"] = bias.cpu()
    return state


def hard_surface_state_dict(model, surfaces) -> dict[str, torch.Tensor]:
    return deployment_surface_state_dict(
        model,
        surfaces,
        preserve_rotated_basis=False,
    )


def runtime_surface_state_dict(model, surfaces) -> dict[str, torch.Tensor]:
    return deployment_surface_state_dict(
        model,
        surfaces,
        preserve_rotated_basis=True,
    )
