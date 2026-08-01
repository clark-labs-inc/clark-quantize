"""Train only the four already-deployed RSTB fusion biases."""

from __future__ import annotations

from pathlib import Path

import torch

from surface_model_state import pretrained


BIAS_NAMES = tuple(
    f"swin2sr.encoder.stages.{index}.conv.bias"
    for index in range(4)
)


def load_bias_student(
    state_path: Path,
    device: str,
) -> tuple[torch.nn.Module, list[torch.nn.Parameter], dict[str, torch.Tensor]]:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    student = pretrained(device)
    student.load_state_dict(state, strict=True)
    student.requires_grad_(False)
    named = dict(student.named_parameters())
    parameters = []
    origins = {}
    for name in BIAS_NAMES:
        parameter = named[name]
        parameter.requires_grad_(True)
        parameters.append(parameter)
        origins[name] = parameter.detach().clone()
    return student, parameters, origins


def clamp_bias_deltas(
    student: torch.nn.Module,
    origins: dict[str, torch.Tensor],
    maximum: float = 0.05,
) -> None:
    named = dict(student.named_parameters())
    with torch.no_grad():
        for name, origin in origins.items():
            named[name].copy_((named[name] - origin).clamp(-maximum, maximum) + origin)


def bias_delta_receipt(
    student: torch.nn.Module,
    origins: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    named = dict(student.named_parameters())
    deltas = torch.cat([
        (named[name].detach() - origin).reshape(-1)
        for name, origin in origins.items()
    ])
    fp16 = torch.cat([
        named[name].detach().to(torch.float16).to(torch.float32).reshape(-1)
        for name in origins
    ])
    values = torch.cat([named[name].detach().reshape(-1) for name in origins])
    return {
        "parameters": int(deltas.numel()),
        "mean_delta": float(deltas.mean()),
        "max_abs_delta": float(deltas.abs().max()),
        "fp16_roundtrip_max_error": float((fp16 - values).abs().max()),
    }
