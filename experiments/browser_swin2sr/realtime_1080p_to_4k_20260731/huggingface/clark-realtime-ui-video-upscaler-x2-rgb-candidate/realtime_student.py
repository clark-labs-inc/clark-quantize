"""Training and export contract for the BackdropKit realtime x2 video student."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RealtimeStudentConfig:
    channels: int = 12
    blocks: int = 3
    residual_scale: float = 1.0
    color_mode: str = "luma"

    def validate(self) -> None:
        if self.channels < 4:
            raise ValueError("channels must be at least four")
        if self.blocks < 1:
            raise ValueError("blocks must be positive")
        if self.color_mode not in {"luma", "rgb"}:
            raise ValueError("color_mode must be luma or rgb")


class ResidualBlock(nn.Module):
    """Two bias-free convolutions matching one fused WGSL dispatch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.second = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.first(values))
        return F.relu(self.second(hidden) + values)


class RealtimeLumaStudent(nn.Module):
    """A roughly 8K-parameter luma residual model for browser 1080p to 4K."""

    def __init__(self, config: RealtimeStudentConfig | None = None) -> None:
        super().__init__()
        self.config = config or RealtimeStudentConfig()
        self.config.validate()
        channels = self.config.channels
        input_channels = 4 if self.config.color_mode == "luma" else 12
        output_channels = 16 if self.config.color_mode == "luma" else 48
        self.input_projection = nn.Conv2d(input_channels, channels, 1, bias=False)
        self.blocks = nn.ModuleList(
            ResidualBlock(channels) for _ in range(self.config.blocks)
        )
        self.output_projection = nn.Conv2d(channels, output_channels, 1, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    @staticmethod
    def luma(values: torch.Tensor) -> torch.Tensor:
        coefficients = values.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
        return (values * coefficients).sum(dim=1, keepdim=True)

    def residual(self, low_resolution: torch.Tensor) -> torch.Tensor:
        if low_resolution.ndim != 4 or low_resolution.shape[1] != 3:
            raise ValueError("expected NCHW RGB input")
        if low_resolution.shape[-2] % 2 or low_resolution.shape[-1] % 2:
            raise ValueError("input height and width must be even")
        inputs = self.luma(low_resolution) if self.config.color_mode == "luma" else low_resolution
        features = F.pixel_unshuffle(inputs, 2)
        features = F.relu(self.input_projection(features))
        for block in self.blocks:
            features = block(features)
        return F.pixel_shuffle(self.output_projection(features), 4)

    def forward(self, low_resolution: torch.Tensor) -> torch.Tensor:
        baseline = F.interpolate(
            low_resolution,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        correction = self.residual(low_resolution) * self.config.residual_scale
        return (baseline + correction).clamp(0.0, 1.0)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def wgsl_weight_arrays(model: RealtimeLumaStudent) -> dict[str, np.ndarray]:
    """Return FP16 arrays in the exact storage order used by the fused shaders."""

    arrays: dict[str, np.ndarray] = {}
    arrays["input_projection"] = (
        model.input_projection.weight.detach().cpu().numpy().reshape(
            model.config.channels,
            model.input_projection.in_channels,
        ).astype(np.float16)
    )
    for index, block in enumerate(model.blocks):
        first = block.first.weight.detach().cpu().permute(0, 2, 3, 1).numpy()
        second = block.second.weight.detach().cpu().permute(0, 2, 3, 1).numpy()
        arrays[f"block_{index}"] = np.concatenate(
            (first.reshape(-1), second.reshape(-1))
        ).astype(np.float16)
    arrays["output_projection"] = (
        model.output_projection.weight.detach().cpu().numpy().reshape(
            model.output_projection.out_channels,
            model.config.channels,
        ).astype(np.float16)
    )
    return arrays


def export_wgsl_weights(model: RealtimeLumaStudent, output: Path) -> dict:
    """Write the tiny FP16 payload and a stable offset ledger."""

    arrays = wgsl_weight_arrays(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    ledger = {}
    with output.open("wb") as destination:
        for name, values in arrays.items():
            payload = values.tobytes(order="C")
            destination.write(payload)
            ledger[name] = {
                "offset": offset,
                "bytes": len(payload),
                "elements": values.size,
            }
            offset += len(payload)
    return {
        "format": "backdropkit-realtime-luma-student-f16-v1",
        "channels": model.config.channels,
        "blocks": model.config.blocks,
        "residual_scale": model.config.residual_scale,
        "color_mode": model.config.color_mode,
        "parameters": model.parameter_count(),
        "bytes": offset,
        "tensors": ledger,
    }


def initialize_rgb_from_luma(
    rgb: RealtimeLumaStudent,
    luma_state: dict[str, torch.Tensor],
) -> None:
    """Make an RGB student initially reproduce a trained luma student exactly."""

    if rgb.config.color_mode != "rgb":
        raise ValueError("target student must use RGB residuals")
    rgb_state = rgb.state_dict()
    for name, value in luma_state.items():
        if name.startswith("blocks."):
            rgb_state[name].copy_(value)
    coefficients = luma_state["input_projection.weight"].new_tensor(
        (0.2126, 0.7152, 0.0722)
    )
    source_input = luma_state["input_projection.weight"]
    for channel, coefficient in enumerate(coefficients):
        start = channel * 4
        rgb_state["input_projection.weight"][:, start:start + 4].copy_(
            source_input * coefficient
        )
    source_output = luma_state["output_projection.weight"]
    for channel in range(3):
        start = channel * 16
        rgb_state["output_projection.weight"][start:start + 16].copy_(source_output)
    rgb.load_state_dict(rgb_state, strict=True)
