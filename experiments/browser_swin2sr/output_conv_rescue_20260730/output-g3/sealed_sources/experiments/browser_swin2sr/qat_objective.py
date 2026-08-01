"""Distillation objective for low-bit Swin2SR training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def charbonnier(
    actual: torch.Tensor,
    expected: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    return torch.sqrt((actual - expected).square() + epsilon**2).mean()


def gradient_loss(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual_dx = actual[:, :, :, 1:] - actual[:, :, :, :-1]
    expected_dx = expected[:, :, :, 1:] - expected[:, :, :, :-1]
    actual_dy = actual[:, :, 1:, :] - actual[:, :, :-1, :]
    expected_dy = expected[:, :, 1:, :] - expected[:, :, :-1, :]
    return charbonnier(actual_dx, expected_dx) + charbonnier(actual_dy, expected_dy)


def feature_loss(student_states, teacher_states) -> torch.Tensor:
    losses = []
    for student, teacher in zip(student_states[1:], teacher_states[1:]):
        student = F.layer_norm(student, (student.shape[-1],))
        teacher = F.layer_norm(teacher, (teacher.shape[-1],))
        losses.append(F.smooth_l1_loss(student, teacher))
    return torch.stack(losses).mean()


def psnr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    mse = F.mse_loss(actual.float(), expected.float()).item()
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


def distillation_losses(reconstruction, teacher_reconstruction, high, states):
    student_states, teacher_states = states
    return {
        "teacher": charbonnier(reconstruction, teacher_reconstruction),
        "ground_truth": charbonnier(reconstruction, high),
        "edge": gradient_loss(reconstruction, teacher_reconstruction),
        "feature": feature_loss(student_states, teacher_states),
    }
