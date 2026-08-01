"""Distillation objective for low-bit Swin2SR training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


PIXEL_OBJECTIVES = ("charbonnier", "psnr_aligned")


def charbonnier(
    actual: torch.Tensor,
    expected: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    return torch.sqrt((actual - expected).square() + epsilon**2).mean()


def stabilized_log_mse(
    actual: torch.Tensor,
    expected: torch.Tensor,
    mse_floor: float,
) -> torch.Tensor:
    """Approximate mean per-sample PSNR without unstable easy-sample gradients.

    The floor is the frozen target error scale. Multiplying by
    sqrt(floor) / log(2) keeps the loss on the same order as an RMSE-like
    objective and exactly matches sqrt(floor) when MSE equals the floor.
    """

    if mse_floor <= 0:
        raise ValueError("mse_floor must be positive")
    squared_error = (actual.float() - expected.float()).square()
    per_sample_mse = squared_error.flatten(1).mean(dim=1)
    scale = math.sqrt(mse_floor) / math.log(2.0)
    return scale * torch.log1p(per_sample_mse / mse_floor).mean()


def pixel_loss(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    objective: str,
    mse_floor: float,
) -> torch.Tensor:
    if objective == "charbonnier":
        return charbonnier(actual, expected)
    if objective == "psnr_aligned":
        return stabilized_log_mse(actual, expected, mse_floor)
    raise ValueError(f"unknown pixel objective: {objective}")


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


def distillation_losses(
    reconstruction,
    teacher_reconstruction,
    high,
    states,
    *,
    pixel_objective: str,
    ground_truth_mse_floor: float,
    teacher_mse_floor: float,
):
    student_states, teacher_states = states
    return {
        "teacher": pixel_loss(
            reconstruction,
            teacher_reconstruction,
            objective=pixel_objective,
            mse_floor=teacher_mse_floor,
        ),
        "ground_truth": pixel_loss(
            reconstruction,
            high,
            objective=pixel_objective,
            mse_floor=ground_truth_mse_floor,
        ),
        "edge": gradient_loss(reconstruction, teacher_reconstruction),
        "feature": feature_loss(student_states, teacher_states),
    }
