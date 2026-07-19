"""Diffusion transformer teacher-velocity objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def velocity_loss(student_velocity: torch.Tensor, teacher_velocity: torch.Tensor) -> torch.Tensor:
    """MSE teacher-velocity loss."""

    return F.mse_loss(student_velocity.float(), teacher_velocity.detach().float())


def symmetric_velocity_loss(
    student_on_teacher_states: torch.Tensor,
    teacher_on_teacher_states: torch.Tensor,
    student_on_student_states: torch.Tensor,
    teacher_on_student_states: torch.Tensor,
    *,
    teacher_weight: float = 0.5,
    student_weight: float = 0.5,
) -> torch.Tensor:
    """Symmetric teacher-state/student-state path loss."""

    lt = velocity_loss(student_on_teacher_states, teacher_on_teacher_states)
    ls = velocity_loss(student_on_student_states, teacher_on_student_states)
    return teacher_weight * lt + student_weight * ls
