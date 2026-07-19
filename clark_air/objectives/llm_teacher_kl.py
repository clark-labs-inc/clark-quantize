"""LLM teacher KL objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def teacher_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL[p_teacher || p_student] over token logits."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_logp = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    t_prob = t_logp.exp()
    per_token = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=-1) * (temperature**2)
    if mask is None:
        return per_token.mean()
    weights = mask.to(per_token.device, dtype=per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)
