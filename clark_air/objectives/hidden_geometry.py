"""Hidden-state geometry preservation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError("hidden states must have shape [batch, tokens, channels] or richer")
    return x.float().reshape(x.shape[0], -1, x.shape[-1])


def relational_gram_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    """Match normalized token-token Gram geometry."""

    s = F.normalize(_flatten_tokens(student_hidden), dim=-1)
    t = F.normalize(_flatten_tokens(teacher_hidden).detach(), dim=-1)
    gs = s @ s.transpose(-1, -2)
    gt = t @ t.transpose(-1, -2)
    return (gs - gt).square().mean()


def centered_cka_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return 1 - linear CKA between student and teacher token features."""

    s = _flatten_tokens(student_hidden).reshape(-1, student_hidden.shape[-1])
    t = _flatten_tokens(teacher_hidden).detach().reshape(-1, teacher_hidden.shape[-1])
    s = s - s.mean(dim=0, keepdim=True)
    t = t - t.mean(dim=0, keepdim=True)
    hsic = (s.transpose(0, 1) @ t).square().sum()
    s_norm = (s.transpose(0, 1) @ s).square().sum().sqrt()
    t_norm = (t.transpose(0, 1) @ t).square().sum().sqrt()
    return 1.0 - hsic / (s_norm * t_norm + eps)
