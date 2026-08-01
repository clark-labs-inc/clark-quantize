"""Reusable low-state optimizers developed in Clark Quantize QAT experiments.

These implementations were originally embedded in the Qwen all-2D training
script. Keeping them here lets small-model studies compare the algorithms
without importing a model-specific training program.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def _group_normalized_update(
    gradient: torch.Tensor,
    *,
    group_size: int,
    eps: float,
) -> torch.Tensor:
    flat = gradient.detach().float().reshape(-1)
    if group_size <= 0:
        rms = flat.square().mean().sqrt().clamp_min(eps)
        return (flat / rms).reshape_as(gradient)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    view = flat.view(-1, group_size)
    rms = view.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    update = (view / rms).reshape(-1)
    if pad:
        update = update[:-pad]
    return update.reshape_as(gradient)


@torch.no_grad()
def _proximal_pull(parameter: torch.nn.Parameter, rho: float) -> None:
    if rho <= 0:
        return
    make_quantized = getattr(parameter, "_bonsai_make_quantized", None)
    if not callable(make_quantized):
        return
    projected = make_quantized().to(device=parameter.device, dtype=parameter.dtype)
    if projected.shape != parameter.shape:
        raise RuntimeError(
            f"quantized callback returned {projected.shape}, expected {parameter.shape}"
        )
    parameter.lerp_(projected, min(1.0, rho))


class GroupMirrorSGD(torch.optim.Optimizer):
    """Group-normalized mirror-style SGD with optional group AdaGrad state."""

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        group_size: int,
        eps: float = 1e-6,
        prox_rho: float = 0.0,
        max_update_rms: float = 1.0,
        group_adagrad: bool = False,
    ) -> None:
        super().__init__(
            [{"params": params}],
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "group_size": group_size,
                "eps": eps,
                "prox_rho": prox_rho,
                "max_update_rms": max_update_rms,
                "group_adagrad": group_adagrad,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        group_size = int(group["group_size"])
        eps = float(group["eps"])
        prox_rho = float(group["prox_rho"])
        max_update_rms = float(group["max_update_rms"])
        group_adagrad = bool(group["group_adagrad"])
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("GroupMirrorSGD does not support sparse gradients")
            if group_adagrad and parameter.ndim >= 2:
                flat = parameter.grad.detach().float().reshape(-1)
                pad = (-flat.numel()) % group_size
                if pad:
                    flat = F.pad(flat, (0, pad))
                view = flat.view(-1, group_size)
                group_square = view.square().mean(dim=1)
                state = self.state[parameter]
                if "group_accum" not in state:
                    state["group_accum"] = torch.zeros_like(group_square)
                accumulator = state["group_accum"]
                accumulator.add_(group_square)
                update = (
                    view / accumulator.sqrt().clamp_min(eps).view(-1, 1)
                ).reshape(-1)
                if pad:
                    update = update[:-pad]
                update = update.reshape_as(parameter.grad)
            else:
                update = _group_normalized_update(
                    parameter.grad,
                    group_size=group_size if parameter.ndim >= 2 else 0,
                    eps=eps,
                )
            if max_update_rms > 0:
                rms = update.detach().float().square().mean().sqrt()
                if torch.isfinite(rms) and rms > max_update_rms:
                    update.mul_(max_update_rms / rms.clamp_min(eps))
            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr)
            _proximal_pull(parameter, prox_rho)
        return loss


class FactoredAdafactorQAT(torch.optim.Optimizer):
    """Adafactor-style optimizer with factored state for 2D parameters."""

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        beta2: float = 0.999,
        eps: float = 1e-6,
        prox_rho: float = 0.0,
        max_update_rms: float = 1.0,
        update_chunk_rows: int = 1024,
    ) -> None:
        if beta2 < 0 or beta2 >= 1:
            raise ValueError("beta2 must be in [0, 1)")
        super().__init__(
            [{"params": params}],
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "beta2": beta2,
                "eps": eps,
                "prox_rho": prox_rho,
                "max_update_rms": max_update_rms,
                "update_chunk_rows": update_chunk_rows,
            },
        )

    @torch.no_grad()
    def step_param(
        self,
        parameter: torch.nn.Parameter,
        group: dict[str, Any],
    ) -> None:
        if parameter.grad is None:
            return
        if parameter.grad.is_sparse:
            raise RuntimeError("FactoredAdafactorQAT does not support sparse gradients")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        beta2 = float(group["beta2"])
        eps = float(group["eps"])
        prox_rho = float(group["prox_rho"])
        maximum_rms = float(group["max_update_rms"])
        chunk_rows = max(1, int(group["update_chunk_rows"]))
        gradient = parameter.grad.detach().float()
        state = self.state[parameter]
        if gradient.ndim == 2:
            rows, columns = gradient.shape
            row_mean = torch.empty(rows, device=gradient.device)
            column_sum = torch.zeros(columns, device=gradient.device)
            for start in range(0, rows, chunk_rows):
                square = gradient[start : start + chunk_rows].square()
                row_mean[start : start + square.shape[0]] = square.mean(dim=1)
                column_sum.add_(square.sum(dim=0))
            column_mean = column_sum / max(rows, 1)
            if "row_accum" not in state:
                state["row_accum"] = torch.zeros_like(row_mean)
                state["col_accum"] = torch.zeros_like(column_mean)
            state["row_accum"].mul_(beta2).add_(row_mean, alpha=1 - beta2)
            state["col_accum"].mul_(beta2).add_(column_mean, alpha=1 - beta2)
            row_factor = state["row_accum"] / state["row_accum"].mean().clamp_min(eps)
            update_scale = 1.0
            if maximum_rms > 0:
                update_square_sum = 0.0
                for start in range(0, rows, chunk_rows):
                    end = min(start + chunk_rows, rows)
                    variance = (
                        row_factor[start:end, None] * state["col_accum"][None, :]
                    )
                    update = gradient[start:end] / variance.sqrt().clamp_min(eps)
                    update_square_sum += float(update.square().sum().item())
                update_rms = math.sqrt(update_square_sum / max(gradient.numel(), 1))
                if math.isfinite(update_rms) and update_rms > maximum_rms:
                    update_scale = maximum_rms / max(update_rms, eps)
            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            for start in range(0, rows, chunk_rows):
                end = min(start + chunk_rows, rows)
                variance = (
                    row_factor[start:end, None] * state["col_accum"][None, :]
                )
                update = gradient[start:end] / variance.sqrt().clamp_min(eps)
                if update_scale != 1:
                    update.mul_(update_scale)
                parameter[start:end].add_(
                    update.to(dtype=parameter.dtype),
                    alpha=-lr,
                )
            _proximal_pull(parameter, prox_rho)
            return
        else:
            square = gradient.square()
            if "accum" not in state:
                state["accum"] = torch.zeros_like(square)
            state["accum"].mul_(beta2).add_(square, alpha=1 - beta2)
            update = gradient / state["accum"].sqrt().clamp_min(eps)
        if maximum_rms > 0:
            rms = update.detach().float().square().mean().sqrt()
            if torch.isfinite(rms) and rms > maximum_rms:
                update.mul_(maximum_rms / rms.clamp_min(eps))
        if weight_decay:
            parameter.mul_(1.0 - lr * weight_decay)
        parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr)
        _proximal_pull(parameter, prox_rho)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        for parameter in group["params"]:
            self.step_param(parameter, group)
        return loss
