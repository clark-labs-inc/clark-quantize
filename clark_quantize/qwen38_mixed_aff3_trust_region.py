"""Calibration-only hard-forward trust region for mixed Qwen3.8 AFF3."""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import torch


LossClosure = Callable[[], torch.Tensor]
Projection = Callable[[], None]


def _checked_loss(loss_closure: LossClosure) -> torch.Tensor:
    loss = loss_closure()
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise ValueError("AFF3 trust-region loss must be one finite scalar")
    return loss


def calibration_only_hard_forward_trust_region(
    parameters: Iterable[torch.nn.Parameter],
    loss_closure: LossClosure,
    project: Projection,
    *,
    outer_steps: int,
    global_l2_radius: float,
    backtracking_factors: tuple[float, ...],
) -> dict[str, Any]:
    """Run the frozen normalized-gradient, first-nonregressing schedule.

    ``loss_closure`` is responsible for the deployable hard forward.  No
    validation or confirmation input is accepted by this API.
    """
    parameters = list(parameters)
    if (
        not parameters
        or len({id(parameter) for parameter in parameters}) != len(parameters)
        or any(not parameter.requires_grad for parameter in parameters)
        or outer_steps <= 0
        or not math.isfinite(global_l2_radius)
        or global_l2_radius <= 0
        or not backtracking_factors
        or any(
            not math.isfinite(factor) or factor <= 0 or factor > 1
            for factor in backtracking_factors
        )
        or any(
            backtracking_factors[index] >= backtracking_factors[index - 1]
            for index in range(1, len(backtracking_factors))
        )
    ):
        raise ValueError("invalid AFF3 trust-region configuration")

    with torch.no_grad():
        initial_loss = float(_checked_loss(loss_closure))
    accepted_loss = initial_loss
    history: list[dict[str, Any]] = []
    for step in range(1, outer_steps + 1):
        for parameter in parameters:
            parameter.grad = None
        training_loss = _checked_loss(loss_closure)
        training_loss.backward()
        if any(parameter.grad is None for parameter in parameters):
            raise RuntimeError("AFF3 trainable parameter received no gradient")
        gradients = [parameter.grad.detach().float().clone() for parameter in parameters]
        gradient_squared_sum = sum(
            float(gradient.square().sum()) for gradient in gradients
        )
        gradient_norm = math.sqrt(gradient_squared_sum)
        if not math.isfinite(gradient_norm):
            raise RuntimeError("AFF3 global gradient norm is non-finite")
        snapshots = [parameter.detach().clone() for parameter in parameters]
        proposal_records: list[dict[str, Any]] = []
        accepted_factor: float | None = None
        if gradient_norm > 0:
            for factor in backtracking_factors:
                with torch.no_grad():
                    for parameter, snapshot, gradient in zip(
                        parameters, snapshots, gradients
                    ):
                        parameter.copy_(
                            snapshot
                            - (global_l2_radius * factor / gradient_norm)
                            * gradient.to(snapshot.dtype)
                        )
                    project()
                    proposal = loss_closure()
                    proposal_finite = bool(torch.isfinite(proposal))
                    proposal_loss = float(proposal) if proposal_finite else None
                accepted = bool(
                    proposal_finite
                    and proposal_loss is not None
                    and proposal_loss <= accepted_loss
                )
                proposal_records.append(
                    {
                        "factor": factor,
                        "hard_forward_calibration_nmse": proposal_loss,
                        "finite": proposal_finite,
                        "accepted": accepted,
                    }
                )
                if accepted:
                    accepted_factor = factor
                    accepted_loss = float(proposal_loss)
                    break
            if accepted_factor is None:
                with torch.no_grad():
                    for parameter, snapshot in zip(parameters, snapshots):
                        parameter.copy_(snapshot)
                    project()
        history.append(
            {
                "outer_step": step,
                "prior_accepted_hard_forward_calibration_nmse": float(
                    training_loss.detach()
                ),
                "global_gradient_l2_norm": gradient_norm,
                "global_l2_radius": global_l2_radius,
                "proposals": proposal_records,
                "accepted_factor": accepted_factor,
                "accepted_hard_forward_calibration_nmse": accepted_loss,
            }
        )
    for parameter in parameters:
        parameter.grad = None
    with torch.no_grad():
        final_loss = float(_checked_loss(loss_closure))
    if final_loss != accepted_loss:
        raise AssertionError("AFF3 final hard-forward state changed after acceptance")
    return {
        "selection_data": "calibration_only",
        "outer_steps_requested": outer_steps,
        "outer_steps_completed": len(history),
        "initial_hard_forward_calibration_nmse": initial_loss,
        "final_hard_forward_calibration_nmse": final_loss,
        "relative_calibration_nmse_improvement": (
            (initial_loss - final_loss) / max(initial_loss, 1e-30)
        ),
        "accepted_steps": sum(item["accepted_factor"] is not None for item in history),
        "all_values_finite": all(
            math.isfinite(item["global_gradient_l2_norm"])
            and math.isfinite(item["accepted_hard_forward_calibration_nmse"])
            for item in history
        ),
        "history": history,
    }
