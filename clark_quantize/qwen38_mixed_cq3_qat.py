"""Calibration-only CQ3 row-shard training for the Qwen3.8-27B mixed pilot.

This module deliberately has no confirmation/validation input.  It selects a
hard packed state from step zero through the frozen final step using only the
preregistered calibration activations.  Fresh confirmation is a later,
separate complete-block operation.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from .qwen38_layer0_qat import (
    CQ3_BLOCK_WEIGHTS,
    CQ3_GROUP_WEIGHTS,
    RowShardTrainingConfig,
    normalized_squared_error,
)
from .qwen_qat import hard_ternary_ste
from .ternary import ternary_codes_scales


def _hard_codes(master: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ternary_codes_scales(
        master.detach().float(), group=CQ3_GROUP_WEIGHTS, method="optimal"
    )


def _hard_weight(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (
        codes.reshape(codes.shape[0], -1, CQ3_GROUP_WEIGHTS).float()
        * scales.float().unsqueeze(-1)
    ).reshape_as(codes)


def _squared_error_stats(
    candidate: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    difference = candidate.float() - target.float()
    return float(difference.square().sum()), float(target.float().square().sum())


def train_row_shard_calibration_only(
    source_weight: torch.Tensor,
    calibration_inputs: torch.Tensor,
    *,
    config: RowShardTrainingConfig,
    teacher_calibration_inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return the lowest calibration-loss hard state from step zero..N."""
    if source_weight.ndim != 2 or source_weight.shape[1] % CQ3_BLOCK_WEIGHTS:
        raise ValueError("source row shard must be a CQ3-compatible matrix")
    if (
        calibration_inputs.ndim != 2
        or calibration_inputs.shape[1] != source_weight.shape[1]
        or calibration_inputs.shape[0] == 0
    ):
        raise ValueError("calibration matrix does not match source columns")
    source = source_weight.detach().float().contiguous()
    train_x = calibration_inputs.detach().float().contiguous()
    teacher_x = (
        train_x
        if teacher_calibration_inputs is None
        else teacher_calibration_inputs.detach().float().contiguous()
    )
    if teacher_x.shape != train_x.shape:
        raise ValueError("teacher and candidate calibration shapes must match")

    with torch.no_grad():
        target = functional.linear(teacher_x, source)
        baseline_codes, baseline_scales = _hard_codes(source)
        baseline_weight = _hard_weight(baseline_codes, baseline_scales)
        baseline_output = functional.linear(train_x, baseline_weight)
        baseline_nmse = float(normalized_squared_error(baseline_output, target))
        baseline_sse, target_energy = _squared_error_stats(baseline_output, target)
    if not all(torch.isfinite(value).all() for value in (target, baseline_weight)):
        raise ValueError("non-finite calibration teacher or step-zero CQ3 state")

    best_step = 0
    best_nmse = baseline_nmse
    best_sse = baseline_sse
    best_codes = baseline_codes.contiguous().clone()
    best_scales = baseline_scales.contiguous().clone()
    best_churn = 0.0
    master = torch.nn.Parameter(source.clone())
    optimizer = torch.optim.AdamW(
        [master], lr=config.learning_rate, weight_decay=0.0
    )
    history: list[dict[str, float | int | bool]] = []
    finite = True
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        hard = hard_ternary_ste(
            master, group_size=config.group_size, row_batch=master.shape[0]
        )
        activation_loss = normalized_squared_error(
            functional.linear(train_x, hard), target
        )
        weight_loss = normalized_squared_error(hard, source)
        loss = activation_loss + config.weight_reconstruction_weight * weight_loss
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [master], config.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm)):
            finite = False
            break
        optimizer.step()
        with torch.no_grad():
            codes, scales = _hard_codes(master)
            packed_weight = _hard_weight(codes, scales)
            output = functional.linear(train_x, packed_weight)
            hard_nmse = float(normalized_squared_error(output, target))
            hard_sse, checked_energy = _squared_error_stats(output, target)
            if checked_energy != target_energy:
                raise AssertionError("calibration target energy changed within one shard")
            churn = float((codes != baseline_codes).float().mean())
            is_best = bool(torch.isfinite(output).all()) and hard_nmse < best_nmse
            if is_best:
                best_step = step
                best_nmse = hard_nmse
                best_sse = hard_sse
                best_codes = codes.contiguous().clone()
                best_scales = scales.contiguous().clone()
                best_churn = churn
        history.append(
            {
                "step": step,
                "surrogate_loss": float(loss.detach()),
                "surrogate_activation_nmse": float(activation_loss.detach()),
                "surrogate_weight_nmse": float(weight_loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm.detach()),
                "hard_packed_calibration_nmse": hard_nmse,
                "code_churn_fraction_vs_step_zero": churn,
                "selected_as_best": is_best,
            }
        )

    metrics: dict[str, Any] = {
        "rows": source.shape[0],
        "columns": source.shape[1],
        "steps_requested": config.steps,
        "steps_completed": len(history),
        "finite": finite,
        "selection_data": "calibration_only",
        "step_zero_calibration_nmse": baseline_nmse,
        "step_zero_calibration_squared_error_sum": baseline_sse,
        "calibration_target_squared_sum": target_energy,
        "selected_step": best_step,
        "selected_calibration_nmse": best_nmse,
        "selected_calibration_squared_error_sum": best_sse,
        "relative_calibration_nmse_improvement": (
            (baseline_nmse - best_nmse) / max(baseline_nmse, 1e-12)
        ),
        "selected_code_churn_fraction": best_churn,
        "accepted": finite and best_step > 0,
        "distinct_teacher_inputs": teacher_calibration_inputs is not None,
        "history": history,
    }
    return best_codes, best_scales, metrics


def aggregate_calibration_metrics(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one CQ3 shard metric is required")
    rows = sum(int(item["rows"]) for item in shards)
    target_energy = sum(float(item["calibration_target_squared_sum"]) for item in shards)
    step_zero_error = sum(
        float(item["step_zero_calibration_squared_error_sum"]) for item in shards
    )
    selected_error = sum(
        float(item["selected_calibration_squared_error_sum"]) for item in shards
    )
    if rows <= 0 or target_energy <= 0:
        raise ValueError("invalid aggregate calibration denominator")
    step_zero_nmse = step_zero_error / target_energy
    selected_nmse = selected_error / target_energy
    return {
        "rows": rows,
        "shards": len(shards),
        "accepted_shards": sum(bool(item["accepted"]) for item in shards),
        "all_finite": all(bool(item["finite"]) for item in shards),
        "step_zero_calibration_nmse": step_zero_nmse,
        "selected_calibration_nmse": selected_nmse,
        "step_zero_calibration_squared_error_sum": step_zero_error,
        "selected_calibration_squared_error_sum": selected_error,
        "calibration_target_squared_sum": target_energy,
        "relative_calibration_nmse_improvement": (
            (step_zero_nmse - selected_nmse) / max(step_zero_nmse, 1e-12)
        ),
        "weighted_selected_code_churn_fraction": sum(
            float(item["selected_code_churn_fraction"]) * int(item["rows"])
            for item in shards
        )
        / rows,
        "selection_data": "calibration_only",
    }
