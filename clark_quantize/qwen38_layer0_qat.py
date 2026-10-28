"""Bounded CPU helpers for Qwen3.8-27B layer-0 activation-aware ternary QAT.

The whole 27B graph is deliberately absent from this module.  Training is done
one output-row shard at a time against cached layer-0 activations, and the final
CQ3_0 payload is emitted without retaining optimizer state for completed rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from .qwen_qat import hard_ternary_ste
from .ternary import ternary_codes_scales


CQ3_BLOCK_WEIGHTS = 256
CQ3_GROUP_WEIGHTS = 128
CQ3_CODE_BYTES = 64
CQ3_SCALE_BYTES = 4
CQ3_BLOCK_BYTES = CQ3_CODE_BYTES + CQ3_SCALE_BYTES


def expected_cq3_payload_bytes(rows: int, columns: int) -> int:
    if rows <= 0 or columns <= 0 or columns % CQ3_BLOCK_WEIGHTS:
        raise ValueError("CQ3_0 requires positive rows and columns divisible by 256")
    return rows * (columns // CQ3_BLOCK_WEIGHTS) * CQ3_BLOCK_BYTES


def pack_cq3_rows(codes: torch.Tensor, scales: torch.Tensor) -> bytes:
    """Pack artifact trits/scales into llama.cpp CQ3_0 row-major blocks."""
    if codes.ndim != 2 or scales.ndim != 2:
        raise ValueError("codes and scales must be matrices")
    rows, columns = codes.shape
    if columns % CQ3_BLOCK_WEIGHTS:
        raise ValueError("CQ3_0 columns must be divisible by 256")
    if scales.shape != (rows, columns // CQ3_GROUP_WEIGHTS):
        raise ValueError("scale matrix does not match group-128 code matrix")
    if codes.dtype != torch.int8 or bool(((codes < -1) | (codes > 1)).any()):
        raise ValueError("codes must be int8 trits")

    blocks = columns // CQ3_BLOCK_WEIGHTS
    trits = (codes.detach().cpu().numpy().astype(np.int16) + 1).astype(np.uint8)
    grouped = trits.reshape(rows, blocks, 2, CQ3_GROUP_WEIGHTS)
    packed = np.zeros((rows, blocks, 2, 32), dtype=np.uint8)
    for quarter in range(4):
        packed |= grouped[..., quarter * 32 : (quarter + 1) * 32] << (2 * quarter)
    code_bytes = packed.reshape(rows, blocks, CQ3_CODE_BYTES)
    scale_values = scales.detach().cpu().numpy().astype("<f2", copy=False)
    scale_bytes = scale_values.reshape(rows, blocks, 2).view(np.uint8).reshape(rows, blocks, 4)
    payload = np.concatenate((code_bytes, scale_bytes), axis=-1)
    result = payload.tobytes(order="C")
    if len(result) != expected_cq3_payload_bytes(rows, columns):
        raise AssertionError("internal CQ3_0 payload size mismatch")
    return result


def unpack_cq3_rows(payload: bytes, *, rows: int, columns: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact inverse of :func:`pack_cq3_rows`, used for export authentication."""
    expected = expected_cq3_payload_bytes(rows, columns)
    if len(payload) != expected:
        raise ValueError(f"expected {expected} payload bytes, got {len(payload)}")
    blocks = columns // CQ3_BLOCK_WEIGHTS
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(rows, blocks, CQ3_BLOCK_BYTES)
    packed = raw[..., :CQ3_CODE_BYTES].reshape(rows, blocks, 2, 32)
    grouped = np.empty((rows, blocks, 2, CQ3_GROUP_WEIGHTS), dtype=np.int8)
    for quarter in range(4):
        digits = ((packed >> (2 * quarter)) & 3).astype(np.int8)
        if bool((digits > 2).any()):
            raise ValueError("CQ3_0 payload contains reserved two-bit code 3")
        grouped[..., quarter * 32 : (quarter + 1) * 32] = digits - 1
    scale_bytes = raw[..., CQ3_CODE_BYTES:].copy()
    scales = scale_bytes.reshape(rows, blocks, 4).view("<f2").reshape(rows, blocks * 2)
    return (
        torch.from_numpy(grouped.reshape(rows, columns).copy()),
        torch.from_numpy(scales.copy()),
    )


def normalized_squared_error(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if candidate.shape != target.shape or candidate.numel() == 0:
        raise ValueError("candidate and target must be equal non-empty shapes")
    numerator = functional.mse_loss(candidate.float(), target.float())
    denominator = target.float().square().mean().clamp_min(1e-12)
    return numerator / denominator


def _squared_error_stats(candidate: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    difference = candidate.float() - target.float()
    return float(difference.square().sum()), float(target.float().square().sum())


@dataclass(frozen=True)
class RowShardTrainingConfig:
    steps: int = 6
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    weight_reconstruction_weight: float = 0.01
    group_size: int = CQ3_GROUP_WEIGHTS

    def __post_init__(self) -> None:
        if (
            self.steps <= 0
            or self.learning_rate <= 0
            or self.gradient_clip_norm <= 0
            or self.weight_reconstruction_weight < 0
            or self.group_size != CQ3_GROUP_WEIGHTS
        ):
            raise ValueError("invalid Qwen3.8-27B row-shard training configuration")


def _hard_codes(master: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return ternary_codes_scales(master.detach().float(), group=group_size, method="optimal")


def train_row_shard(
    source_weight: torch.Tensor,
    calibration_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    *,
    config: RowShardTrainingConfig,
    teacher_calibration_inputs: torch.Tensor | None = None,
    teacher_validation_inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Train one output-row shard and retain it only on held-out improvement.

    The BF16/FP32 source row shard is the local teacher.  Calibration and
    validation inputs must come from disjoint preregistered text sequences.
    Returned codes/scales always describe the hard forward, never latent
    master weights.
    """
    if source_weight.ndim != 2 or source_weight.shape[1] % CQ3_BLOCK_WEIGHTS:
        raise ValueError("source row shard must be a CQ3-compatible matrix")
    if (
        calibration_inputs.ndim != 2
        or validation_inputs.ndim != 2
        or calibration_inputs.shape[1] != source_weight.shape[1]
        or validation_inputs.shape[1] != source_weight.shape[1]
        or calibration_inputs.shape[0] == 0
        or validation_inputs.shape[0] == 0
    ):
        raise ValueError("activation matrices do not match source columns")

    source = source_weight.detach().float().contiguous()
    train_x = calibration_inputs.detach().float().contiguous()
    valid_x = validation_inputs.detach().float().contiguous()
    teacher_train_x = (
        train_x
        if teacher_calibration_inputs is None
        else teacher_calibration_inputs.detach().float().contiguous()
    )
    teacher_valid_x = (
        valid_x
        if teacher_validation_inputs is None
        else teacher_validation_inputs.detach().float().contiguous()
    )
    if (
        teacher_train_x.shape != train_x.shape
        or teacher_valid_x.shape != valid_x.shape
        or teacher_train_x.shape[1] != source.shape[1]
    ):
        raise ValueError("teacher and candidate activation shapes must match")
    with torch.no_grad():
        train_target = functional.linear(teacher_train_x, source)
        valid_target = functional.linear(teacher_valid_x, source)
        baseline_codes, baseline_scales = _hard_codes(source, config.group_size)
        baseline_hard = (
            baseline_codes.reshape(source.shape[0], -1, config.group_size).float()
            * baseline_scales.float().unsqueeze(-1)
        ).reshape_as(source)
        baseline_train_output = functional.linear(train_x, baseline_hard)
        baseline_validation_output = functional.linear(valid_x, baseline_hard)
        initial_train = float(normalized_squared_error(baseline_train_output, train_target))
        initial_validation = float(normalized_squared_error(baseline_validation_output, valid_target))
        initial_train_sse, train_target_energy = _squared_error_stats(
            baseline_train_output, train_target
        )
        initial_validation_sse, validation_target_energy = _squared_error_stats(
            baseline_validation_output, valid_target
        )

    master = torch.nn.Parameter(source.clone())
    optimizer = torch.optim.AdamW([master], lr=config.learning_rate, weight_decay=0.0)
    history: list[dict[str, float | int]] = []
    finite = True
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        hard = hard_ternary_ste(master, group_size=config.group_size, row_batch=master.shape[0])
        output_loss = normalized_squared_error(functional.linear(train_x, hard), train_target)
        weight_loss = normalized_squared_error(hard, source)
        loss = output_loss + config.weight_reconstruction_weight * weight_loss
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([master], config.gradient_clip_norm)
        if not bool(torch.isfinite(gradient_norm)):
            finite = False
            break
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "activation_nmse": float(output_loss.detach()),
                "weight_nmse": float(weight_loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm.detach()),
            }
        )

    with torch.no_grad():
        candidate_codes, candidate_scales = _hard_codes(master, config.group_size)
        candidate_hard = (
            candidate_codes.reshape(source.shape[0], -1, config.group_size).float()
            * candidate_scales.float().unsqueeze(-1)
        ).reshape_as(source)
        final_train = float(normalized_squared_error(functional.linear(train_x, candidate_hard), train_target))
        candidate_validation_output = functional.linear(valid_x, candidate_hard)
        final_validation = float(normalized_squared_error(candidate_validation_output, valid_target))
        final_validation_sse, checked_validation_target_energy = _squared_error_stats(
            candidate_validation_output, valid_target
        )
        if checked_validation_target_energy != validation_target_energy:
            raise AssertionError("validation target energy changed within one row shard")
        churn = float((candidate_codes != baseline_codes).float().mean())

    accepted = finite and final_train < initial_train and final_validation < initial_validation
    selected_codes = candidate_codes if accepted else baseline_codes
    selected_scales = candidate_scales if accepted else baseline_scales
    metrics: dict[str, Any] = {
        "rows": source.shape[0],
        "columns": source.shape[1],
        "steps_completed": len(history),
        "finite": finite,
        "accepted": accepted,
        "initial_train_nmse": initial_train,
        "initial_train_squared_error_sum": initial_train_sse,
        "train_target_squared_sum": train_target_energy,
        "final_train_nmse": final_train,
        "relative_train_improvement": (initial_train - final_train) / max(initial_train, 1e-12),
        "initial_validation_nmse": initial_validation,
        "initial_validation_squared_error_sum": initial_validation_sse,
        "final_validation_squared_error_sum": final_validation_sse,
        "validation_target_squared_sum": validation_target_energy,
        "final_validation_nmse": final_validation,
        "relative_validation_improvement": (initial_validation - final_validation) / max(initial_validation, 1e-12),
        "candidate_code_churn_fraction": churn,
        "selected_code_churn_fraction": churn if accepted else 0.0,
        "distinct_teacher_inputs": (
            teacher_calibration_inputs is not None or teacher_validation_inputs is not None
        ),
        "history": history,
    }
    return selected_codes.contiguous(), selected_scales.contiguous(), metrics


def aggregate_shard_metrics(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one shard metric is required")
    rows = sum(int(item["rows"]) for item in shards)
    if rows <= 0:
        raise ValueError("invalid aggregate row count")
    weighted = lambda key: sum(float(item[key]) * int(item["rows"]) for item in shards) / rows
    target_energy = sum(float(item["validation_target_squared_sum"]) for item in shards)
    initial_error = sum(float(item["initial_validation_squared_error_sum"]) for item in shards)
    selected_error = sum(
        float(
            item["final_validation_squared_error_sum"]
            if item["accepted"]
            else item["initial_validation_squared_error_sum"]
        )
        for item in shards
    )
    if target_energy <= 0:
        raise ValueError("aggregate validation target energy must be positive")
    initial = initial_error / target_energy
    final_selected = selected_error / target_energy
    return {
        "rows": rows,
        "shards": len(shards),
        "accepted_shards": sum(bool(item["accepted"]) for item in shards),
        "all_finite": all(bool(item["finite"]) for item in shards),
        "weighted_initial_validation_nmse": initial,
        "weighted_selected_validation_nmse": final_selected,
        "initial_validation_squared_error_sum": initial_error,
        "selected_validation_squared_error_sum": selected_error,
        "validation_target_squared_sum": target_energy,
        "relative_validation_improvement": (initial - final_selected) / max(initial, 1e-12),
        "weighted_selected_code_churn_fraction": weighted("selected_code_churn_fraction"),
    }
