"""Core algebra for the Qwen3.8-27B affine-base3 butterfly experiment.

The representation has exactly three values per group::

    weight[j] = mu[group] + alpha[group] * trit[j]

and an optional shared orthogonal activation transform represented as a product
of sparse Givens stages.  These helpers deliberately contain no sampled-model
or synthetic promotion gate; they are the exact primitives consumed by the
complete-block experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from .ternary import ternary_codes_scales


@dataclass(frozen=True)
class AffineTernary:
    codes: torch.Tensor
    mu: torch.Tensor
    alpha: torch.Tensor
    group_size: int

    def reconstruct(self) -> torch.Tensor:
        if self.codes.ndim != 2:
            raise ValueError("affine ternary codes must be a matrix")
        rows, columns = self.codes.shape
        groups = (columns + self.group_size - 1) // self.group_size
        if self.mu.shape != (rows, groups) or self.alpha.shape != (rows, groups):
            raise ValueError("affine ternary parameter dimensions changed")
        pad = groups * self.group_size - columns
        codes = self.codes.float()
        if pad:
            codes = torch.nn.functional.pad(codes, (0, pad))
        rebuilt = self.mu.unsqueeze(-1) + self.alpha.unsqueeze(-1) * codes.reshape(
            rows, groups, self.group_size
        )
        return rebuilt.reshape(rows, -1)[:, :columns].contiguous()


@dataclass(frozen=True)
class SharedAffineRotationFit:
    angles: torch.Tensor
    pairings: torch.Tensor
    quantized: dict[str, AffineTernary]
    metrics: dict[str, Any]


def deterministic_givens_pairings(
    dimension: int, *, stages: int, seed: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Generate reproducible perfect matchings; the matchings need no storage."""
    if dimension <= 0 or dimension % 2 or stages <= 0 or seed < 0:
        raise ValueError("Givens dimensions, stages, and seed are invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pairings = torch.stack(
        [torch.randperm(dimension, generator=generator).reshape(-1, 2) for _ in range(stages)]
    )
    return pairings.to(device=device)


def apply_givens_product(
    values: torch.Tensor,
    angles: torch.Tensor,
    pairings: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply a product of disjoint-pair Givens stages on the last dimension."""
    if values.ndim == 0 or angles.ndim != 2 or pairings.ndim != 3:
        raise ValueError("invalid Givens operand ranks")
    stages, pairs = angles.shape
    dimension = values.shape[-1]
    if pairings.shape != (stages, pairs, 2) or pairs * 2 != dimension:
        raise ValueError("Givens angles and pairings do not cover the activation")
    if pairings.device != values.device or angles.device != values.device:
        raise ValueError("Givens values, angles, and pairings must share a device")
    expected = torch.arange(dimension, device=values.device)
    for stage in pairings:
        if not torch.equal(torch.sort(stage.reshape(-1)).values, expected):
            raise ValueError("each Givens stage must be a perfect matching")

    result = values
    order = range(stages - 1, -1, -1) if inverse else range(stages)
    for index in order:
        first = pairings[index, :, 0]
        second = pairings[index, :, 1]
        theta = -angles[index] if inverse else angles[index]
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        left = result.index_select(-1, first)
        right = result.index_select(-1, second)
        stage_result = torch.empty_like(result)
        stage_result.index_copy_(-1, first, cosine * left - sine * right)
        stage_result.index_copy_(-1, second, sine * left + cosine * right)
        result = stage_result
    return result


def _solve_affine(
    groups: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = mask.to(groups.dtype)
    count = valid.sum(-1)
    trits = codes.to(groups.dtype) * valid
    sum_t = trits.sum(-1)
    sum_t2 = trits.square().sum(-1)
    sum_w = (groups * valid).sum(-1)
    sum_tw = (groups * trits).sum(-1)
    determinant = count * sum_t2 - sum_t.square()
    safe = determinant.abs() > 1e-12
    mu = torch.where(
        safe,
        (sum_w * sum_t2 - sum_tw * sum_t) / determinant.clamp_min(1e-12),
        sum_w / count.clamp_min(1),
    )
    alpha = torch.where(
        safe,
        (count * sum_tw - sum_t * sum_w) / determinant.clamp_min(1e-12),
        torch.zeros_like(sum_w),
    )
    negative = alpha < 0
    alpha = alpha.abs()
    codes = torch.where(negative.unsqueeze(-1), -codes, codes)
    return mu, alpha, codes


@torch.no_grad()
def initialize_affine_ternary(
    weight: torch.Tensor, *, group_size: int = 128, iterations: int = 4
) -> AffineTernary:
    """Fit a deterministic Euclidean initializer that cannot lose to direct ternary.

    Activation-metric relocation is performed by the complete-block runner.  This
    initializer supplies its exact three-state starting point and falls back per
    group to the frozen optimal symmetric solution if Lloyd relocation is worse.
    """
    if weight.ndim != 2 or not weight.is_floating_point() or group_size <= 0 or iterations <= 0:
        raise ValueError("affine ternary initialization expects a floating matrix")
    rows, columns = weight.shape
    if rows <= 0 or columns <= 0:
        raise ValueError("affine ternary initialization expects non-empty weights")
    source = weight.float().contiguous()
    groups = (columns + group_size - 1) // group_size
    pad = groups * group_size - columns
    padded = torch.nn.functional.pad(source, (0, pad)) if pad else source
    grouped = padded.reshape(rows, groups, group_size)
    mask = torch.ones_like(grouped, dtype=torch.bool)
    if pad:
        mask[:, -1, -pad:] = False

    direct_codes, direct_scales = ternary_codes_scales(source, group=group_size, method="optimal")
    padded_direct = torch.nn.functional.pad(direct_codes, (0, pad)) if pad else direct_codes
    direct_grouped = padded_direct.reshape(rows, groups, group_size)
    codes = direct_grouped.clone()

    for _ in range(iterations):
        mu, alpha, codes = _solve_affine(grouped, codes, mask)
        levels = torch.stack((mu - alpha, mu, mu + alpha), dim=-1)
        distances = (grouped.unsqueeze(-1) - levels.unsqueeze(-2)).square()
        codes = distances.argmin(-1).to(torch.int8) - 1
        codes = torch.where(mask, codes, torch.zeros_like(codes))
    mu, alpha, codes = _solve_affine(grouped, codes, mask)

    candidate = mu.unsqueeze(-1) + alpha.unsqueeze(-1) * codes.float()
    direct = direct_scales.float().unsqueeze(-1) * direct_grouped.float()
    candidate_error = ((candidate - grouped).square() * mask).sum(-1)
    direct_error = ((direct - grouped).square() * mask).sum(-1)
    use_candidate = candidate_error <= direct_error
    selected_codes = torch.where(use_candidate.unsqueeze(-1), codes, direct_grouped)
    selected_mu = torch.where(use_candidate, mu, torch.zeros_like(mu))
    selected_alpha = torch.where(use_candidate, alpha, direct_scales.float())
    return AffineTernary(
        codes=selected_codes.reshape(rows, -1)[:, :columns].to(torch.int8).contiguous(),
        mu=selected_mu.contiguous(),
        alpha=selected_alpha.contiguous(),
        group_size=group_size,
    )


def affine_ternary_linear(inputs: torch.Tensor, quantized: AffineTernary) -> torch.Tensor:
    """Evaluate the fused group-sum plus ternary-dot CPU algebra."""
    if inputs.shape[-1] != quantized.codes.shape[1]:
        raise ValueError("activation width does not match affine ternary weights")
    rows, columns = quantized.codes.shape
    groups = (columns + quantized.group_size - 1) // quantized.group_size
    pad = groups * quantized.group_size - columns
    values = inputs.float()
    codes = quantized.codes.float()
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))
        codes = torch.nn.functional.pad(codes, (0, pad))
    activation_groups = values.reshape(*values.shape[:-1], groups, quantized.group_size)
    code_groups = codes.reshape(rows, groups, quantized.group_size)
    trit_dot = torch.einsum("...gk,ogk->...og", activation_groups, code_groups)
    group_sum = activation_groups.sum(-1).unsqueeze(-2)
    return (
        quantized.alpha.float() * trit_dot + quantized.mu.float() * group_sum
    ).sum(-1)


@torch.no_grad()
def activation_metric_affine_ternary(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    *,
    group_size: int = 128,
) -> AffineTernary:
    """Relocate one affine ternary matrix under its exact cached activations.

    Initialization uses activation-diagonal weighted group means and three
    clusters.  A group-coordinate affine solve accounts for cross-group
    residual, one deterministic coordinate pass relocates every trit, and a
    final affine solve closes the prescribed calibration update.  Workspace is
    proportional to ``activation_rows * output_row_shard`` rather than the
    full matrix.
    """
    if (
        weight.ndim != 2
        or inputs.ndim != 2
        or weight.shape[1] != inputs.shape[1]
        or weight.shape[0] == 0
        or inputs.shape[0] == 0
        or group_size <= 0
        or not weight.is_floating_point()
        or not inputs.is_floating_point()
    ):
        raise ValueError("activation-metric affine ternary dimensions changed")
    source = weight.detach().float().contiguous()
    calibration = inputs.detach().float().contiguous()
    rows, columns = source.shape
    groups = (columns + group_size - 1) // group_size
    energy = calibration.square().sum(0)
    codes = torch.zeros((rows, columns), dtype=torch.int8, device=source.device)
    mu = torch.zeros((rows, groups), dtype=torch.float32, device=source.device)
    alpha = torch.zeros_like(mu)

    # Frozen weighted three-cluster initialization around each group mean.
    for group in range(groups):
        start = group * group_size
        stop = min(start + group_size, columns)
        group_weight = source[:, start:stop]
        group_energy = energy[start:stop]
        denominator = group_energy.sum()
        if float(denominator) > 0:
            center = (group_weight * group_energy).sum(-1) / denominator
            centered = group_weight - center.unsqueeze(-1)
            radius = torch.sqrt(
                (centered.square() * group_energy).sum(-1) / denominator
            )
        else:
            center = group_weight.mean(-1)
            centered = group_weight - center.unsqueeze(-1)
            radius = torch.sqrt(centered.square().mean(-1))
        threshold = radius.unsqueeze(-1) * 0.5
        group_codes = torch.where(
            centered > threshold,
            torch.ones_like(centered, dtype=torch.int8),
            torch.where(
                centered < -threshold,
                -torch.ones_like(centered, dtype=torch.int8),
                torch.zeros_like(centered, dtype=torch.int8),
            ),
        )
        codes[:, start:stop] = group_codes
        mu[:, group] = center
        alpha[:, group] = radius

    target = torch.nn.functional.linear(calibration, source)
    payload = AffineTernary(codes=codes, mu=mu, alpha=alpha, group_size=group_size)
    residual = target - affine_ternary_linear(calibration, payload)

    def solve_all_groups() -> None:
        nonlocal residual
        for group in range(groups):
            start = group * group_size
            stop = min(start + group_size, columns)
            x = calibration[:, start:stop]
            group_codes = codes[:, start:stop].float()
            old_weight = mu[:, group, None] + alpha[:, group, None] * group_codes
            residual_without = residual + torch.nn.functional.linear(x, old_weight)
            sums = x.sum(-1)
            trit_projection = x @ group_codes.t()
            ss = sums.square().sum()
            sz = (sums[:, None] * trit_projection).sum(0)
            zz = trit_projection.square().sum(0)
            sy = (sums[:, None] * residual_without).sum(0)
            zy = (trit_projection * residual_without).sum(0)
            determinant = ss * zz - sz.square()
            safe = determinant.abs() > 1e-12
            fitted_mu = torch.where(
                safe,
                (sy * zz - zy * sz) / torch.where(safe, determinant, torch.ones_like(determinant)),
                sy / ss.clamp_min(1e-12),
            )
            fitted_alpha = torch.where(
                safe,
                (ss * zy - sz * sy) / torch.where(safe, determinant, torch.ones_like(determinant)),
                torch.zeros_like(zy),
            )
            negative = fitted_alpha < 0
            if bool(negative.any()):
                codes[negative, start:stop] *= -1
                group_codes = codes[:, start:stop].float()
                trit_projection[:, negative] *= -1
                fitted_alpha = fitted_alpha.abs()
            mu[:, group] = fitted_mu
            alpha[:, group] = fitted_alpha
            residual = residual_without - (
                sums[:, None] * fitted_mu[None, :]
                + trit_projection * fitted_alpha[None, :]
            )

    solve_all_groups()

    # One deterministic coordinate pass under the exact activation residual.
    candidate_trits = torch.tensor([-1.0, 0.0, 1.0], device=source.device)
    for group in range(groups):
        start = group * group_size
        stop = min(start + group_size, columns)
        group_alpha = alpha[:, group]
        for column in range(start, stop):
            x = calibration[:, column]
            current = codes[:, column].float()
            residual_without = residual + x[:, None] * (
                group_alpha * current
            )[None, :]
            correlation = (residual_without * x[:, None]).sum(0)
            x_energy = x.square().sum()
            amplitudes = candidate_trits[:, None] * group_alpha[None, :]
            score = amplitudes.square() * x_energy - 2 * amplitudes * correlation
            selected = score.argmin(0).to(torch.int8) - 1
            selected = torch.where(
                group_alpha > 0, selected, torch.zeros_like(selected)
            )
            codes[:, column] = selected
            residual = residual_without - x[:, None] * (
                group_alpha * selected.float()
            )[None, :]

    solve_all_groups()
    result = AffineTernary(
        codes=codes.contiguous(),
        mu=mu.contiguous(),
        alpha=alpha.contiguous(),
        group_size=group_size,
    )
    if (
        set(result.codes.unique().tolist()) - {-1, 0, 1}
        or bool((result.alpha < 0).any())
        or not bool(torch.isfinite(result.reconstruct()).all())
    ):
        raise RuntimeError("activation-metric affine ternary fit is invalid")
    return result


def _direct_symmetric(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    codes, scales = ternary_codes_scales(weight, group=group_size, method="optimal")
    rows, columns = codes.shape
    groups = (columns + group_size - 1) // group_size
    pad = groups * group_size - columns
    values = torch.nn.functional.pad(codes.float(), (0, pad)) if pad else codes.float()
    return (values.reshape(rows, groups, group_size) * scales.float().unsqueeze(-1)).reshape(
        rows, -1
    )[:, :columns]


def _consumer_error(
    inputs: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    candidates: Mapping[str, torch.Tensor],
) -> tuple[float, float]:
    squared_error = 0.0
    target_energy = 0.0
    with torch.no_grad():
        for name, weight in weights.items():
            target = torch.nn.functional.linear(inputs.float(), weight.float())
            observed = torch.nn.functional.linear(inputs.float(), candidates[name].float())
            squared_error += float((observed - target).square().sum())
            target_energy += float(target.square().sum())
    return squared_error, target_energy


def fit_shared_affine_rotation(
    consumer_weights: Mapping[str, torch.Tensor],
    calibration_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    *,
    pairings: torch.Tensor,
    steps: int = 8,
    learning_rate: float = 2e-3,
    row_shard: int = 128,
    group_size: int = 128,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> SharedAffineRotationFit:
    """Fit one shared hard-forward rotation over every row of every consumer.

    Every supplied consumer row enters every angle step. Calibration drives the
    update; disjoint validation inputs are evaluated only after the frozen steps.
    """
    if not consumer_weights or steps <= 0 or learning_rate <= 0 or row_shard <= 0:
        raise ValueError("invalid shared affine-rotation fit")
    names = sorted(consumer_weights)
    first = consumer_weights[names[0]]
    if first.ndim != 2:
        raise ValueError("shared affine-rotation consumers must be matrices")
    dimension = first.shape[1]
    if (
        calibration_inputs.ndim != 2
        or validation_inputs.ndim != 2
        or calibration_inputs.shape[1] != dimension
        or validation_inputs.shape[1] != dimension
        or calibration_inputs.shape[0] == 0
        or validation_inputs.shape[0] == 0
    ):
        raise ValueError("shared affine-rotation activation dimensions changed")
    if any(
        weight.ndim != 2 or weight.shape[1] != dimension
        for weight in consumer_weights.values()
    ):
        raise ValueError("all shared affine-rotation consumers need one input width")
    if pairings.ndim != 3 or pairings.shape[1:] != (dimension // 2, 2):
        raise ValueError("shared affine-rotation pairings changed")
    device = first.device
    if any(weight.device != device for weight in consumer_weights.values()):
        raise ValueError("all shared affine-rotation consumers need one device")
    train_x = calibration_inputs.detach().to(device=device, dtype=torch.float32).contiguous()
    valid_x = validation_inputs.detach().to(device=device, dtype=torch.float32).contiguous()
    frozen_weights = {
        name: consumer_weights[name].detach().float().contiguous() for name in names
    }
    pairings = pairings.to(device=device)
    angles = torch.nn.Parameter(
        torch.zeros(pairings.shape[:2], device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([angles], lr=learning_rate)

    direct_calibration_sse = 0.0
    direct_validation_sse = 0.0
    calibration_energy = 0.0
    validation_energy = 0.0
    with torch.no_grad():
        for weight in frozen_weights.values():
            for start in range(0, weight.shape[0], row_shard):
                stop = min(start + row_shard, weight.shape[0])
                source_shard = weight[start:stop]
                direct = _direct_symmetric(source_shard, group_size)
                target_train = torch.nn.functional.linear(train_x, source_shard)
                target_valid = torch.nn.functional.linear(valid_x, source_shard)
                direct_calibration_sse += float(
                    (torch.nn.functional.linear(train_x, direct) - target_train)
                    .square()
                    .sum()
                )
                direct_validation_sse += float(
                    (torch.nn.functional.linear(valid_x, direct) - target_valid)
                    .square()
                    .sum()
                )
                calibration_energy += float(target_train.square().sum())
                validation_energy += float(target_valid.square().sum())
    if calibration_energy <= 0 or validation_energy <= 0:
        raise ValueError("shared affine-rotation teacher energy is zero")

    history: list[dict[str, float | int]] = []
    total_rows = sum(weight.shape[0] for weight in frozen_weights.values())
    total_parameters = sum(weight.numel() for weight in frozen_weights.values())
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        processed_rows = 0
        rotated_inputs = apply_givens_product(train_x, angles, pairings)
        for name in names:
            weight = frozen_weights[name]
            for start in range(0, weight.shape[0], row_shard):
                stop = min(start + row_shard, weight.shape[0])
                source_shard = weight[start:stop]
                rotated_weight = apply_givens_product(source_shard, angles, pairings)
                hard = activation_metric_affine_ternary(
                    rotated_weight.detach(),
                    rotated_inputs.detach(),
                    group_size=group_size,
                ).reconstruct()
                straight_through = rotated_weight + (hard - rotated_weight).detach()
                target = torch.nn.functional.linear(train_x, source_shard)
                # Detach the activation-side copy in the surrogate gradient.
                # Letting both exact orthogonal paths differentiate would cancel
                # identically (W R^T)(R x) = W x and produce zero angle signal.
                observed = torch.nn.functional.linear(
                    rotated_inputs.detach(), straight_through
                )
                loss = (observed - target).square().sum() / calibration_energy
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite shared rotation loss for {name}")
                loss.backward()
                step_loss += float(loss.detach())
                processed_rows += stop - start
        if processed_rows != total_rows:
            raise AssertionError("shared affine-rotation step omitted consumer rows")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_([angles], 1.0))
        if not torch.isfinite(torch.tensor(gradient_norm)):
            raise RuntimeError("non-finite shared rotation gradient")
        optimizer.step()
        progress = {
            "step": step,
            "calibration_nmse_before_update": step_loss,
            "gradient_norm_before_clip": gradient_norm,
            "processed_rows": processed_rows,
        }
        history.append(progress)
        if progress_callback is not None:
            progress_callback(dict(progress))

    # The frozen storage contract is FP16 angles.  Refit every final payload
    # after the encode/decode boundary so the quality gate never sees the
    # optimizer's transient FP32 angles.
    final_angles = angles.detach().to(torch.float16).to(torch.float32).contiguous()
    rotated_train_x = apply_givens_product(train_x, final_angles, pairings)
    rotated_valid_x = apply_givens_product(valid_x, final_angles, pairings)
    quantized: dict[str, AffineTernary] = {}
    final_calibration_sse = 0.0
    final_validation_sse = 0.0
    for name in names:
        weight = frozen_weights[name]
        code_shards = []
        mu_shards = []
        alpha_shards = []
        for start in range(0, weight.shape[0], row_shard):
            stop = min(start + row_shard, weight.shape[0])
            source_shard = weight[start:stop]
            fitted_shard = activation_metric_affine_ternary(
                apply_givens_product(source_shard, final_angles, pairings),
                rotated_train_x,
                group_size=group_size,
            )
            code_shards.append(fitted_shard.codes)
            mu_shards.append(fitted_shard.mu)
            alpha_shards.append(fitted_shard.alpha)
        payload = AffineTernary(
            codes=torch.cat(code_shards, dim=0).contiguous(),
            mu=torch.cat(mu_shards, dim=0).contiguous(),
            alpha=torch.cat(alpha_shards, dim=0).contiguous(),
            group_size=group_size,
        )
        quantized[name] = payload
        with torch.no_grad():
            target_train = torch.nn.functional.linear(train_x, weight)
            target_valid = torch.nn.functional.linear(valid_x, weight)
            final_calibration_sse += float(
                (affine_ternary_linear(rotated_train_x, payload) - target_train)
                .square()
                .sum()
            )
            final_validation_sse += float(
                (affine_ternary_linear(rotated_valid_x, payload) - target_valid)
                .square()
                .sum()
            )
    covered_rows = sum(payload.codes.shape[0] for payload in quantized.values())
    covered_parameters = sum(payload.codes.numel() for payload in quantized.values())
    if covered_rows != total_rows or covered_parameters != total_parameters:
        raise AssertionError("final shared affine-rotation payload is incomplete")

    metrics: dict[str, Any] = {
        "consumers": names,
        "consumer_count": len(names),
        "rows": total_rows,
        "parameters": total_parameters,
        "calibration_rows": train_x.shape[0],
        "validation_rows": valid_x.shape[0],
        "direct_calibration_nmse": direct_calibration_sse / calibration_energy,
        "direct_validation_nmse": direct_validation_sse / validation_energy,
        "final_calibration_nmse": final_calibration_sse / calibration_energy,
        "final_validation_nmse": final_validation_sse / validation_energy,
        "relative_validation_improvement_vs_direct": (
            direct_validation_sse - final_validation_sse
        )
        / max(direct_validation_sse, 1e-12),
        "finite": all(
            bool(torch.isfinite(value).all())
            for value in [final_angles]
            + [payload.mu for payload in quantized.values()]
            + [payload.alpha for payload in quantized.values()]
        ),
        "history": history,
    }
    return SharedAffineRotationFit(
        angles=final_angles,
        pairings=pairings.detach().contiguous(),
        quantized=quantized,
        metrics=metrics,
    )
