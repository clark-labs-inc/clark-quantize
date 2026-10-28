"""Activation-dual binary residual factors for Qwen3.8-27B.

For a ternary weight residual ``E`` and calibration activations ``X``, the
runtime correction remains ``A (C x)`` with signed-INT4 ``A`` and binary-row
``C``.  Unlike a weight-space SVD, this construction first finds the dominant
subspace of the actual residual outputs ``X E.T``.  It then lifts those sample
features back through ``X`` before binary-quantizing ``C``.  The representation
rate and CPU algebra are unchanged; only the geometry used to select the right
factor changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .qwen38_activation_binary_mlp import fit_activation_left_int4
from .qwen38_binary_low_rank_residual import (
    BinaryRightFactor,
    binary_right_values,
    quantize_binary_rows,
)


@dataclass(frozen=True)
class ActivationDualBinaryFit:
    right: BinaryRightFactor
    left_codes: tuple[torch.Tensor, ...]
    left_scales: tuple[torch.Tensor, ...]
    left_coefficients: tuple[torch.Tensor, ...]
    initial_squared_error: float
    objective_history: tuple[float, ...]
    best_iteration: int


def ridge_lift_activation_features(
    activations: torch.Tensor,
    features: torch.Tensor,
    *,
    relative_ridge: float,
) -> torch.Tensor:
    """Return minimum-norm right rows whose activation image fits ``features``.

    The dual solve avoids a ``d x d`` system when calibration rows are much
    fewer than the model width:

    ``C.T = X.T (X X.T + lambda I)^-1 F``.
    """
    if (
        activations.ndim != 2
        or features.ndim != 2
        or activations.shape[0] != features.shape[0]
        or activations.shape[0] == 0
        or activations.shape[1] == 0
        or features.shape[1] == 0
        or not activations.is_floating_point()
        or not features.is_floating_point()
        or not bool(torch.isfinite(activations).all())
        or not bool(torch.isfinite(features).all())
        or relative_ridge <= 0
    ):
        raise ValueError("invalid activation-feature lifting problem")
    x = activations.to(torch.float32)
    target = features.to(torch.float32)
    gram = x @ x.T
    ridge = relative_ridge * max(
        float(torch.diagonal(gram).mean().item()), 1e-20
    )
    regularized = gram + torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    ) * ridge
    dual = torch.linalg.solve(regularized, target)
    return (dual.T @ x).contiguous()


def _ridge_feature_targets(
    targets: torch.Tensor,
    left_coefficients: torch.Tensor,
    *,
    relative_ridge: float,
) -> torch.Tensor:
    """Fit sample-space rank features for a frozen exported left factor."""
    gram = left_coefficients.T @ left_coefficients
    ridge = relative_ridge * max(
        float(torch.diagonal(gram).mean().item()), 1e-20
    )
    regularized = gram + torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    ) * ridge
    return torch.linalg.solve(
        regularized, (targets @ left_coefficients).T
    ).T.contiguous()


def activation_dual_outputs(
    inputs: torch.Tensor,
    left_coefficients: Sequence[torch.Tensor],
    right: BinaryRightFactor,
) -> tuple[torch.Tensor, ...]:
    """Evaluate one shared binary-right factor for one or more projections."""
    values = binary_right_values(right)
    if (
        inputs.ndim != 2
        or inputs.shape[1] != values.shape[1]
        or not inputs.is_floating_point()
        or not left_coefficients
        or any(
            left.ndim != 2 or left.shape[1] != values.shape[0]
            for left in left_coefficients
        )
    ):
        raise ValueError("activation-dual factor dimensions disagree")
    features = inputs.to(torch.float32) @ values.T
    return tuple(
        (features @ left.to(torch.float32).T).contiguous()
        for left in left_coefficients
    )


@torch.no_grad()
def fit_activation_dual_binary_residual(
    activations: torch.Tensor,
    target_outputs: Sequence[torch.Tensor],
    *,
    rank: int,
    iterations: int = 4,
    pca_iterations: int = 2,
    seed: int = 0xAD017A1,
    relative_ridge: float = 1e-4,
) -> ActivationDualBinaryFit:
    """Fit an INT4-by-binary correction in the activation-output geometry.

    All output matrices share the binary right factor, matching gate/up reuse.
    Each alternating iterate is evaluated after both binary-right and INT4-left
    export; the best exported iterate is retained.
    """
    targets = tuple(target_outputs)
    if (
        activations.ndim != 2
        or activations.numel() == 0
        or not activations.is_floating_point()
        or not bool(torch.isfinite(activations).all())
        or not targets
        or any(
            target.ndim != 2
            or target.shape[0] != activations.shape[0]
            or target.shape[1] == 0
            or not target.is_floating_point()
            or not bool(torch.isfinite(target).all())
            for target in targets
        )
        or rank <= 0
        or rank > min(activations.shape[0], activations.shape[1])
        or rank > sum(target.shape[1] for target in targets)
        or iterations <= 0
        or pca_iterations < 0
        or relative_ridge <= 0
    ):
        raise ValueError("invalid activation-dual binary fitting problem")

    x = activations.detach().to(torch.float32).contiguous()
    outputs = tuple(
        target.detach().to(torch.float32).contiguous() for target in targets
    )
    joined = torch.cat(outputs, dim=1)
    initial_error = float(joined.square().sum(dtype=torch.float64).item())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        left_vectors, singular_values, _ = torch.pca_lowrank(
            joined, q=rank, center=False, niter=pca_iterations
        )
    desired_features = (
        left_vectors * singular_values.to(torch.float32)[None, :]
    ).contiguous()

    history: list[float] = []
    best_error = math.inf
    best_iteration = -1
    best_right = None
    best_left_codes = best_left_scales = best_left = None
    for iteration in range(iterations):
        continuous_right = ridge_lift_activation_features(
            x, desired_features, relative_ridge=relative_ridge
        )
        right = quantize_binary_rows(continuous_right)
        features = x @ binary_right_values(right).T
        left_fits = tuple(
            fit_activation_left_int4(
                features, target, relative_ridge=relative_ridge
            )
            for target in outputs
        )
        left = tuple(fit.coefficients for fit in left_fits)
        predicted = torch.cat(
            tuple(features @ coefficients.T for coefficients in left), dim=1
        )
        error = float(
            (joined - predicted).square().sum(dtype=torch.float64).item()
        )
        history.append(error)
        if error < best_error:
            best_error = error
            best_iteration = iteration + 1
            best_right = BinaryRightFactor(
                signs=right.signs.clone(), scales=right.scales.clone()
            )
            best_left_codes = tuple(fit.codes.clone() for fit in left_fits)
            best_left_scales = tuple(fit.scales.clone() for fit in left_fits)
            best_left = tuple(fit.coefficients.clone() for fit in left_fits)
        desired_features = _ridge_feature_targets(
            joined,
            torch.cat(left, dim=0),
            relative_ridge=relative_ridge,
        )

    if (
        best_right is None
        or best_left_codes is None
        or best_left_scales is None
        or best_left is None
    ):
        raise AssertionError("activation-dual binary fitting produced no export")
    return ActivationDualBinaryFit(
        right=best_right,
        left_codes=best_left_codes,
        left_scales=best_left_scales,
        left_coefficients=best_left,
        initial_squared_error=initial_error,
        objective_history=tuple(history),
        best_iteration=best_iteration,
    )
