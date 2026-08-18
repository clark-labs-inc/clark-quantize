"""Hard-forward training primitives for the stored Qwen3.8 affine payload."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .qwen38_affine_butterfly import (
    AffineTernary,
    apply_givens_product,
)
from .qwen38_affine_codebook import (
    CompressedAffineTernary,
    SignedRowNormalizedCodebook,
    reconstruct_signed_row_normalized,
)
from .qwen38_row_scale_codebook import (
    RowNormalizedScaleCodebook,
    reconstruct_row_normalized_scales,
)


def hard_fp16_ste(values: torch.Tensor) -> torch.Tensor:
    """Use exact FP16 forward values and an identity backward surrogate."""
    frozen = values.to(torch.float16).to(torch.float32)
    return values + (frozen - values).detach()


def hard_round_ste(
    values: torch.Tensor, *, minimum: int, maximum: int
) -> torch.Tensor:
    frozen = values.round().clamp(minimum, maximum)
    return values + (frozen - values).detach()


def _sorted_signed_encoding(
    values: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize signed ratios for a meaningful scalar-index neighborhood."""
    order = np.argsort(values.astype(np.float32), kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return values[order].copy(), inverse[indices].astype(np.uint8, copy=False)


class TrainableAffinePayload(torch.nn.Module):
    """Train exactly the values that can be packed in one affine tensor."""

    def __init__(self, compressed: CompressedAffineTernary):
        super().__init__()
        payload = compressed.payload
        if payload.codes.ndim != 2:
            raise ValueError("QAT affine codes must be a matrix")
        mu_values, mu_indices = _sorted_signed_encoding(
            compressed.mu.values, compressed.mu.indices
        )
        self.group_size = payload.group_size
        self.rows, self.columns = payload.codes.shape
        self.groups = (self.columns + self.group_size - 1) // self.group_size
        self.codes_shadow = torch.nn.Parameter(payload.codes.float().clone())
        self.alpha_index_shadow = torch.nn.Parameter(
            torch.from_numpy(compressed.alpha.indices.copy()).float()
        )
        self.mu_index_shadow = torch.nn.Parameter(
            torch.from_numpy(mu_indices.copy()).float()
        )
        self.alpha_anchors = torch.nn.Parameter(
            torch.from_numpy(compressed.alpha.anchors.astype(np.float32))
        )
        self.alpha_codebook = torch.nn.Parameter(
            torch.from_numpy(compressed.alpha.values.astype(np.float32))
        )
        self.mu_anchors = torch.nn.Parameter(
            torch.from_numpy(compressed.mu.anchors.astype(np.float32))
        )
        self.mu_codebook = torch.nn.Parameter(
            torch.from_numpy(mu_values.astype(np.float32))
        )
        zero_entries = np.flatnonzero(mu_values == np.float16(0))
        if zero_entries.size == 0:
            raise ValueError("QAT signed mu codebook lost its exact zero")
        self.mu_zero_index = int(zero_entries[0])
        if (
            self.alpha_index_shadow.shape != (self.rows, self.groups)
            or self.mu_index_shadow.shape != (self.rows, self.groups)
            or self.alpha_codebook.shape != (16,)
            or self.mu_codebook.shape != (16,)
        ):
            raise ValueError("QAT affine storage dimensions changed")

    @staticmethod
    def _indexed_values(codebook: torch.Tensor, shadow: torch.Tensor) -> torch.Tensor:
        """Hard integer lookup with a centered finite-difference index surrogate."""
        hard_index = shadow.round().clamp(0, 15).long()
        hard_value = codebook[hard_index]
        lower = (hard_index - 1).clamp(0, 15)
        upper = (hard_index + 1).clamp(0, 15)
        width = (upper - lower).clamp_min(1).to(codebook.dtype)
        slope = (codebook[upper] - codebook[lower]) / width
        surrogate = hard_value + (shadow - hard_index.detach().to(shadow.dtype)) * slope
        return surrogate + (hard_value - surrogate).detach()

    def hard_codes(self) -> torch.Tensor:
        return hard_round_ste(self.codes_shadow, minimum=-1, maximum=1)

    def decoded_group_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_anchors = hard_fp16_ste(self.alpha_anchors.clamp_min(0))
        alpha_codebook = hard_fp16_ste(self.alpha_codebook.clamp_min(0))
        mu_anchors = hard_fp16_ste(self.mu_anchors.clamp_min(0))
        mu_codebook = hard_fp16_ste(self.mu_codebook)
        alpha = alpha_anchors[:, None] * self._indexed_values(
            alpha_codebook, self.alpha_index_shadow
        )
        mu = mu_anchors[:, None] * self._indexed_values(
            mu_codebook, self.mu_index_shadow
        )
        return alpha, mu

    def decoded_rotated_weight(self) -> torch.Tensor:
        codes = self.hard_codes()
        alpha, mu = self.decoded_group_parameters()
        pad = self.groups * self.group_size - self.columns
        if pad:
            codes = torch.nn.functional.pad(codes, (0, pad))
        return (
            mu[:, :, None] + alpha[:, :, None] * codes.reshape(
                self.rows, self.groups, self.group_size
            )
        ).reshape(self.rows, -1)[:, : self.columns]

    def decoded_effective_weight(
        self,
        *,
        angles: torch.Tensor | None = None,
        pairings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.decoded_rotated_weight()
        if angles is None:
            if pairings is not None:
                raise ValueError("QAT pairings supplied without angles")
            return weight
        if pairings is None:
            raise ValueError("QAT angles supplied without pairings")
        return apply_givens_product(
            weight, hard_fp16_ste(angles), pairings, inverse=True
        )

    @torch.no_grad()
    def project_training_state(self) -> None:
        self.codes_shadow.clamp_(-1, 1)
        self.alpha_index_shadow.clamp_(0, 15)
        self.mu_index_shadow.clamp_(0, 15)
        self.alpha_anchors.clamp_(min=0)
        self.mu_anchors.clamp_(min=0)
        self.alpha_codebook.clamp_(min=0)
        self.alpha_codebook[0] = 0
        self.mu_codebook[self.mu_zero_index] = 0

    @torch.no_grad()
    def export_compressed(self) -> CompressedAffineTernary:
        codes = self.codes_shadow.round().clamp(-1, 1).to(torch.int8).cpu()
        alpha_indices = (
            self.alpha_index_shadow.round().clamp(0, 15).to(torch.uint8).cpu().numpy()
        )
        mu_indices = (
            self.mu_index_shadow.round().clamp(0, 15).to(torch.uint8).cpu().numpy()
        )
        alpha_anchors = self.alpha_anchors.clamp_min(0).to(torch.float16).cpu().numpy()
        alpha_values = self.alpha_codebook.clamp_min(0).to(torch.float16).cpu().numpy()
        mu_anchors = self.mu_anchors.clamp_min(0).to(torch.float16).cpu().numpy()
        mu_values = self.mu_codebook.to(torch.float16).cpu().numpy()
        alpha_values[0] = np.float16(0)
        mu_values[self.mu_zero_index] = np.float16(0)
        alpha_reconstructed = reconstruct_row_normalized_scales(
            alpha_anchors, alpha_values, alpha_indices
        )
        mu_reconstructed = reconstruct_signed_row_normalized(
            mu_anchors, mu_values, mu_indices
        )
        payload = AffineTernary(
            codes=codes,
            alpha=torch.from_numpy(alpha_reconstructed.copy()),
            mu=torch.from_numpy(mu_reconstructed.copy()),
            group_size=self.group_size,
        )
        return CompressedAffineTernary(
            payload=payload,
            alpha=RowNormalizedScaleCodebook(
                anchors=alpha_anchors,
                values=alpha_values,
                indices=alpha_indices,
                reconstructed=alpha_reconstructed,
            ),
            mu=SignedRowNormalizedCodebook(
                anchors=mu_anchors,
                values=mu_values,
                indices=mu_indices,
                reconstructed=mu_reconstructed,
            ),
        )


class TrainableStoredRotation(torch.nn.Module):
    def __init__(self, angles: torch.Tensor, pairings: torch.Tensor):
        super().__init__()
        if angles.ndim != 2 or pairings.shape != (*angles.shape, 2):
            raise ValueError("QAT stored rotation dimensions changed")
        self.angles = torch.nn.Parameter(angles.detach().float().clone())
        self.register_buffer("pairings", pairings.detach().clone(), persistent=False)

    def effective_weight(self, rotated_weight: torch.Tensor) -> torch.Tensor:
        return apply_givens_product(
            rotated_weight, hard_fp16_ste(self.angles), self.pairings, inverse=True
        )

    @torch.no_grad()
    def export_angles(self) -> torch.Tensor:
        return self.angles.to(torch.float16).to(torch.float32).cpu().contiguous()


class StoredAffineLinear(torch.nn.Module):
    """Hard-forward affine linear using the actual rotation-then-GEMV algebra."""

    def __init__(
        self,
        payload: TrainableAffinePayload,
        rotation: TrainableStoredRotation | None,
    ):
        super().__init__()
        self.payload = payload
        # A rotation is shared across fanout consumers. Keep a non-owning module
        # reference so it is registered and optimized exactly once by the site.
        object.__setattr__(self, "_rotation_ref", rotation)

    def _decoded_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.float()
        rotation = self._rotation_ref
        if rotation is not None:
            values = apply_givens_product(
                values,
                hard_fp16_ste(rotation.angles),
                rotation.pairings,
            )
        return torch.nn.functional.linear(
            values, self.payload.decoded_rotated_weight()
        ).to(inputs.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # A 27B block would otherwise retain every reconstructed dense matrix
        # until backward. Non-reentrant checkpointing recomputes the exact hard
        # decode and keeps memory-bounded post-training probes feasible without
        # changing forward values or the deployed stored representation.
        if torch.is_grad_enabled():
            return checkpoint(self._decoded_forward, inputs, use_reentrant=False)
        return self._decoded_forward(inputs)


class StoredAffineConvWeight(torch.nn.Module):
    """Weight proxy for Qwen's functional depthwise causal convolution path."""

    def __init__(self, payload: TrainableAffinePayload, shape: tuple[int, ...]):
        super().__init__()
        if len(shape) != 3 or shape[0] != payload.rows:
            raise ValueError("stored affine convolution shape changed")
        self.payload = payload
        self.shape = shape
        self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        return self.payload.decoded_rotated_weight().reshape(self.shape).to(torch.bfloat16)


@dataclass(frozen=True)
class QATParameterGroups:
    groups: list[dict]
    ternary_parameters: int
    index_parameters: int
    continuous_parameters: int


def qat_parameter_groups(
    payloads: Iterable[TrainableAffinePayload],
    rotations: Iterable[TrainableStoredRotation],
    *,
    ternary_learning_rate: float,
    index_learning_rate: float,
    continuous_learning_rate: float,
) -> QATParameterGroups:
    payloads = list(payloads)
    rotations = list(rotations)
    ternary = [payload.codes_shadow for payload in payloads]
    indices = [
        parameter
        for payload in payloads
        for parameter in (payload.alpha_index_shadow, payload.mu_index_shadow)
    ]
    continuous = [
        parameter
        for payload in payloads
        for parameter in (
            payload.alpha_anchors,
            payload.alpha_codebook,
            payload.mu_anchors,
            payload.mu_codebook,
        )
    ] + [rotation.angles for rotation in rotations]
    if not ternary or not indices or not continuous:
        raise ValueError("QAT parameter groups are incomplete")
    return QATParameterGroups(
        groups=[
            {"params": ternary, "lr": ternary_learning_rate},
            {"params": indices, "lr": index_learning_rate},
            {"params": continuous, "lr": continuous_learning_rate},
        ],
        ternary_parameters=sum(value.numel() for value in ternary),
        index_parameters=sum(value.numel() for value in indices),
        continuous_parameters=sum(value.numel() for value in continuous),
    )
