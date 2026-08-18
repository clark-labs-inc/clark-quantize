"""Frozen 4-bit parameter coding for Qwen3.8 affine ternary groups.

The affine representation needs two group parameters instead of the usual one:
non-negative ``alpha`` and signed ``mu``.  Both are encoded with one FP16 row
anchor, a per-tensor 16-entry FP16 ratio codebook, and packed 4-bit indices.
This module performs the actual encode/decode used by the complete-block gate;
uncompressed group parameters are never scored as the candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .qwen38_affine_butterfly import AffineTernary
from .qwen38_row_scale_codebook import (
    RowNormalizedScaleCodebook,
    fit_row_normalized_log_codebook,
)


@dataclass(frozen=True)
class SignedRowNormalizedCodebook:
    """One FP16 magnitude anchor per row and signed FP16 ratio entries."""

    anchors: np.ndarray
    values: np.ndarray
    indices: np.ndarray
    reconstructed: np.ndarray

    @property
    def storage_bytes(self) -> int:
        return (
            int(self.anchors.nbytes)
            + int(self.values.nbytes)
            + math.ceil(int(self.indices.size) / 2)
        )


@dataclass(frozen=True)
class CompressedAffineTernary:
    """Affine ternary payload after applying both frozen 4-bit codebooks."""

    payload: AffineTernary
    alpha: RowNormalizedScaleCodebook
    mu: SignedRowNormalizedCodebook

    @property
    def parameter_storage_bytes(self) -> int:
        return self.alpha.storage_bytes + self.mu.storage_bytes


def reconstruct_signed_row_normalized(
    anchors: np.ndarray, values: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    row_anchors = np.asarray(anchors)
    ratios = np.asarray(values)
    codes = np.asarray(indices)
    if (
        row_anchors.ndim != 1
        or ratios.shape != (16,)
        or codes.ndim != 2
        or row_anchors.size != codes.shape[0]
        or codes.size == 0
        or np.any(codes >= 16)
    ):
        raise ValueError("invalid signed row-normalized payload")
    frozen_anchors = row_anchors.astype(np.float16, copy=False).astype(np.float32)
    frozen_ratios = ratios.astype(np.float16, copy=False).astype(np.float32)
    if (
        not np.all(np.isfinite(frozen_anchors))
        or not np.all(np.isfinite(frozen_ratios))
        or np.any(frozen_anchors < 0)
        or frozen_ratios[0] != 0
    ):
        raise ValueError("non-canonical signed row-normalized payload")
    return frozen_anchors[:, None] * frozen_ratios[codes]


def _initial_centroids(values: np.ndarray, slots: int) -> np.ndarray:
    ordered = np.sort(values, kind="stable")
    unique = np.unique(ordered)
    if unique.size <= slots:
        result = np.empty(slots, dtype=np.float32)
        result[: unique.size] = unique
        result[unique.size :] = unique[-1]
        return result
    targets = (np.arange(slots, dtype=np.float64) + 0.5) * ordered.size / slots
    positions = np.minimum(np.ceil(targets).astype(np.int64) - 1, ordered.size - 1)
    return ordered[positions].astype(np.float32, copy=True)


def fit_signed_row_normalized_codebook(
    parameters: np.ndarray,
    *,
    iterations: int = 50,
    chunk_rows: int = 256,
) -> SignedRowNormalizedCodebook:
    """Fit the deterministic signed counterpart of the alpha codebook.

    Index zero is an exact zero.  Each active row uses the geometric mean of
    nonzero absolute values as its FP16 anchor.  The other fifteen entries are
    deterministic one-dimensional Lloyd centroids of signed normalized values;
    final assignment minimizes error after anchors and centroids are frozen to
    FP16.
    """
    source = np.asarray(parameters, dtype=np.float32)
    if (
        source.ndim != 2
        or source.size == 0
        or iterations <= 0
        or chunk_rows <= 0
        or not np.all(np.isfinite(source))
    ):
        raise ValueError("invalid signed row-normalized input")

    nonzero = source != 0
    counts = nonzero.sum(axis=1)
    active = counts > 0
    anchors = np.zeros(source.shape[0], dtype=np.float32)
    if np.any(active):
        log_abs = np.zeros_like(source, dtype=np.float64)
        np.log(np.abs(source), out=log_abs, where=nonzero)
        anchors[active] = np.exp(
            log_abs[active].sum(axis=1) / counts[active]
        ).astype(np.float32)
    anchors_f16 = anchors.astype(np.float16)
    if np.any(active & (anchors_f16 == 0)) or not np.all(np.isfinite(anchors_f16)):
        raise ValueError("signed FP16 row anchor is not representable")
    anchor_f32 = anchors_f16.astype(np.float32)

    normalized = np.zeros_like(source, dtype=np.float32)
    normalized[nonzero] = source[nonzero] / np.repeat(anchor_f32[active], counts[active])
    samples = normalized[nonzero]
    slots = 15
    if samples.size:
        centroids = _initial_centroids(samples, slots)
        chunk_values = max(1, chunk_rows * source.shape[1])
        for _ in range(iterations):
            sums = np.zeros(slots, dtype=np.float64)
            populations = np.zeros(slots, dtype=np.int64)
            farthest_residual = -1.0
            farthest_value = float(samples[0])
            for start in range(0, samples.size, chunk_values):
                block = samples[start : start + chunk_values]
                assigned = np.argmin(
                    np.abs(block[:, None] - centroids[None, :]), axis=1
                )
                sums += np.bincount(
                    assigned, weights=block.astype(np.float64), minlength=slots
                )
                populations += np.bincount(assigned, minlength=slots)
                residual = (
                    block.astype(np.float64)
                    - centroids[assigned].astype(np.float64)
                ) ** 2
                local = int(np.argmax(residual))
                if float(residual[local]) > farthest_residual:
                    farthest_residual = float(residual[local])
                    farthest_value = float(block[local])
            updated = centroids.copy()
            populated = populations > 0
            updated[populated] = (
                sums[populated] / populations[populated]
            ).astype(np.float32)
            updated[~populated] = farthest_value
            updated.sort()
            if np.array_equal(updated, centroids):
                break
            centroids = updated
    else:
        centroids = np.ones(slots, dtype=np.float32)

    codebook = np.empty(16, dtype=np.float32)
    codebook[0] = 0
    codebook[1:] = centroids
    codebook_f16 = codebook.astype(np.float16)
    if not np.all(np.isfinite(codebook_f16)) or codebook_f16[0] != 0:
        raise ValueError("signed FP16 ratio codebook is not representable")

    indices = np.zeros(source.shape, dtype=np.uint8)
    reconstructed = np.zeros(source.shape, dtype=np.float32)
    frozen_codebook = codebook_f16.astype(np.float32)
    for start in range(0, source.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source.shape[0])
        candidates = anchor_f32[start:stop, None, None] * frozen_codebook[None, None, :]
        assigned = np.argmin(
            np.abs(source[start:stop, :, None] - candidates), axis=-1
        ).astype(np.uint8)
        assigned[source[start:stop] == 0] = 0
        indices[start:stop] = assigned
        reconstructed[start:stop] = np.take_along_axis(
            candidates, assigned[:, :, None], axis=-1
        )[..., 0]

    decoded = reconstruct_signed_row_normalized(
        anchors_f16, codebook_f16, indices
    )
    if not np.array_equal(decoded, reconstructed):
        raise AssertionError("signed row-normalized decode contract diverged")
    return SignedRowNormalizedCodebook(
        anchors=anchors_f16,
        values=codebook_f16,
        indices=indices,
        reconstructed=reconstructed,
    )


def compress_affine_parameters(
    payload: AffineTernary,
    *,
    iterations: int = 50,
    chunk_rows: int = 256,
) -> CompressedAffineTernary:
    """Apply the storage representation before any candidate is evaluated."""
    alpha = fit_row_normalized_log_codebook(
        payload.alpha.detach().cpu().float().numpy(),
        entries=16,
        iterations=iterations,
        chunk_rows=chunk_rows,
    )
    mu = fit_signed_row_normalized_codebook(
        payload.mu.detach().cpu().float().numpy(),
        iterations=iterations,
        chunk_rows=chunk_rows,
    )
    device = payload.codes.device
    compressed = AffineTernary(
        codes=payload.codes,
        mu=torch.from_numpy(mu.reconstructed.copy()).to(device=device),
        alpha=torch.from_numpy(alpha.reconstructed.copy()).to(device=device),
        group_size=payload.group_size,
    )
    if not bool(torch.isfinite(compressed.reconstruct()).all()):
        raise RuntimeError("compressed affine payload is non-finite")
    return CompressedAffineTernary(payload=compressed, alpha=alpha, mu=mu)
