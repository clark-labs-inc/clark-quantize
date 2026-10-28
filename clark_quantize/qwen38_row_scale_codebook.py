"""Row-normalized 4-bit scale coding for Qwen3.8-27B ternary matrices."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RowNormalizedScaleCodebook:
    """One FP16 anchor per output row plus per-tensor FP16 ratio entries."""

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


def reconstruct_row_normalized_scales(
    anchors: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Decode stored FP16 factors with the frozen FP32 multiply contract."""
    row_anchors = np.asarray(anchors)
    ratios = np.asarray(values)
    codes = np.asarray(indices)
    if (
        row_anchors.ndim != 1
        or ratios.ndim != 1
        or codes.ndim != 2
        or row_anchors.size != codes.shape[0]
        or ratios.size < 2
        or ratios.size > 256
        or codes.size == 0
        or np.any(codes >= ratios.size)
    ):
        raise ValueError("invalid row-normalized scale payload")
    frozen_anchors = row_anchors.astype(np.float16, copy=False).astype(np.float32)
    frozen_ratios = ratios.astype(np.float16, copy=False).astype(np.float32)
    return frozen_anchors[:, None] * frozen_ratios[codes]


def _initial_log_centroids(log_values: np.ndarray, slots: int) -> np.ndarray:
    ordered = np.sort(log_values, kind="stable")
    unique = np.unique(ordered)
    if unique.size <= slots:
        centroids = np.empty(slots, dtype=np.float32)
        centroids[: unique.size] = unique
        centroids[unique.size :] = unique[-1]
        return centroids
    targets = (np.arange(slots, dtype=np.float64) + 0.5) * ordered.size / slots
    positions = np.minimum(np.ceil(targets).astype(np.int64) - 1, ordered.size - 1)
    return ordered[positions].astype(np.float32, copy=True)


def fit_row_normalized_log_codebook(
    scales: np.ndarray,
    *,
    entries: int = 16,
    iterations: int = 50,
    chunk_rows: int = 256,
) -> RowNormalizedScaleCodebook:
    """Fit a deterministic multiplicative codebook with bounded workspace.

    Source scales are a ``[rows, groups]`` matrix. Every nonzero row gets an
    FP16 geometric-mean anchor. A per-tensor codebook is trained in log-ratio
    space, then final indices minimize absolute error against the frozen FP16
    anchor/ratio products. Index zero represents an exact zero scale.
    """
    source = np.asarray(scales, dtype=np.float32)
    if (
        source.ndim != 2
        or source.size == 0
        or entries < 2
        or entries > 16
        or iterations <= 0
        or chunk_rows <= 0
        or not np.all(np.isfinite(source))
        or np.any(source < 0)
    ):
        raise ValueError("invalid row-normalized scale-codebook input")

    positive = source > 0
    positive_counts = positive.sum(axis=1)
    anchors = np.zeros(source.shape[0], dtype=np.float32)
    active = positive_counts > 0
    if np.any(active):
        log_source = np.zeros_like(source, dtype=np.float64)
        np.log(source, out=log_source, where=positive)
        anchors[active] = np.exp(
            log_source[active].sum(axis=1) / positive_counts[active]
        ).astype(np.float32)
    anchors_frozen = anchors.astype(np.float16)
    if np.any(active & (anchors_frozen == 0)):
        raise ValueError("FP16 row anchor underflowed to zero")
    anchor_f32 = anchors_frozen.astype(np.float32)

    normalized_logs = np.zeros_like(source, dtype=np.float32)
    if np.any(positive):
        normalized_logs[positive] = (
            np.log(source[positive].astype(np.float64))
            - np.repeat(np.log(anchor_f32[active].astype(np.float64)), positive_counts[active])
        ).astype(np.float32)
    logs = normalized_logs[positive]
    slots = entries - 1
    centroids = np.zeros(slots, dtype=np.float32)
    if logs.size:
        centroids = _initial_log_centroids(logs, slots)
        for _ in range(iterations):
            sums = np.zeros(slots, dtype=np.float64)
            counts = np.zeros(slots, dtype=np.int64)
            maximum_residual = -1.0
            maximum_value = float(logs[0])
            chunk_values = max(1, chunk_rows * source.shape[1])
            for start in range(0, logs.size, chunk_values):
                block = logs[start : start + chunk_values]
                assignments = np.argmin(
                    np.abs(block[:, None] - centroids[None, :]), axis=1
                )
                sums += np.bincount(
                    assignments, weights=block.astype(np.float64), minlength=slots
                )
                counts += np.bincount(assignments, minlength=slots)
                residual = (
                    block.astype(np.float64) - centroids[assignments].astype(np.float64)
                ) ** 2
                local = int(np.argmax(residual))
                if float(residual[local]) > maximum_residual:
                    maximum_residual = float(residual[local])
                    maximum_value = float(block[local])
            updated = centroids.copy()
            populated = counts > 0
            updated[populated] = (sums[populated] / counts[populated]).astype(np.float32)
            updated[~populated] = maximum_value
            updated.sort()
            if np.array_equal(updated, centroids):
                break
            centroids = updated

    ratios = np.zeros(entries, dtype=np.float32)
    if logs.size:
        ratios[1:] = np.exp(centroids.astype(np.float64)).astype(np.float32)
    else:
        # There is nothing to approximate, but retain a valid positive payload
        # for unused nonzero entries so an all-zero tensor decodes exactly.
        ratios[1:] = 1.0
    ratios_frozen = ratios.astype(np.float16)
    if not np.all(np.isfinite(ratios_frozen)) or np.any(ratios_frozen[1:] <= 0):
        raise ValueError("FP16 ratio codebook is non-finite or underflowed")
    frozen_ratios = ratios_frozen.astype(np.float32)

    indices = np.zeros(source.shape, dtype=np.uint8)
    reconstructed = np.zeros(source.shape, dtype=np.float32)
    for row_start in range(0, source.shape[0], chunk_rows):
        row_stop = min(row_start + chunk_rows, source.shape[0])
        block = source[row_start:row_stop]
        candidates = (
            anchor_f32[row_start:row_stop, None, None]
            * frozen_ratios[None, None, :]
        )
        assigned = np.argmin(np.abs(block[:, :, None] - candidates), axis=-1).astype(np.uint8)
        assigned[block == 0] = 0
        indices[row_start:row_stop] = assigned
        reconstructed[row_start:row_stop] = np.take_along_axis(
            candidates, assigned[:, :, None], axis=-1
        )[..., 0]

    decoded = reconstruct_row_normalized_scales(
        anchors_frozen, ratios_frozen, indices
    )
    if not np.array_equal(decoded, reconstructed):
        raise AssertionError("row-normalized scale decode contract diverged")

    return RowNormalizedScaleCodebook(
        anchors=anchors_frozen,
        values=ratios_frozen,
        indices=indices,
        reconstructed=reconstructed,
    )
