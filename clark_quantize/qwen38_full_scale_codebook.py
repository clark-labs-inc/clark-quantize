"""Bounded-memory scale-codebook utilities for the complete Qwen3.8-27B image."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .scale_codebook import ScaleCodebook


@dataclass(frozen=True)
class ScaleErrorTotals:
    tensors: int
    scales: int
    source_squared: float
    error_squared: float
    maximum_tensor_nrmse: float
    maximum_tensor_p99_relative_error: float
    maximum_absolute_error: float

    @property
    def global_nrmse(self) -> float:
        if self.source_squared == 0:
            return 0.0
        return math.sqrt(self.error_squared / self.source_squared)


def fit_scale_codebook_chunked(
    scales: np.ndarray,
    *,
    entries: int = 16,
    iterations: int = 50,
    chunk_size: int = 262_144,
) -> ScaleCodebook:
    """Fit the frozen FP16 scale codebook without an ``N x entries`` allocation.

    This is the unweighted Lloyd fit used for the previously uncovered Qwen3.8
    tensors.  Assignment and final reconstruction are chunked, keeping peak
    memory bounded even for the embedding and vision matrices.  Ties select the
    lowest codebook index and all reductions have a deterministic order.
    """
    values = np.asarray(scales, dtype=np.float32)
    shape = values.shape
    flat = values.reshape(-1)
    if entries < 2 or entries > 256 or iterations <= 0 or chunk_size <= 0:
        raise ValueError("invalid scale-codebook dimensions")
    if flat.size == 0 or not np.all(np.isfinite(flat)) or np.any(flat < 0):
        raise ValueError("scales must be a non-empty finite non-negative array")

    positive = flat[flat > 0]
    codebook = np.zeros(entries, dtype=np.float32)
    slots = entries - 1
    if positive.size:
        ordered = np.sort(positive, kind="stable")
        unique = np.unique(ordered)
        if unique.size <= slots:
            codebook[1 : 1 + unique.size] = unique
            if unique.size < slots:
                codebook[1 + unique.size :] = unique[-1]
        else:
            targets = (np.arange(slots, dtype=np.float64) + 0.5) * ordered.size / slots
            positions = np.minimum(np.ceil(targets).astype(np.int64) - 1, ordered.size - 1)
            centroids = ordered[positions].astype(np.float32, copy=True)
            for _ in range(iterations):
                sums = np.zeros(slots, dtype=np.float64)
                counts = np.zeros(slots, dtype=np.int64)
                maximum_residual = -1.0
                maximum_residual_value = float(positive[0])
                for start in range(0, positive.size, chunk_size):
                    block = positive[start : start + chunk_size]
                    assignments = np.argmin(
                        np.abs(block[:, None] - centroids[None, :]), axis=1
                    )
                    sums += np.bincount(
                        assignments, weights=block.astype(np.float64), minlength=slots
                    )
                    counts += np.bincount(assignments, minlength=slots)
                    residual = (
                        block.astype(np.float64)
                        - centroids[assignments].astype(np.float64)
                    ) ** 2
                    local = int(np.argmax(residual))
                    if float(residual[local]) > maximum_residual:
                        maximum_residual = float(residual[local])
                        maximum_residual_value = float(block[local])
                updated = centroids.copy()
                populated = counts > 0
                updated[populated] = (
                    sums[populated] / counts[populated]
                ).astype(np.float32)
                updated[~populated] = maximum_residual_value
                updated.sort()
                if np.array_equal(updated, centroids):
                    break
                centroids = updated
            codebook[1:] = centroids

    frozen = codebook.astype(np.float16).astype(np.float32)
    indices = np.empty(flat.size, dtype=np.uint8)
    reconstructed = np.empty(flat.size, dtype=np.float32)
    squared_error = 0.0
    maximum_absolute_error = 0.0
    for start in range(0, flat.size, chunk_size):
        block = flat[start : start + chunk_size]
        assigned = np.argmin(np.abs(block[:, None] - frozen[None, :]), axis=1)
        rebuilt = frozen[assigned]
        indices[start : start + block.size] = assigned.astype(np.uint8)
        reconstructed[start : start + block.size] = rebuilt
        error = block.astype(np.float64) - rebuilt.astype(np.float64)
        squared_error += float(np.dot(error, error))
        maximum_absolute_error = max(
            maximum_absolute_error, float(np.max(np.abs(error), initial=0.0))
        )
    return ScaleCodebook(
        values=frozen.astype(np.float16),
        indices=indices.reshape(shape),
        reconstructed=reconstructed.reshape(shape),
        weighted_mse=squared_error / flat.size,
        maximum_absolute_error=maximum_absolute_error,
    )


def scale_error_metrics(scales: np.ndarray, fitted: ScaleCodebook) -> dict[str, float]:
    source = np.asarray(scales, dtype=np.float32).reshape(-1)
    rebuilt = np.asarray(fitted.reconstructed, dtype=np.float32).reshape(-1)
    if source.size == 0 or rebuilt.shape != source.shape:
        raise ValueError("scale reconstruction shape mismatch")
    error = source.astype(np.float64) - rebuilt.astype(np.float64)
    source_squared = float(np.dot(source.astype(np.float64), source.astype(np.float64)))
    error_squared = float(np.dot(error, error))
    positive = source > 0
    relative = (
        np.abs(error[positive]) / source[positive]
        if np.any(positive)
        else np.zeros(1, dtype=np.float64)
    )
    return {
        "sum_source_squared": source_squared,
        "sum_error_squared": error_squared,
        "normalized_rmse": math.sqrt(error_squared / source_squared)
        if source_squared
        else 0.0,
        "maximum_absolute_error": float(np.max(np.abs(error), initial=0.0)),
        "p99_relative_error_nonzero_scales": float(np.quantile(relative, 0.99)),
    }


def aggregate_scale_records(records: Iterable[dict]) -> ScaleErrorTotals:
    rows = list(records)
    if not rows:
        raise ValueError("at least one scale-codebook record is required")
    tensors = scales = 0
    source_squared = error_squared = 0.0
    maximum_tensor_nrmse = maximum_tensor_p99 = maximum_absolute = 0.0
    names: set[str] = set()
    for record in rows:
        name = record.get("tensor")
        metrics = record.get("metrics")
        count = record.get("scale_count")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("scale-codebook records require unique tensor names")
        if not isinstance(metrics, dict) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"invalid scale-codebook record for {name}")
        names.add(name)
        tensors += 1
        scales += count
        source_squared += float(metrics["sum_source_squared"])
        error_squared += float(metrics["sum_error_squared"])
        maximum_tensor_nrmse = max(
            maximum_tensor_nrmse, float(metrics["normalized_rmse"])
        )
        maximum_tensor_p99 = max(
            maximum_tensor_p99,
            float(metrics["p99_relative_error_nonzero_scales"]),
        )
        maximum_absolute = max(
            maximum_absolute, float(metrics["maximum_absolute_error"])
        )
    return ScaleErrorTotals(
        tensors=tensors,
        scales=scales,
        source_squared=source_squared,
        error_squared=error_squared,
        maximum_tensor_nrmse=maximum_tensor_nrmse,
        maximum_tensor_p99_relative_error=maximum_tensor_p99,
        maximum_absolute_error=maximum_absolute,
    )
