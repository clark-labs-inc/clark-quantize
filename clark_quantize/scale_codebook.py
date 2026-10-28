"""Deterministic weighted FP16 scale codebooks for Qwen ternary groups."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleCodebook:
    values: np.ndarray
    indices: np.ndarray
    reconstructed: np.ndarray
    weighted_mse: float
    maximum_absolute_error: float


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    total = float(cumulative[-1])
    targets = (np.arange(count, dtype=np.float64) + 0.5) * total / count
    positions = np.searchsorted(cumulative, targets, side="left")
    return sorted_values[np.minimum(positions, sorted_values.size - 1)].astype(np.float32)


def fit_scale_codebook(
    scales: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    entries: int = 16,
    iterations: int = 50,
) -> ScaleCodebook:
    """Fit a deterministic weighted 1-D codebook and freeze it to FP16.

    Entry zero is fixed to exact zero. Positive scales use the remaining
    entries. ``weights`` should be each group's sum of squared ternary codes
    (equivalently its non-zero count) when reconstruction-MSE weighting is
    desired.
    """
    values = np.asarray(scales, dtype=np.float32)
    shape = values.shape
    flat = values.reshape(-1)
    if entries < 2 or entries > 256 or iterations <= 0:
        raise ValueError("entries must be in [2, 256] and iterations positive")
    if flat.size == 0 or not np.all(np.isfinite(flat)) or np.any(flat < 0):
        raise ValueError("scales must be a non-empty finite non-negative array")
    if weights is None:
        importance = np.ones(flat.size, dtype=np.float64)
    else:
        importance = np.asarray(weights, dtype=np.float64).reshape(-1)
        if importance.size != flat.size or not np.all(np.isfinite(importance)) or np.any(importance < 0):
            raise ValueError("weights must match scales and be finite and non-negative")
    if not np.any(importance > 0):
        raise ValueError("at least one scale weight must be positive")

    positive = (flat > 0) & (importance > 0)
    codebook = np.zeros(entries, dtype=np.float32)
    positive_slots = entries - 1
    if np.any(positive):
        source = flat[positive]
        source_weights = importance[positive]
        unique = np.unique(source)
        if unique.size <= positive_slots:
            codebook[1 : 1 + unique.size] = unique
            if unique.size < positive_slots:
                codebook[1 + unique.size :] = unique[-1]
        else:
            centroids = _weighted_quantiles(source, source_weights, positive_slots)
            for _ in range(iterations):
                distances = np.abs(source[:, None] - centroids[None, :])
                assignments = np.argmin(distances, axis=1)
                updated = centroids.copy()
                residual = source_weights * (source - centroids[assignments]) ** 2
                for slot in range(positive_slots):
                    selected = assignments == slot
                    total = source_weights[selected].sum()
                    if total > 0:
                        updated[slot] = np.sum(source[selected] * source_weights[selected]) / total
                    else:
                        updated[slot] = source[int(np.argmax(residual))]
                updated.sort()
                if np.array_equal(updated, centroids):
                    break
                centroids = updated
            codebook[1:] = centroids

    frozen = codebook.astype(np.float16).astype(np.float32)
    distances = np.abs(flat[:, None] - frozen[None, :])
    indices = np.argmin(distances, axis=1).astype(np.uint8)
    reconstructed = frozen[indices]
    squared = (flat.astype(np.float64) - reconstructed.astype(np.float64)) ** 2
    weighted_mse = float(np.sum(squared * importance) / np.sum(importance))
    return ScaleCodebook(
        values=frozen.astype(np.float16),
        indices=indices.reshape(shape),
        reconstructed=reconstructed.reshape(shape),
        weighted_mse=weighted_mse,
        maximum_absolute_error=float(np.max(np.abs(flat - reconstructed))),
    )


def pack_nibbles(indices: np.ndarray) -> bytes:
    flat = np.asarray(indices, dtype=np.uint8).reshape(-1)
    if flat.size and int(flat.max()) > 15:
        raise ValueError("4-bit scale indices must be in [0, 15]")
    padded = np.pad(flat, (0, flat.size & 1), constant_values=0)
    return bytes(padded[0::2] | (padded[1::2] << 4))


def unpack_nibbles(payload: bytes, count: int) -> np.ndarray:
    if count < 0 or len(payload) != (count + 1) // 2:
        raise ValueError("4-bit scale-index payload length mismatch")
    packed = np.frombuffer(payload, dtype=np.uint8)
    output = np.empty(packed.size * 2, dtype=np.uint8)
    output[0::2] = packed & 0x0F
    output[1::2] = packed >> 4
    if count & 1 and output[count] != 0:
        raise ValueError("non-zero scale-index padding nibble")
    return output[:count]
