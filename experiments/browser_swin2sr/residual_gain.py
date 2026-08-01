"""Deterministic zero-parameter residual-gain helpers for Swin2SR."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np


def residual_gain_output(
    student: np.ndarray,
    bicubic: np.ndarray,
    gain: float,
) -> np.ndarray:
    if student.shape != bicubic.shape or student.ndim != 3:
        raise ValueError("student and bicubic images must have equal HWC shapes")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("residual gain must be finite and positive")
    return np.clip(bicubic + gain * (student - bicubic), 0.0, 1.0)


def balanced_folds(
    records: list[dict],
    *,
    screen_per_source: int,
    confirmation_per_source: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if screen_per_source <= 0 or confirmation_per_source <= 0:
        raise ValueError("fold sizes must be positive")
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        identity = record.get("sha256")
        source = record.get("source")
        if not identity or not source:
            raise ValueError("records require source and sha256 identities")
        groups[str(source)].append(record)
    screen = []
    confirmation = []
    needed = screen_per_source + confirmation_per_source
    for source, rows in sorted(groups.items()):
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{source}:{row['sha256']}".encode()
            ).hexdigest(),
        )
        if len(ranked) < needed:
            raise ValueError(f"source {source} has fewer than {needed} samples")
        screen.extend(ranked[:screen_per_source])
        confirmation.extend(ranked[screen_per_source:needed])
    return screen, confirmation
