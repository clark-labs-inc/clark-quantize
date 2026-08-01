"""Deterministic exact-byte allocation for output-fragility scale choices."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionChoice:
    label: str
    storage_bytes: int
    fragility: float


@dataclass(frozen=True)
class Allocation:
    choices: dict[str, PrecisionChoice]
    storage_bytes: int
    fragility: float


def aligned_scale_storage_bytes(
    groups: int,
    *,
    bit_width: int | None = None,
    center_count: int | None = None,
) -> int:
    """Return exact 4-byte-aligned scale payload for one surface."""

    if groups <= 0:
        raise ValueError("scale group count must be positive")
    if bit_width is None:
        payload = groups * 2
        return payload + (-payload) % 4
    if bit_width < 1 or bit_width > 8:
        raise ValueError("scale-codebook bits must be between one and eight")
    maximum_centers = min(1 << bit_width, groups)
    centers = maximum_centers if center_count is None else center_count
    if centers < 1 or centers > maximum_centers:
        raise ValueError("scale center count is inconsistent with bit width")
    stored_bits = 0 if centers == 1 else math.ceil(math.log2(centers))
    center_bytes = centers * 2
    index_bytes = (groups * stored_bits + 7) // 8
    payload = center_bytes + index_bytes
    return payload + (-payload) % 4


def _validate_choices(choices_by_surface: dict[str, list[PrecisionChoice]]) -> None:
    if not choices_by_surface:
        raise ValueError("allocation requires at least one surface")
    for surface, choices in choices_by_surface.items():
        if not surface or not choices:
            raise ValueError("every named surface requires choices")
        labels = [choice.label for choice in choices]
        if len(labels) != len(set(labels)):
            raise ValueError(f"surface has duplicate precision labels: {surface}")
        for choice in choices:
            if choice.storage_bytes < 0:
                raise ValueError("precision storage must be non-negative")
            if not math.isfinite(choice.fragility) or choice.fragility < 0:
                raise ValueError("precision fragility must be finite and non-negative")


def _prune(states: dict[int, tuple[float, tuple[str, ...]]]):
    """Drop states whose cost and fragility are both no better."""

    result = {}
    best_fragility = math.inf
    for cost in sorted(states):
        fragility, labels = states[cost]
        if fragility < best_fragility:
            result[cost] = (fragility, labels)
            best_fragility = fragility
    return result


def allocate_precision(
    choices_by_surface: dict[str, list[PrecisionChoice]],
    budget_bytes: int,
) -> Allocation:
    """Solve the multiple-choice byte-budget problem with deterministic DP."""

    _validate_choices(choices_by_surface)
    if budget_bytes < 0:
        raise ValueError("allocation budget must be non-negative")
    surfaces = sorted(choices_by_surface)
    states: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, ())}
    for surface in surfaces:
        updated: dict[int, tuple[float, tuple[str, ...]]] = {}
        choices = sorted(choices_by_surface[surface], key=lambda item: item.label)
        for prior_cost, (prior_fragility, prior_labels) in states.items():
            for choice in choices:
                cost = prior_cost + choice.storage_bytes
                if cost > budget_bytes:
                    continue
                candidate = (
                    prior_fragility + choice.fragility,
                    (*prior_labels, choice.label),
                )
                incumbent = updated.get(cost)
                if incumbent is None or candidate < incumbent:
                    updated[cost] = candidate
        if not updated:
            raise ValueError(f"byte budget cannot cover surface: {surface}")
        states = _prune(updated)
    best_cost, (best_fragility, labels) = min(
        states.items(),
        key=lambda item: (item[1][0], item[0], item[1][1]),
    )
    selected = {
        surface: next(
            choice
            for choice in choices_by_surface[surface]
            if choice.label == label
        )
        for surface, label in zip(surfaces, labels)
    }
    return Allocation(selected, best_cost, best_fragility)
