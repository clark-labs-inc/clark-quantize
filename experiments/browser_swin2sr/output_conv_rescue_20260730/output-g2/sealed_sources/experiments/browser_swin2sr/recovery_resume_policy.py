"""Fail-closed policy for recovery sampler continuation."""

from __future__ import annotations


def sampler_resume_mode(
    previous_sampler: dict | None,
    current_sampler: dict,
    reset_data_stream: bool,
) -> str:
    if reset_data_stream:
        return "reset"
    if previous_sampler is None:
        return "fresh"
    if previous_sampler != current_sampler:
        raise ValueError(
            "recovery sampler or training corpus changed without an "
            "explicit data-stream reset"
        )
    return "exact"
