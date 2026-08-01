"""Cumulative recovery-step accounting independent of the base QAT race."""

from __future__ import annotations


def recovery_offsets(payload: dict) -> tuple[int, int]:
    recovery = payload.get("recovery", {})
    initial_global_step = int(
        recovery.get("initial_global_step", payload["step"])
    )
    completed = int(
        recovery.get(
            "recovery_steps_completed",
            max(0, int(payload["step"]) - initial_global_step),
        )
    )
    if initial_global_step + completed != int(payload["step"]):
        raise ValueError("recovery step accounting is inconsistent")
    return initial_global_step, completed


def checkpoint_recovery_config(config: dict, global_step: int) -> dict:
    initial = int(config["initial_global_step"])
    completed = int(global_step) - initial
    if completed < 0:
        raise ValueError("recovery checkpoint predates its initial step")
    return {
        **config,
        "recovery_steps_completed": completed,
    }
