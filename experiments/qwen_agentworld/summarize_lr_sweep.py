"""Summarize hard-forward Qwen QAT LR sweep reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


KEYS = [
    "loss",
    "kl",
    "weighted_kl",
    "top1",
    "mass",
    "teacher_top_mass",
    "student_top_mass",
    "code",
    "master_delta_rms",
    "master_delta_mean_abs",
    "master_delta_moved_frac",
    "master_delta_diffusion_expected_rms",
    "master_delta_diffusion_ratio",
    "code_changed_frac",
    "code_support_added_frac",
    "code_support_dropped_frac",
    "code_scale_ratio_mean",
    "code_scale_ratio_std",
    "code_persistence_frac",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def read_train_log(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(errors="replace").splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            continue
        if "step" in record and any(key in record for key in ("kl", "loss", "top1")):
            records.append(record)
    return records


def parse_lr_from_arm_name(name: str) -> float | None:
    text = name.removeprefix("lr_").replace("_", ".")
    if "em" in text:
        text = text.replace("em", "e-", 1)
    elif "ep" in text:
        text = text.replace("ep", "e+", 1)
    try:
        return float(text)
    except ValueError:
        return None


def read_sweep_config(path: Path) -> dict[str, Any]:
    config = read_json(path / "sweep_config.json")
    return config or {}


def infer_diffusion_ratio(
    record: dict[str, Any],
    *,
    lr: float | None,
    max_update_rms: float | None,
) -> dict[str, Any]:
    if "master_delta_diffusion_ratio" in record:
        return record
    if lr is None or max_update_rms is None or max_update_rms <= 0:
        return record
    if "master_delta_rms" not in record or "step" not in record:
        return record
    try:
        lr_scale = float(record.get("lr_scale", 1.0))
        step = float(record["step"])
        master_delta_rms = float(record["master_delta_rms"])
    except (TypeError, ValueError):
        return record
    expected = lr * lr_scale * max_update_rms * math.sqrt(max(step, 0.0))
    if expected <= 0 or not math.isfinite(expected):
        return record
    enriched = dict(record)
    enriched["cap_step_rms"] = lr * lr_scale * max_update_rms
    enriched["master_delta_diffusion_expected_rms"] = expected
    enriched["master_delta_diffusion_ratio"] = master_delta_rms / expected
    return enriched


def linear_slope(records: list[dict[str, Any]], key: str) -> float | None:
    points = [
        (float(record["step"]), float(record[key]))
        for record in records
        if key in record and record.get("step") is not None and math.isfinite(float(record[key]))
    ]
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom <= 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom


def pick_metrics(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        key: record[key]
        for key in KEYS
        if key in record
    } | {"step": record.get("step"), "task": record.get("task"), "tokens": record.get("tokens")}


def summarize_arm(path: Path, sweep_config: dict[str, Any]) -> dict[str, Any]:
    report = read_json(path / "report.json")
    history = (report or {}).get("history") or []
    source = "report"
    if not history:
        history = read_train_log(path / "train.log")
        source = "train_log" if history else "none"
    lr = None
    if report is not None:
        try:
            lr = float(report.get("lr"))
        except (TypeError, ValueError):
            lr = None
    if lr is None:
        lr = parse_lr_from_arm_name(path.name)
    try:
        max_update_rms = float((report or {}).get("max_update_rms", sweep_config.get("max_update_rms")))
    except (TypeError, ValueError):
        max_update_rms = None
    history = [
        infer_diffusion_ratio(record, lr=lr, max_update_rms=max_update_rms)
        for record in history
    ]

    first = history[0] if history else None
    last = history[-1] if history else None
    best_kl = min((record for record in history if "kl" in record), key=lambda r: float(r["kl"]), default=None)
    best_top1 = max((record for record in history if "top1" in record), key=lambda r: float(r["top1"]), default=None)
    token_values = [
        int(record["tokens"])
        for record in history
        if record.get("tokens") is not None
    ]
    avg_tokens = sum(token_values) / len(token_values) if token_values else None
    max_train_tokens = sweep_config.get("max_train_tokens") or (report or {}).get("max_train_tokens")
    token_fill = (
        avg_tokens / float(max_train_tokens)
        if avg_tokens is not None and max_train_tokens
        else None
    )
    return {
        "arm": path.name,
        "path": str(path),
        "source": source,
        "complete": report is not None,
        "lr": lr,
        "max_update_rms": max_update_rms,
        "history_steps": len(history),
        "early_stop": (report or {}).get("early_stop"),
        "skip_final_save": (report or {}).get("skip_final_save"),
        "out_patch": (report or {}).get("out_patch"),
        "patch_saved": bool(report and not report.get("skip_final_save") and report.get("out_patch")),
        "avg_logged_tokens_per_step": avg_tokens,
        "token_fill_vs_max_train_tokens": token_fill,
        "total_logged_tokens": sum(token_values) if token_values else None,
        "first": pick_metrics(first),
        "last": pick_metrics(last),
        "best_kl": pick_metrics(best_kl),
        "best_top1": pick_metrics(best_top1),
        "slopes_per_step": {
            "kl": linear_slope(history, "kl"),
            "top1": linear_slope(history, "top1"),
            "mass": linear_slope(history, "mass"),
            "code_changed_frac": linear_slope(history, "code_changed_frac"),
            "master_delta_rms": linear_slope(history, "master_delta_rms"),
            "master_delta_diffusion_ratio": linear_slope(history, "master_delta_diffusion_ratio"),
        },
    }


def choose_best(arms: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [arm for arm in arms if arm.get("last")]
    if not completed:
        return None
    return min(
        completed,
        key=lambda arm: (
            float((arm.get("last") or {}).get("kl", float("inf"))),
            -float((arm.get("last") or {}).get("top1", 0.0)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    sweep_config = read_sweep_config(args.sweep_dir)

    arms = [
        summarize_arm(path, sweep_config)
        for path in sorted(args.sweep_dir.glob("lr_*"))
        if path.is_dir()
    ]
    summary = {
        "sweep_dir": str(args.sweep_dir),
        "arms": arms,
        "best_by_last_kl": choose_best(arms),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
