#!/usr/bin/env python3
"""Unit tests for the Qwen 27B checkpoint-forensics helpers."""

from __future__ import annotations

import torch

from forensic_checkpoint_ablation import (
    projected_hidden_metrics,
    select_balanced_rows,
    summarize_metric_rows,
)


def main() -> None:
    rows = [
        {"task": "a", "id": f"a{i}", "tokens": 10 + i} for i in range(5)
    ] + [
        {"task": "b", "id": f"b{i}", "tokens": 20 + i} for i in range(3)
    ]
    selected = select_balanced_rows(rows, max_rows_per_task=2, seed=17, max_tokens=99)
    assert {row["task"] for row in selected} == {"a", "b"}
    assert sum(row["task"] == "a" for row in selected) == 2
    assert sum(row["task"] == "b" for row in selected) == 2

    perfect = projected_hidden_metrics(torch.eye(3), torch.eye(3))
    assert perfect["relative_mse"] == 0.0
    assert abs(perfect["cosine"] - 1.0) < 1e-6

    summary = summarize_metric_rows(
        [
            {
                "positions": 2,
                "teacher_topk_kl_full_mean": 1.0,
                "teacher_topk_kl_topk_mean": 0.2,
                "student_mass_on_teacher_topk_mean": 0.8,
                "top1_agreement": 0.5,
            },
            {
                "positions": 3,
                "teacher_topk_kl_full_mean": 3.0,
                "teacher_topk_kl_topk_mean": 0.4,
                "student_mass_on_teacher_topk_mean": 0.6,
                "top1_agreement": 1.0,
            },
        ]
    )
    assert summary["rows"] == 2
    assert summary["positions"] == 5
    assert summary["teacher_topk_kl_full_mean"] == 2.0
    assert summary["top1_agreement"] == 0.75
    print("forensic checkpoint ablation tests passed")


if __name__ == "__main__":
    main()
