"""Likelihood and teacher-KL calculations shared by the Gate B evaluator."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional


def causal_likelihood(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
) -> dict[str, float | int | None]:
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and ids [batch, sequence]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logit and token sequence shapes differ")
    if input_ids.shape[1] < 2:
        raise ValueError("at least two tokens are required")
    student = logits[:, :-1].float()
    labels = input_ids[:, 1:]
    token_count = labels.numel()
    nll_sum = functional.cross_entropy(
        student.reshape(-1, student.shape[-1]),
        labels.reshape(-1),
        reduction="sum",
    )
    teacher_kl_sum = None
    if teacher_logits is not None:
        if teacher_logits.shape != logits.shape:
            raise ValueError("student and teacher logit shapes differ")
        teacher = teacher_logits[:, :-1].float()
        teacher_log_probs = functional.log_softmax(teacher, dim=-1)
        student_log_probs = functional.log_softmax(student, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        teacher_kl_sum = (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum()
    return {
        "token_count": token_count,
        "nll_sum": float(nll_sum),
        "mean_nll": float(nll_sum) / token_count,
        "teacher_kl_sum": (
            float(teacher_kl_sum) if teacher_kl_sum is not None else None
        ),
        "mean_teacher_kl": (
            float(teacher_kl_sum) / token_count
            if teacher_kl_sum is not None
            else None
        ),
    }


def aggregate_likelihood(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate zero likelihood records")
    total_tokens = sum(int(record["token_count"]) for record in records)
    total_nll = sum(float(record["nll_sum"]) for record in records)
    has_kl = all(record["teacher_kl_sum"] is not None for record in records)
    if any(record["teacher_kl_sum"] is not None for record in records) != has_kl:
        raise ValueError("teacher KL is present for only part of the records")

    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[str(record["stratum"])].append(record)
    strata = []
    for label in sorted(by_stratum):
        group = by_stratum[label]
        tokens = sum(int(record["token_count"]) for record in group)
        nll = sum(float(record["nll_sum"]) for record in group) / tokens
        kl = (
            sum(float(record["teacher_kl_sum"]) for record in group) / tokens
            if has_kl
            else None
        )
        strata.append(
            {
                "stratum": label,
                "documents": len(group),
                "tokens": tokens,
                "mean_nll": nll,
                "perplexity": math.exp(min(nll, 80.0)),
                "mean_teacher_kl": kl,
            }
        )
    macro_nll = sum(row["mean_nll"] for row in strata) / len(strata)
    sorted_strata = sorted(strata, key=lambda row: row["mean_nll"], reverse=True)
    worst_count = max(1, math.ceil(len(strata) * 0.1))
    return {
        "documents": len(records),
        "tokens": total_tokens,
        "micro_mean_nll": total_nll / total_tokens,
        "micro_perplexity": math.exp(min(total_nll / total_tokens, 80.0)),
        "macro_language_mean_nll": macro_nll,
        "macro_language_perplexity": math.exp(min(macro_nll, 80.0)),
        "micro_mean_teacher_kl": (
            sum(float(record["teacher_kl_sum"]) for record in records)
            / total_tokens
            if has_kl
            else None
        ),
        "language_strata": strata,
        "worst_decile": {
            "count": worst_count,
            "mean_nll": sum(
                row["mean_nll"] for row in sorted_strata[:worst_count]
            )
            / worst_count,
            "strata": [row["stratum"] for row in sorted_strata[:worst_count]],
        },
    }
