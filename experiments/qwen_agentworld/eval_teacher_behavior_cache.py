"""Evaluate a BF16 or packed student against a cached teacher behavior file."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake_quant_policy import apply_fake_quant_policy
from packed_manifest_loader import apply_packed_manifest, parse_rate_scales
from weight_patch import apply_weight_patch


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def topk_teacher_metrics(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    *,
    temperature: float,
) -> dict[str, float]:
    """Compare student full logits to teacher top-k logits.

    The KL uses the teacher distribution renormalized over its top-k tokens, but
    scores student probabilities under the full vocabulary normalization. This
    penalizes student probability mass that leaves the teacher's high-probability
    support while staying cheap enough to cache.
    """

    if temperature <= 0:
        raise ValueError("--temperature must be positive")

    positions = int(teacher_values.shape[0])
    student = student_logits[-positions:, :].float() / temperature
    teacher_values = teacher_values.to(student.device, dtype=torch.float32) / temperature
    teacher_indices = teacher_indices.to(student.device, dtype=torch.long)

    student_gather = student.gather(dim=-1, index=teacher_indices)
    student_logp_full = student_gather - torch.logsumexp(student, dim=-1, keepdim=True)
    student_logp_topk = F.log_softmax(student_gather, dim=-1)
    teacher_logp = F.log_softmax(teacher_values, dim=-1)
    teacher_prob = teacher_logp.exp()

    kl_full = (teacher_prob * (teacher_logp - student_logp_full)).sum(dim=-1)
    kl_topk = (teacher_prob * (teacher_logp - student_logp_topk)).sum(dim=-1)
    topk_mass = student_logp_full.exp().sum(dim=-1)
    top1_agreement = (student.argmax(dim=-1) == teacher_indices[:, 0]).float()
    return {
        "teacher_topk_kl_full_mean": float((kl_full * (temperature**2)).mean().item()),
        "teacher_topk_kl_topk_mean": float((kl_topk * (temperature**2)).mean().item()),
        "student_mass_on_teacher_topk_mean": float(topk_mass.mean().item()),
        "top1_agreement": float(top1_agreement.mean().item()),
        "positions": positions,
    }


def topk_logprob_other_metrics(
    student_logits: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_indices: torch.Tensor,
    *,
    temperature: float,
) -> dict[str, float]:
    """Compare student logits to teacher top-k full-distribution logprobs.

    vLLM dense caches store prompt log-probabilities, not raw teacher logits.
    This computes the same coarsened objective as training: top-k teacher
    probabilities plus one aggregate "other" bucket for the tail.
    """

    if temperature <= 0:
        raise ValueError("--temperature must be positive")

    positions = int(teacher_logprobs.shape[0])
    student = student_logits[-positions:, :].float() / temperature
    teacher_logprobs = teacher_logprobs.to(student.device, dtype=torch.float32)
    teacher_indices = teacher_indices.to(student.device, dtype=torch.long)

    student_gather = student.gather(dim=-1, index=teacher_indices)
    student_logp_full = student_gather - torch.logsumexp(student, dim=-1, keepdim=True)
    student_prob = student_logp_full.exp()
    student_top_mass = student_prob.sum(dim=-1).clamp(max=1.0)
    student_other = (1.0 - student_top_mass).clamp_min(1e-8)

    teacher_prob = teacher_logprobs.exp()
    teacher_top_mass = teacher_prob.sum(dim=-1).clamp(max=1.0)
    teacher_other = (1.0 - teacher_top_mass).clamp_min(1e-8)

    top_term = (teacher_prob * (teacher_logprobs - student_logp_full)).sum(dim=-1)
    other_term = teacher_other * (teacher_other.log() - student_other.log())
    kl_full = top_term + other_term
    top1_agreement = (student.argmax(dim=-1) == teacher_indices[:, 0]).float()
    return {
        "teacher_topk_kl_full_mean": float((kl_full * (temperature**2)).mean().item()),
        "teacher_topk_kl_topk_mean": None,
        "student_mass_on_teacher_topk_mean": float(student_top_mass.mean().item()),
        "teacher_topk_mass_mean": float(teacher_top_mass.mean().item()),
        "other_bucket_mean": float(student_other.mean().item()),
        "top1_agreement": float(top1_agreement.mean().item()),
        "positions": positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--policy", type=Path, help="apply a mixed-rate policy as in-place fake quant before eval")
    parser.add_argument("--weight-patch", type=Path, help="safetensors patch to copy before manifest/policy eval")
    parser.add_argument("--weight-patch-exclude-regex")
    parser.add_argument("--include-experts", action="store_true")
    parser.add_argument("--rates")
    parser.add_argument("--buckets")
    parser.add_argument(
        "--scale-rate",
        action="append",
        help="multiply reconstructed groups for one rate, e.g. ternary40_log8=0.75; repeatable",
    )
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--manifest-report", type=Path)
    parser.add_argument("--policy-report", type=Path)
    parser.add_argument("--weight-patch-report", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-rows-per-task", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--shuffle-rows", action="store_true")
    parser.add_argument("--row-seed", type=int, default=17)
    parser.add_argument(
        "--task-filter",
        action="append",
        help="evaluate only rows with this task after shuffle/start-row; repeatable or comma-separated",
    )
    args = parser.parse_args()

    if args.manifest is not None and args.policy is not None:
        raise ValueError("use either --manifest or --policy, not both")
    if args.policy is not None and args.scale_rate:
        raise ValueError("--scale-rate is only supported with --manifest")

    cache = torch.load(args.cache, map_location="cpu")
    if cache.get("format") not in {"qwen_agentworld_teacher_behavior_v1", "qwen_onpolicy_survival_cache_v1"}:
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    manifest_result = None
    policy_result = None
    weight_patch_result = None
    scale_by_rate = parse_rate_scales(args.scale_rate)
    if args.weight_patch is not None:
        weight_patch_result = apply_weight_patch(
            model,
            args.weight_patch,
            out_path=args.weight_patch_report,
            exclude_regex=args.weight_patch_exclude_regex,
        )
    if args.manifest is not None:
        manifest_result = apply_packed_manifest(
            model,
            args.manifest,
            max_groups=args.max_groups,
            include_experts=args.include_experts,
            rates=args.rates,
            buckets=args.buckets,
            scale_by_rate=scale_by_rate,
            out_path=args.manifest_report,
        )
    if args.policy is not None:
        policy_result = apply_fake_quant_policy(
            model,
            args.policy,
            max_groups=args.max_groups,
            include_experts=args.include_experts,
            rates=args.rates,
            buckets=args.buckets,
            out_path=args.policy_report,
        )

    device = first_parameter_device(model)
    rows: list[dict[str, Any]] = []
    selected_rows = list(cache["rows"])
    if args.max_tokens is not None:
        selected_rows = [row for row in selected_rows if int(row.get("tokens") or 0) <= args.max_tokens]
    if args.shuffle_rows:
        rng = random.Random(args.row_seed)
        rng.shuffle(selected_rows)
    if args.start_row > 0:
        selected_rows = selected_rows[args.start_row :]
    task_filters: set[str] = set()
    for value in args.task_filter or []:
        task_filters.update(part.strip() for part in value.split(",") if part.strip())
    if task_filters:
        selected_rows = [row for row in selected_rows if str(row.get("task")) in task_filters]
    if args.max_rows_per_task is not None:
        counts_by_task: dict[str, int] = {}
        balanced_rows = []
        for row in selected_rows:
            task = str(row.get("task"))
            count = counts_by_task.get(task, 0)
            if count >= args.max_rows_per_task:
                continue
            balanced_rows.append(row)
            counts_by_task[task] = count + 1
        selected_rows = balanced_rows
    if args.max_rows is not None:
        selected_rows = selected_rows[: args.max_rows]
    for row in selected_rows:
        input_ids = row["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0)
        attention_mask = row["attention_mask"].to(device=device, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            teacher_values = row.get("top_values")
            teacher_logprobs = row.get("teacher_top_logprobs")
            if isinstance(teacher_values, torch.Tensor):
                keep = int(teacher_values.shape[0])
            elif isinstance(teacher_logprobs, torch.Tensor):
                keep = int(teacher_logprobs.shape[0])
            else:
                raise RuntimeError("cache row must contain top_values or teacher_top_logprobs")
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=keep,
            ).logits[0]
        if isinstance(teacher_logprobs, torch.Tensor):
            metrics = topk_logprob_other_metrics(
                logits,
                teacher_logprobs,
                row["top_indices"],
                temperature=args.temperature,
            )
        else:
            metrics = topk_teacher_metrics(
                logits,
                teacher_values,
                row["top_indices"],
                temperature=args.temperature,
            )
        out_row = {
            "task": row.get("task"),
            "id": row.get("id"),
            "tokens": int(row["tokens"]),
            **metrics,
        }
        rows.append(out_row)
        print(json.dumps(out_row), flush=True)

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(subset),
            "teacher_topk_kl_full_mean": sum(row["teacher_topk_kl_full_mean"] for row in subset)
            / max(len(subset), 1),
            "teacher_topk_kl_topk_mean": (
                sum(row["teacher_topk_kl_topk_mean"] for row in subset if row["teacher_topk_kl_topk_mean"] is not None)
                / max(sum(1 for row in subset if row["teacher_topk_kl_topk_mean"] is not None), 1)
            ),
            "student_mass_on_teacher_topk_mean": sum(row["student_mass_on_teacher_topk_mean"] for row in subset)
            / max(len(subset), 1),
            "teacher_topk_mass_mean": (
                sum(row.get("teacher_topk_mass_mean", 0.0) for row in subset)
                / max(sum(1 for row in subset if "teacher_topk_mass_mean" in row), 1)
            ),
            "top1_agreement": sum(row["top1_agreement"] for row in subset) / max(len(subset), 1),
            "positions": sum(int(row.get("positions") or 0) for row in subset),
            "tokens": sum(int(row.get("tokens") or 0) for row in subset),
        }

    summary = summarize(rows)
    summary_by_task: dict[str, Any] = {}
    for task in sorted({str(row.get("task")) for row in rows}):
        task_rows = [row for row in rows if str(row.get("task")) == task]
        summary_by_task[task] = summarize(task_rows)
    legacy_summary = {
        "rows": len(rows),
        "teacher_topk_kl_full_mean": sum(row["teacher_topk_kl_full_mean"] for row in rows) / max(len(rows), 1),
        "teacher_topk_kl_topk_mean": (
            sum(row["teacher_topk_kl_topk_mean"] for row in rows if row["teacher_topk_kl_topk_mean"] is not None)
            / max(sum(1 for row in rows if row["teacher_topk_kl_topk_mean"] is not None), 1)
        ),
        "student_mass_on_teacher_topk_mean": sum(row["student_mass_on_teacher_topk_mean"] for row in rows)
        / max(len(rows), 1),
        "teacher_topk_mass_mean": (
            sum(row.get("teacher_topk_mass_mean", 0.0) for row in rows)
            / max(sum(1 for row in rows if "teacher_topk_mass_mean" in row), 1)
        ),
        "top1_agreement": sum(row["top1_agreement"] for row in rows) / max(len(rows), 1),
    }
    summary.update(legacy_summary)
    result = {
        "model_dir": str(args.model_dir),
        "cache": str(args.cache),
        "manifest": str(args.manifest) if args.manifest else None,
        "policy": str(args.policy) if args.policy else None,
        "weight_patch": str(args.weight_patch) if args.weight_patch else None,
        "weight_patch_exclude_regex": args.weight_patch_exclude_regex,
        "include_experts": args.include_experts,
        "rates": args.rates,
        "buckets": args.buckets,
        "scale_by_rate": scale_by_rate,
        "max_groups": args.max_groups,
        "temperature": args.temperature,
        "start_row": args.start_row,
        "max_rows": args.max_rows,
        "max_rows_per_task": args.max_rows_per_task,
        "max_tokens": args.max_tokens,
        "shuffle_rows": args.shuffle_rows,
        "row_seed": args.row_seed,
        "task_filter": sorted(task_filters),
        "manifest_result": (
            {
                "applied_count": manifest_result.get("applied_count"),
                "missing_count": manifest_result.get("missing_count"),
                "report": str(args.manifest_report) if args.manifest_report else None,
            }
            if manifest_result is not None
            else None
        ),
        "weight_patch_result": (
            {
                "applied_count": weight_patch_result.get("applied_count"),
                "missing_count": weight_patch_result.get("missing_count"),
                "report": str(args.weight_patch_report) if args.weight_patch_report else None,
            }
            if weight_patch_result is not None
            else None
        ),
        "policy_result": (
            {
                "applied_count": policy_result.get("applied_count"),
                "missing_count": policy_result.get("missing_count"),
                "report": str(args.policy_report) if args.policy_report else None,
            }
            if policy_result is not None
            else None
        ),
        "rows_detail": rows,
        "summary": summary,
        "summary_by_task": summary_by_task,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), **summary}, indent=2))


if __name__ == "__main__":
    main()
