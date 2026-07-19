"""Measure bounded teacher-behavior damage for Qwen AgentWorld policy groups.

This is a local, non-judge calibration pass. By default it compares BF16
teacher logits against the same loaded model after applying one binary/ternary
policy group. It can also apply a weight patch and a base low-bit policy first,
then measure candidate groups in that context.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake_quant_policy import _parse_csv, _quantized, apply_fake_quant_policy, resolve_parameter_name
from weight_patch import apply_weight_patch


QUANT_RATES = {"ternary40_log8", "binary_log8"}


def load_jobs(data_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    jobs = []
    for path in sorted(data_dir.glob("*_test.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                jobs.append(json.loads(line))
                if limit is not None and len(jobs) >= limit:
                    return jobs
    return jobs


def messages_for_job(job: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = job.get("system_str", "")
    current_prompt = job.get("current_prompt", "")
    if not current_prompt:
        prompts = job.get("prompt", [])
        turn_idx = int(job.get("turn_idx", 1)) - 1
        if isinstance(prompts, list) and turn_idx < len(prompts):
            current_prompt = prompts[turn_idx]
        elif isinstance(prompts, str):
            current_prompt = prompts

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": current_prompt})
    return messages


def choice_allowed(
    choice: dict[str, Any],
    *,
    include_experts: bool,
    rates: set[str] | None,
    buckets: set[str] | None,
    max_params: int | None,
    min_route_fraction: float | None,
    max_route_fraction: float | None,
) -> bool:
    meta = choice.get("meta") or {}
    if "::expert_" in choice["group"] and not include_experts:
        return False
    if rates is not None and choice["rate"] not in rates:
        return False
    if buckets is not None and meta.get("bucket") not in buckets:
        return False
    if max_params is not None and int(meta.get("params", 0)) > max_params:
        return False
    route_fraction = meta.get("route_fraction")
    if route_fraction is not None:
        route_fraction = float(route_fraction)
        if min_route_fraction is not None and route_fraction < min_route_fraction:
            return False
        if max_route_fraction is not None and route_fraction > max_route_fraction:
            return False
    return True


def iter_choices(
    policy: dict[str, Any],
    *,
    include_experts: bool,
    rates: set[str] | None,
    buckets: set[str] | None,
    max_groups: int | None,
    max_params: int | None,
    sort_by: str,
    min_route_fraction: float | None,
    max_route_fraction: float | None,
) -> Iterable[dict[str, Any]]:
    choices = [
        choice
        for choice in policy["choices"]
        if choice_allowed(
            choice,
            include_experts=include_experts,
            rates=rates,
            buckets=buckets,
            max_params=max_params,
            min_route_fraction=min_route_fraction,
            max_route_fraction=max_route_fraction,
        )
    ]
    if sort_by == "route_fraction_desc":
        choices.sort(key=lambda row: float((row.get("meta") or {}).get("route_fraction") or 0.0), reverse=True)
    elif sort_by == "route_fraction_asc":
        choices.sort(key=lambda row: float((row.get("meta") or {}).get("route_fraction") or 0.0))
    elif sort_by == "damage_desc":
        choices.sort(key=lambda row: float(row.get("damage") or 0.0), reverse=True)
    elif sort_by == "damage_asc":
        choices.sort(key=lambda row: float(row.get("damage") or 0.0))
    elif sort_by == "params_desc":
        choices.sort(key=lambda row: int((row.get("meta") or {}).get("params") or 0), reverse=True)

    if max_groups is not None:
        choices = choices[:max_groups]
    for choice in choices:
        yield choice


def selected_logits(logits: torch.Tensor, last_n: int) -> torch.Tensor:
    return logits[:, -last_n:, :].detach().float().cpu()


@torch.no_grad()
def run_logits(model: torch.nn.Module, inputs: dict[str, torch.Tensor], last_n: int) -> torch.Tensor:
    outputs = model(**inputs, use_cache=False)
    return selected_logits(outputs.logits, min(last_n, outputs.logits.shape[1]))


def behavior_metrics(base_logits: list[torch.Tensor], cand_logits: list[torch.Tensor], temperature: float) -> dict[str, float]:
    kls = []
    mse = []
    top1 = []
    max_abs = []
    for base, cand in zip(base_logits, cand_logits):
        base = base.float() / temperature
        cand = cand.float() / temperature
        base_logp = F.log_softmax(base, dim=-1)
        cand_logp = F.log_softmax(cand, dim=-1)
        base_prob = base_logp.exp()
        kls.append(F.kl_div(cand_logp, base_prob, reduction="batchmean").item() * (temperature**2))
        mse.append((cand - base).square().mean().item())
        max_abs.append((cand - base).abs().max().item())
        top1.append((cand.argmax(dim=-1) == base.argmax(dim=-1)).float().mean().item())
    return {
        "kl_mean": float(sum(kls) / max(len(kls), 1)),
        "logit_mse_mean": float(sum(mse) / max(len(mse), 1)),
        "max_abs_logit_diff": float(max(max_abs) if max_abs else 0.0),
        "top1_agreement": float(sum(top1) / max(len(top1), 1)),
    }


def target_slice(param: torch.nn.Parameter, choice: dict[str, Any]) -> torch.Tensor:
    meta = choice.get("meta") or {}
    if "expert" in meta:
        return param.data[int(meta["expert"])]
    return param.data


def relative_rmse(before: torch.Tensor, after: torch.Tensor) -> float:
    err = (after.float().cpu() - before.float().cpu()).square().mean().sqrt()
    denom = before.float().cpu().square().mean().sqrt().clamp_min(1e-8)
    return float((err / denom).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--data-dir", default="/workspace/AgentWorldBench_smoke_1x7", type=Path)
    parser.add_argument("--policy", default="/workspace/agentworld_runs/policy_qwen_agentworld_8x_router_smoke512.json", type=Path)
    parser.add_argument("--out", default="/workspace/agentworld_runs/group_damage_smoke.json", type=Path)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit-jobs", type=int, default=3)
    parser.add_argument("--last-n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-groups", type=int, default=16)
    parser.add_argument("--max-params", type=int, default=2_200_000)
    parser.add_argument("--include-experts", action="store_true")
    parser.add_argument("--rates")
    parser.add_argument(
        "--measure-rates",
        help="comma-separated quantized rates to measure for each selected policy group; defaults to the group's policy rate",
    )
    parser.add_argument("--buckets")
    parser.add_argument(
        "--sort-by",
        default="policy",
        choices=["policy", "route_fraction_desc", "route_fraction_asc", "damage_desc", "damage_asc", "params_desc"],
        help="selection order before --max-groups is applied",
    )
    parser.add_argument("--min-route-fraction", type=float)
    parser.add_argument("--max-route-fraction", type=float)
    parser.add_argument("--weight-patch", type=Path)
    parser.add_argument("--weight-patch-report", type=Path)
    parser.add_argument("--base-policy", type=Path, help="apply this policy before computing baseline logits")
    parser.add_argument("--base-policy-report", type=Path)
    parser.add_argument("--base-max-groups", type=int)
    parser.add_argument("--base-include-experts", action="store_true")
    parser.add_argument("--base-rates")
    parser.add_argument("--base-buckets")
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    # AgentWorld prompts often exceed the calibration window. Preserve the tail,
    # which contains the current observation/request, instead of repeatedly
    # measuring the shared domain/system prefix.
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    policy = json.loads(args.policy.read_text())
    group_size = int(policy.get("group_size", 112))
    params = dict(model.named_parameters())
    choices = list(
        iter_choices(
            policy,
            include_experts=args.include_experts,
            rates=_parse_csv(args.rates),
            buckets=_parse_csv(args.buckets),
            max_groups=args.max_groups,
            max_params=args.max_params,
            sort_by=args.sort_by,
            min_route_fraction=args.min_route_fraction,
            max_route_fraction=args.max_route_fraction,
        )
    )
    source_by_group: dict[str, torch.Tensor] = {}
    for choice in choices:
        meta = choice.get("meta") or {}
        tensor_name = meta.get("tensor")
        parameter_name = resolve_parameter_name(str(tensor_name), params) if tensor_name else None
        if parameter_name is None:
            continue
        source_by_group[choice["group"]] = target_slice(params[parameter_name], choice).detach().clone()

    patch_result = None
    base_policy_result = None
    if args.weight_patch is not None:
        patch_result = apply_weight_patch(model, args.weight_patch, out_path=args.weight_patch_report)
        print(
            f"weight_patch_applied count={patch_result['applied_count']} "
            f"missing={patch_result['missing_count']}",
            flush=True,
        )
    if args.base_policy is not None:
        base_policy_result = apply_fake_quant_policy(
            model,
            args.base_policy,
            max_groups=args.base_max_groups,
            include_experts=args.base_include_experts,
            rates=args.base_rates,
            buckets=args.base_buckets,
            out_path=args.base_policy_report,
        )
        print(
            f"base_policy_applied count={base_policy_result['applied_count']} "
            f"missing={base_policy_result['missing_count']}",
            flush=True,
        )

    first_device = next(model.parameters()).device
    jobs = load_jobs(args.data_dir, args.limit_jobs)
    inputs_list = []
    task_ids = []
    for job in jobs:
        text = tokenizer.apply_chat_template(messages_for_job(job), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs_list.append({key: value.to(first_device) for key, value in inputs.items()})
        task_ids.append({"task": job.get("task"), "id": job.get("id"), "tokens": int(inputs["input_ids"].shape[-1])})

    print(f"loaded calibration jobs={len(inputs_list)} max_length={args.max_length}", flush=True)
    baseline_logits = [run_logits(model, inputs, args.last_n) for inputs in inputs_list]

    rows = []
    requested_measure_rates = _parse_csv(args.measure_rates)
    total_measurements = 0
    for choice in choices:
        source_rates = requested_measure_rates or {choice["rate"]}
        for rate in sorted(source_rates):
            if rate not in QUANT_RATES:
                raise ValueError(f"unsupported measure rate {rate!r}; expected one of {sorted(QUANT_RATES)}")
            total_measurements += 1

    measurement_index = 0
    for choice in choices:
        meta = choice.get("meta") or {}
        tensor_name = meta.get("tensor")
        parameter_name = resolve_parameter_name(str(tensor_name), params) if tensor_name else None
        if parameter_name is None:
            rows.append({"group": choice["group"], "rate": choice["rate"], "missing": True})
            continue
        param = params[parameter_name]
        target = target_slice(param, choice)
        source_rates = requested_measure_rates or {choice["rate"]}
        for rate in sorted(source_rates):
            if rate not in QUANT_RATES:
                raise ValueError(f"unsupported measure rate {rate!r}; expected one of {sorted(QUANT_RATES)}")
            measurement_index += 1
            original = target.detach().clone()
            source = source_by_group.get(choice["group"], original).to(device=target.device, dtype=target.dtype)
            target.copy_(_quantized(source, rate, group_size))
            rmse = relative_rmse(original, target)
            source_rmse = relative_rmse(source, target)
            cand_logits = [run_logits(model, inputs, args.last_n) for inputs in inputs_list]
            target.copy_(original)
            metrics = behavior_metrics(baseline_logits, cand_logits, args.temperature)
            row = {
                "group": choice["group"],
                "tensor": tensor_name,
                "parameter": parameter_name,
                "policy_rate": choice["rate"],
                "rate": rate,
                "bucket": meta.get("bucket"),
                "expert": meta.get("expert"),
                "route_fraction": meta.get("route_fraction"),
                "params": int(meta.get("params", target.numel())),
                "relative_rmse": rmse,
                "source_relative_rmse": source_rmse,
                **metrics,
            }
            rows.append(row)
            print(
                f"[{measurement_index}/{total_measurements}] "
                f"{choice['group']} policy={choice['rate']} measured={rate} "
                f"kl={row['kl_mean']:.6g}",
                flush=True,
            )

    result = {
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "policy": str(args.policy),
        "weight_patch": str(args.weight_patch) if args.weight_patch else None,
        "base_policy": str(args.base_policy) if args.base_policy else None,
        "base_include_experts": args.base_include_experts,
        "base_rates": args.base_rates,
        "base_buckets": args.base_buckets,
        "weight_patch_result": (
            {
                "applied_count": patch_result.get("applied_count"),
                "missing_count": patch_result.get("missing_count"),
                "report": str(args.weight_patch_report) if args.weight_patch_report else None,
            }
            if patch_result is not None
            else None
        ),
        "base_policy_result": (
            {
                "applied_count": base_policy_result.get("applied_count"),
                "missing_count": base_policy_result.get("missing_count"),
                "report": str(args.base_policy_report) if args.base_policy_report else None,
            }
            if base_policy_result is not None
            else None
        ),
        "calibration_jobs": task_ids,
        "max_length": args.max_length,
        "last_n": args.last_n,
        "temperature": args.temperature,
        "selected_groups": len(choices),
        "measure_rates": sorted(requested_measure_rates) if requested_measure_rates else None,
        "sort_by": args.sort_by,
        "min_route_fraction": args.min_route_fraction,
        "max_route_fraction": args.max_route_fraction,
        "rows": rows,
        "summary": {
            "selected_groups": len(choices),
            "groups": len(rows),
            "missing": sum(1 for row in rows if row.get("missing")),
            "avg_kl": sum(row.get("kl_mean", 0.0) for row in rows) / max(sum(1 for row in rows if "kl_mean" in row), 1),
            "max_kl": max([row.get("kl_mean", 0.0) for row in rows] or [0.0]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
