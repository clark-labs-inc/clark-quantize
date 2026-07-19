"""Measure per-tensor low-bit damage on a cached teacher-behavior set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_teacher_behavior_cache import topk_teacher_metrics
from fake_quant_policy import (
    BINARY_RATES,
    TERNARY_RATES,
    _parse_csv,
    _quantized,
    resolve_parameter_name,
)
from weight_patch import apply_weight_patch


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def selected_choices(
    policy: dict[str, Any],
    *,
    bucket: str | None,
    rates: set[str] | None,
    parameters: set[str] | None,
    max_groups: int | None,
) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for choice in policy["choices"]:
        meta = choice.get("meta") or {}
        tensor = meta.get("tensor")
        if not tensor or tensor in seen:
            continue
        if bucket is not None and meta.get("bucket") != bucket:
            continue
        if rates is not None and choice["rate"] not in rates:
            continue
        if parameters is not None and tensor not in parameters:
            continue
        out.append(choice)
        seen.add(tensor)
        if max_groups is not None and len(out) >= max_groups:
            break
    return out


def load_parameters_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {str(item) for item in data}
    if isinstance(data, dict):
        values = data.get("parameters") or data.get("excluded") or data.get("tensors")
        if isinstance(values, list):
            return {str(item) for item in values}
    raise ValueError(f"unsupported parameters file shape: {path}")


@torch.no_grad()
def run_logits(model: torch.nn.Module, row: dict[str, Any], device: torch.device) -> torch.Tensor:
    input_ids = row["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = row["attention_mask"].to(device=device, dtype=torch.long).unsqueeze(0)
    keep = int(row["top_values"].shape[0])
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=keep,
    ).logits[0].detach().float().cpu()


def behavior_delta_metrics(
    baseline_logits: list[torch.Tensor],
    candidate_logits: list[torch.Tensor],
    *,
    temperature: float,
) -> dict[str, float]:
    kls = []
    top1 = []
    mse = []
    for base, cand in zip(baseline_logits, candidate_logits):
        base_t = base.float() / temperature
        cand_t = cand.float() / temperature
        base_logp = F.log_softmax(base_t, dim=-1)
        cand_logp = F.log_softmax(cand_t, dim=-1)
        base_prob = base_logp.exp()
        kls.append(F.kl_div(cand_logp, base_prob, reduction="batchmean").item() * (temperature**2))
        top1.append((cand_t.argmax(dim=-1) == base_t.argmax(dim=-1)).float().mean().item())
        mse.append((cand_t - base_t).square().mean().item())
    return {
        "delta_kl_mean": float(sum(kls) / max(len(kls), 1)),
        "delta_top1_agreement": float(sum(top1) / max(len(top1), 1)),
        "delta_logit_mse_mean": float(sum(mse) / max(len(mse), 1)),
    }


def summarize_teacher_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "teacher_topk_kl_full_mean": sum(row["teacher_topk_kl_full_mean"] for row in rows) / max(len(rows), 1),
        "teacher_topk_kl_topk_mean": sum(row["teacher_topk_kl_topk_mean"] for row in rows) / max(len(rows), 1),
        "student_mass_on_teacher_topk_mean": sum(row["student_mass_on_teacher_topk_mean"] for row in rows)
        / max(len(rows), 1),
        "top1_agreement": sum(row["top1_agreement"] for row in rows) / max(len(rows), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bucket")
    parser.add_argument("--rates", default="ternary40_fp16")
    parser.add_argument("--measure-rate", default="ternary40_fp16")
    parser.add_argument("--parameters-file", type=Path)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--weight-patch", action="append", type=Path)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    allowed_measure_rates = TERNARY_RATES | BINARY_RATES
    if args.measure_rate not in allowed_measure_rates:
        raise ValueError(f"unsupported --measure-rate {args.measure_rate!r}")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    cache = torch.load(args.cache, map_location="cpu")
    if cache.get("format") != "qwen_agentworld_teacher_behavior_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for patch in args.weight_patch or []:
        result = apply_weight_patch(model, patch)
        print(f"weight_patch {patch} applied={result['applied_count']} missing={result['missing_count']}", flush=True)

    params = dict(model.named_parameters())
    policy = json.loads(args.policy.read_text())
    group_size = int(policy.get("group_size", 112))
    choices = selected_choices(
        policy,
        bucket=args.bucket,
        rates=_parse_csv(args.rates),
        parameters=load_parameters_filter(args.parameters_file),
        max_groups=args.max_groups,
    )
    if not choices:
        raise RuntimeError("no policy choices selected")

    resolved = []
    for choice in choices:
        meta = choice.get("meta") or {}
        tensor_name = str(meta["tensor"])
        parameter_name = resolve_parameter_name(tensor_name, params)
        if parameter_name is None:
            print(f"missing {tensor_name}", flush=True)
            continue
        resolved.append((choice, tensor_name, parameter_name, params[parameter_name]))
    if not resolved:
        raise RuntimeError("no selected choices resolved to model parameters")

    device = first_parameter_device(model)
    rows = list(cache["rows"])
    baseline_logits = [run_logits(model, row, device) for row in rows]

    results = []
    for index, (choice, tensor_name, parameter_name, param) in enumerate(resolved, start=1):
        original = param.data.detach().clone()
        param.data.copy_(_quantized(original, args.measure_rate, group_size))
        candidate_logits = [run_logits(model, row, device) for row in rows]
        param.data.copy_(original)

        teacher_rows = []
        for logits, row in zip(candidate_logits, rows):
            teacher_rows.append(
                topk_teacher_metrics(
                    logits.to(device),
                    row["top_values"],
                    row["top_indices"],
                    temperature=args.temperature,
                )
            )
        meta = choice.get("meta") or {}
        out_row = {
            "group": choice["group"],
            "tensor": tensor_name,
            "parameter": parameter_name,
            "policy_rate": choice["rate"],
            "measure_rate": args.measure_rate,
            "bucket": meta.get("bucket"),
            "params": int(meta.get("params", param.numel())),
            **behavior_delta_metrics(baseline_logits, candidate_logits, temperature=args.temperature),
            **summarize_teacher_rows(teacher_rows),
        }
        results.append(out_row)
        print(
            f"[{index}/{len(resolved)}] {parameter_name} "
            f"delta_kl={out_row['delta_kl_mean']:.6g} "
            f"teacher_kl={out_row['teacher_topk_kl_full_mean']:.6g}",
            flush=True,
        )

    results.sort(key=lambda row: row["delta_kl_mean"], reverse=True)
    payload = {
        "model_dir": str(args.model_dir),
        "cache": str(args.cache),
        "policy": str(args.policy),
        "bucket": args.bucket,
        "rates": args.rates,
        "measure_rate": args.measure_rate,
        "parameters_file": str(args.parameters_file) if args.parameters_file else None,
        "weight_patches": [str(path) for path in args.weight_patch or []],
        "rows": results,
        "summary": {
            "groups": len(results),
            "avg_delta_kl": sum(row["delta_kl_mean"] for row in results) / max(len(results), 1),
            "max_delta_kl": max([row["delta_kl_mean"] for row in results] or [0.0]),
            "avg_teacher_kl": sum(row["teacher_topk_kl_full_mean"] for row in results) / max(len(results), 1),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
