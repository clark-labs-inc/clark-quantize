"""Activation-local binary/ternary QAT for Qwen linear-attention projections.

This is a deliberately narrow experiment for the current AgentWorld blocker:
`linear_attention_projection` PTQ breaks teacher behavior, and binary is worse
than ternary. The script collects real FP16/BF16 activations at those projection
modules, trains hard-forward low-bit weights to preserve each module's local
linear map, then writes a safetensors patch for full-model teacher-cache eval.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant.master_weight_qat import ste_quantized_weight
from fake_quant_policy import _parse_csv, resolve_parameter_name
from weight_patch import apply_weight_patch


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def module_for_parameter(model: torch.nn.Module, parameter_name: str) -> torch.nn.Module | None:
    if not parameter_name.endswith(".weight"):
        return None
    module_name = parameter_name[: -len(".weight")]
    return dict(model.named_modules()).get(module_name)


def select_policy_choices(
    policy: dict[str, Any],
    *,
    bucket: str,
    source_rates: set[str] | None,
    max_modules: int | None,
) -> list[dict[str, Any]]:
    choices = []
    seen_tensors = set()
    for choice in policy["choices"]:
        meta = choice.get("meta") or {}
        tensor = meta.get("tensor")
        if not tensor or tensor in seen_tensors:
            continue
        if meta.get("bucket") != bucket:
            continue
        if source_rates is not None and choice["rate"] not in source_rates:
            continue
        choices.append(choice)
        seen_tensors.add(tensor)
        if max_modules is not None and len(choices) >= max_modules:
            break
    return choices


def load_cache_rows(path: Path, limit_rows: int | None) -> list[dict[str, Any]]:
    cache = torch.load(path, map_location="cpu")
    if cache.get("format") != "qwen_agentworld_teacher_behavior_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    rows = list(cache["rows"])
    return rows[:limit_rows] if limit_rows is not None else rows


def sample_rows(rows: list[dict[str, Any]], *, limit_rows: int | None, seed: int) -> list[dict[str, Any]]:
    rows = rows[:]
    if limit_rows is None or len(rows) <= limit_rows:
        return rows
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit_rows]


def token_sample(flat: torch.Tensor, need: int) -> torch.Tensor:
    if flat.shape[0] <= need:
        return flat
    index = torch.linspace(0, flat.shape[0] - 1, need, device=flat.device).round().long()
    return flat.index_select(0, index)


@torch.no_grad()
def collect_activation_banks(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    modules: dict[str, torch.nn.Module],
    *,
    tokens_per_module: int,
) -> dict[str, torch.Tensor]:
    banks: dict[str, list[torch.Tensor]] = defaultdict(list)
    counts = defaultdict(int)
    handles = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if counts[name] >= tokens_per_module or not inputs:
                return
            x = inputs[0].detach()
            if x.ndim < 2:
                return
            flat = x.reshape(-1, x.shape[-1])
            need = tokens_per_module - counts[name]
            chunk = token_sample(flat, need)
            banks[name].append(chunk.to(device="cpu", dtype=torch.bfloat16))
            counts[name] += int(chunk.shape[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        device = first_parameter_device(model)
        for row_index, row in enumerate(rows, start=1):
            input_ids = row["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0)
            attention_mask = row["attention_mask"].to(device=device, dtype=torch.long).unsqueeze(0)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1)
            ready = sum(1 for name in modules if counts[name] >= tokens_per_module)
            print(f"activation row {row_index}/{len(rows)} ready={ready}/{len(modules)}", flush=True)
            if ready == len(modules):
                break
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: torch.cat(chunks, dim=0)[:tokens_per_module].contiguous()
        for name, chunks in banks.items()
        if chunks
    }


def relative_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    err = (a.float() - b.float()).square().mean().sqrt()
    denom = a.float().square().mean().sqrt().clamp_min(1e-8)
    return float((err / denom).item())


def hard_quantized(master: torch.Tensor, mode: str, group_size: int) -> torch.Tensor:
    return ste_quantized_weight(master, mode=mode, group_size=group_size)


def output_metrics(x: torch.Tensor, original: torch.Tensor, quantized: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        target = F.linear(x.float(), original.float())
        pred = F.linear(x.float(), quantized.float())
    return {
        "output_mse": float((pred - target).square().mean().item()),
        "output_relative_rmse": relative_rmse(target, pred),
        "weight_relative_rmse": relative_rmse(original, quantized),
    }


def train_one_module(
    *,
    inputs_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    mode: str,
    group_size: int,
    steps: int,
    batch_tokens: int,
    lr: float,
    codebook_weight: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = inputs_cpu.to(device=device, dtype=torch.bfloat16)
    original = weight_cpu.to(device=device, dtype=torch.float32)
    master = torch.nn.Parameter(original.detach().clone())
    optimizer = torch.optim.AdamW([master], lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    initial_q = hard_quantized(master, mode, group_size).detach()
    initial = output_metrics(inputs[: min(inputs.shape[0], batch_tokens)], original, initial_q)
    last_loss = None
    for step in range(steps):
        if inputs.shape[0] <= batch_tokens:
            batch = inputs
        else:
            idx = torch.randint(inputs.shape[0], (batch_tokens,), generator=generator)
            batch = inputs.index_select(0, idx.to(inputs.device))
        with torch.no_grad():
            target = F.linear(batch.float(), original)
        quantized = hard_quantized(master, mode, group_size)
        pred = F.linear(batch.float(), quantized.float())
        fit_loss = (pred - target).square().mean()
        code_loss = (master - quantized.detach()).square().mean() / original.square().mean().clamp_min(1e-8)
        loss = fit_loss + codebook_weight * code_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([master], 1.0)
        optimizer.step()
        last_loss = float(loss.detach().item())

    final_q = hard_quantized(master, mode, group_size).detach()
    final = output_metrics(inputs[: min(inputs.shape[0], batch_tokens)], original, final_q)
    report = {
        "steps": steps,
        "batch_tokens": batch_tokens,
        "lr": lr,
        "codebook_weight": codebook_weight,
        "last_loss": last_loss,
        "initial": initial,
        "final": final,
    }
    return final_q.to(device="cpu", dtype=torch.bfloat16).contiguous(), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument(
        "--context-weight-patch",
        action="append",
        type=Path,
        help="apply a safetensors patch before collecting activations; repeatable",
    )
    parser.add_argument("--out-patch", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--bucket", default="linear_attention_projection")
    parser.add_argument("--source-rates", default="ternary40_log8")
    parser.add_argument("--mode", choices=["ternary", "binary"], default="ternary")
    parser.add_argument("--max-modules", type=int)
    parser.add_argument("--limit-rows", type=int, default=8)
    parser.add_argument("--tokens-per-module", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--codebook-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QAT probe")

    policy = json.loads(args.policy.read_text())
    group_size = int(policy.get("group_size", 112))
    choices = select_policy_choices(
        policy,
        bucket=args.bucket,
        source_rates=_parse_csv(args.source_rates),
        max_modules=args.max_modules,
    )
    if not choices:
        raise RuntimeError("no policy choices matched the requested bucket/rates")

    rows = sample_rows(
        load_cache_rows(args.cache, None),
        limit_rows=args.limit_rows,
        seed=args.seed,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    context_patch_results = []
    for patch_path in args.context_weight_patch or []:
        context_patch_results.append(apply_weight_patch(model, patch_path))

    params = dict(model.named_parameters())
    modules = {}
    weights_cpu = {}
    selected = []
    for choice in choices:
        meta = choice.get("meta") or {}
        tensor_name = str(meta["tensor"])
        parameter_name = resolve_parameter_name(tensor_name, params)
        if parameter_name is None:
            print(f"missing parameter for {tensor_name}", flush=True)
            continue
        module = module_for_parameter(model, parameter_name)
        if module is None:
            print(f"missing module for {parameter_name}", flush=True)
            continue
        modules[parameter_name] = module
        weights_cpu[parameter_name] = params[parameter_name].detach().cpu().contiguous()
        selected.append({"choice": choice, "parameter": parameter_name})

    if not modules:
        raise RuntimeError("no modules resolved")

    banks = collect_activation_banks(
        model,
        rows,
        modules,
        tokens_per_module=args.tokens_per_module,
    )
    del model
    torch.cuda.empty_cache()

    patch = {}
    reports = []
    device = torch.device("cuda")
    for index, item in enumerate(selected, start=1):
        parameter_name = item["parameter"]
        inputs = banks.get(parameter_name)
        if inputs is None or inputs.numel() == 0:
            reports.append({"parameter": parameter_name, "skipped": "no activation bank"})
            continue
        print(f"train {index}/{len(selected)} {parameter_name} tokens={inputs.shape[0]}", flush=True)
        quantized, report = train_one_module(
            inputs_cpu=inputs,
            weight_cpu=weights_cpu[parameter_name],
            mode=args.mode,
            group_size=group_size,
            steps=args.steps,
            batch_tokens=args.batch_tokens,
            lr=args.lr,
            codebook_weight=args.codebook_weight,
            device=device,
            seed=args.seed + index,
        )
        patch[parameter_name] = quantized
        reports.append(
            {
                "group": item["choice"]["group"],
                "parameter": parameter_name,
                "mode": args.mode,
                "bucket": (item["choice"].get("meta") or {}).get("bucket"),
                "params": int(weights_cpu[parameter_name].numel()),
                **report,
            }
        )

    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    save_file(patch, args.out_patch)
    result = {
        "model_dir": str(args.model_dir),
        "cache": str(args.cache),
        "policy": str(args.policy),
        "context_weight_patches": [str(path) for path in args.context_weight_patch or []],
        "context_patch_results": [
            {
                "patch": result.get("patch"),
                "applied_count": result.get("applied_count"),
                "missing_count": result.get("missing_count"),
            }
            for result in context_patch_results
        ],
        "patch": str(args.out_patch),
        "mode": args.mode,
        "bucket": args.bucket,
        "source_rates": args.source_rates,
        "group_size": group_size,
        "limit_rows": args.limit_rows,
        "tokens_per_module": args.tokens_per_module,
        "steps": args.steps,
        "batch_tokens": args.batch_tokens,
        "lr": args.lr,
        "codebook_weight": args.codebook_weight,
        "selected_modules": len(selected),
        "patched_modules": len(patch),
        "reports": reports,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"patch": str(args.out_patch), "patched_modules": len(patch)}, indent=2))


if __name__ == "__main__":
    main()
