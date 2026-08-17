"""Block-level on-policy survival QAT for Qwen AgentWorld failures."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant.master_weight_qat import ste_quantized_weight
from train_mlp_triplet_survival_qat import (
    QuantSwiGLUMLP,
    anti_collapse_loss,
    freeze_quant_mlp,
    get_parent_and_attr,
    load_model,
    parse_layers,
    register_hidden_hooks,
    replace_mlp_with_quant,
    row_to_inputs,
    topk_teacher_kl,
)
from train_all2d_ternary_qat import projected_or_full_hidden_loss
from weight_patch import apply_weight_patch


def atomic_save_patch(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_non_text_modules(model: torch.nn.Module) -> None:
    core = getattr(model, "model", model)
    for attribute in ("visual", "vision_model", "vision_tower"):
        if getattr(core, attribute, None) is not None:
            setattr(core, attribute, None)
    if getattr(model, "mtp", None) is not None:
        model.mtp = None
    gc.collect()
    torch.cuda.empty_cache()


def layer_prefix(layer_id: int) -> str:
    return f"model.language_model.layers.{layer_id}"


def mlp_weight_name(layer_id: int, proj: str) -> str:
    return f"{layer_prefix(layer_id)}.mlp.{proj}_proj.weight"


def available_projection_names(model: torch.nn.Module, layer_id: int) -> list[str]:
    prefix = layer_prefix(layer_id)
    candidates = [
        f"{prefix}.linear_attn.in_proj_a",
        f"{prefix}.linear_attn.in_proj_b",
        f"{prefix}.linear_attn.in_proj_qkv",
        f"{prefix}.linear_attn.in_proj_z",
        f"{prefix}.linear_attn.out_proj",
        f"{prefix}.self_attn.q_proj",
        f"{prefix}.self_attn.k_proj",
        f"{prefix}.self_attn.v_proj",
        f"{prefix}.self_attn.o_proj",
    ]
    modules = dict(model.named_modules())
    return [name for name in candidates if name in modules]


def available_norm_names(model: torch.nn.Module, layer_id: int) -> list[str]:
    prefix = layer_prefix(layer_id)
    candidates = [
        f"{prefix}.input_layernorm",
        f"{prefix}.post_attention_layernorm",
        f"{prefix}.linear_attn.norm",
        f"{prefix}.self_attn.q_norm",
        f"{prefix}.self_attn.k_norm",
    ]
    modules = dict(model.named_modules())
    return [name for name in candidates if name in modules and hasattr(modules[name], "weight")]


class QuantLinear(torch.nn.Module):
    def __init__(self, source: torch.nn.Linear, *, mode: str, group_size: int) -> None:
        super().__init__()
        self.master = torch.nn.Parameter(source.weight.detach().float().clone())
        self.mode = mode
        self.group_size = group_size
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def quantized(self) -> torch.Tensor:
        return ste_quantized_weight(self.master, mode=self.mode, group_size=self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, self.quantized().to(dtype=x.dtype), bias)

    def codebook_loss(self) -> torch.Tensor:
        with torch.no_grad():
            quantized = self.quantized().detach()
        denom = self.master.detach().float().square().mean().clamp_min(1e-8)
        return (self.master.float() - quantized.float()).square().mean() / denom

    @torch.no_grad()
    def hard_weight(self) -> torch.Tensor:
        return self.quantized().detach().cpu().to(torch.bfloat16)


def replace_linear_with_quant(
    model: torch.nn.Module,
    module_name: str,
    *,
    mode: str,
    group_size: int,
) -> QuantLinear:
    source = model.get_submodule(module_name)
    quant = QuantLinear(source, mode=mode, group_size=group_size).to(next(source.parameters()).device)
    parent, attr = get_parent_and_attr(model, module_name)
    setattr(parent, attr, quant)
    return quant


def freeze_quant_linear(model: torch.nn.Module, module_name: str, quant: QuantLinear) -> torch.Tensor:
    hard = quant.hard_weight()
    device = quant.master.device
    frozen = torch.nn.Linear(
        quant.master.shape[1],
        quant.master.shape[0],
        bias=quant.bias is not None,
        device=device,
        dtype=torch.bfloat16,
    )
    frozen.weight.data.copy_(hard.to(device=device, dtype=torch.bfloat16))
    if quant.bias is not None and frozen.bias is not None:
        frozen.bias.data.copy_(quant.bias.to(device=device, dtype=torch.bfloat16))
    for param in frozen.parameters():
        param.requires_grad_(False)
    parent, attr = get_parent_and_attr(model, module_name)
    setattr(parent, attr, frozen)
    return hard


def selected_rows(cache: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in cache["rows"] if int(row["tokens"]) <= args.max_train_tokens]
    if args.failures:
        wanted = {part.strip() for part in args.failures.split(",") if part.strip()}
        rows = [row for row in rows if row.get("failure_kind") in wanted or row.get("collapse", {}).get("failure_kind") in wanted]
    if args.shuffle_rows:
        random.Random(args.row_seed).shuffle(rows)
    if args.max_rows_per_task is not None:
        counts: dict[str, int] = {}
        balanced = []
        for row in rows:
            task = str(row.get("task"))
            if counts.get(task, 0) >= args.max_rows_per_task:
                continue
            balanced.append(row)
            counts[task] = counts.get(task, 0) + 1
        rows = balanced
    if args.max_train_rows is not None:
        rows = rows[: args.max_train_rows]
    if getattr(args, "interleave_tasks", False):
        task_order = list(dict.fromkeys(str(row.get("task")) for row in rows))
        by_task = {task: [row for row in rows if str(row.get("task")) == task] for task in task_order}
        interleaved = []
        offset = 0
        while True:
            appended = False
            for task in task_order:
                task_rows = by_task[task]
                if offset < len(task_rows):
                    interleaved.append(task_rows[offset])
                    appended = True
            if not appended:
                break
            offset += 1
        rows = interleaved
    if not rows:
        raise RuntimeError("no cache rows selected")
    return rows


def train_block(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    layer_id: int,
    args: argparse.Namespace,
    device: torch.device,
    hidden_layers: list[int],
    format_ids: torch.Tensor,
    collapse_ids: torch.Tensor,
    hidden_projection_dim: int,
    hidden_projection_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    quant_mlp = (
        replace_mlp_with_quant(
            model,
            layer_id,
            mode=args.mode,
            group_size=args.group_size,
            artifact_exact_ternary=args.artifact_exact_ternary,
        )
        if args.train_mlp
        else None
    )
    quant_linears: dict[str, QuantLinear] = {}
    if args.train_projections:
        for name in available_projection_names(model, layer_id):
            if args.projection_regex and not re.search(args.projection_regex, name):
                continue
            quant_linears[name] = replace_linear_with_quant(model, name, mode=args.mode, group_size=args.group_size)

    norm_params: list[torch.nn.Parameter] = []
    norm_names = available_norm_names(model, layer_id) if args.train_norms else []
    for name in norm_names:
        weight = model.get_submodule(name).weight
        weight.data = weight.data.float()
        weight.requires_grad_(True)
        norm_params.append(weight)

    params: list[torch.nn.Parameter] = []
    if quant_mlp is not None:
        params.extend(list(quant_mlp.parameters()))
    for quant in quant_linears.values():
        params.extend(list(quant.parameters()))
    params.extend(norm_params)
    if not params:
        raise RuntimeError(f"no trainable parameters selected for layer {layer_id}")

    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    keep_ref = {"keep": 1}
    handles, captured = register_hidden_hooks(model, hidden_layers, keep_ref)
    hidden_projection_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    history = []
    try:
        for step in range(1, args.steps_per_layer + 1):
            row = rows[(step - 1) % len(rows)]
            keep_ref["keep"] = int(row["keep"])
            captured.clear()
            outputs = model(**row_to_inputs(row, device), use_cache=False, logits_to_keep=int(row["keep"]))
            kl, stats = topk_teacher_kl(
                outputs.logits,
                row["top_values"],
                row["top_indices"],
                temperature=args.temperature,
                format_ids=format_ids,
                format_boost=args.format_boost,
                teacher_logsumexp=row.get("teacher_logsumexp"),
                use_other_bucket=row.get("teacher_logsumexp") is not None,
            )
            anti = anti_collapse_loss(outputs.logits, row["top_values"], row["top_indices"], collapse_ids, margin=args.anti_margin)
            h_loss = (
                projected_or_full_hidden_loss(
                    captured,
                    row,
                    hidden_layers,
                    device,
                    projected_dim=hidden_projection_dim,
                    projection_seed=hidden_projection_seed,
                    projection_cache=hidden_projection_cache,
                )
                if args.hidden_weight > 0
                else kl.detach().new_zeros(())
            )
            code_terms = []
            if args.code_weight > 0:
                if quant_mlp is not None:
                    code_terms.append(quant_mlp.codebook_loss())
                code_terms.extend(quant.codebook_loss() for quant in quant_linears.values())
            code = torch.stack(code_terms).mean() if code_terms else kl.detach().new_zeros(())
            loss = kl + args.hidden_weight * h_loss + args.anti_weight * anti + args.code_weight * code
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps_per_layer:
                record = {
                    "layer": layer_id,
                    "step": step,
                    "task": row.get("task"),
                    "id": row.get("id"),
                    "tokens": int(row["tokens"]),
                    "loss": float(loss.detach().cpu()),
                    "hidden": float(h_loss.detach().cpu()),
                    "anti": float(anti.detach().cpu()),
                    "code": float(code.detach().cpu()),
                    **stats,
                }
                history.append(record)
                print(json.dumps(record), flush=True)
            del outputs, loss, kl, anti, h_loss, code
            captured.clear()
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    output: dict[str, torch.Tensor] = {}
    if quant_mlp is not None:
        hard = freeze_quant_mlp(model, layer_id, quant_mlp)
        output[mlp_weight_name(layer_id, "gate")] = hard["gate"]
        output[mlp_weight_name(layer_id, "up")] = hard["up"]
        output[mlp_weight_name(layer_id, "down")] = hard["down"]
    for name, quant in quant_linears.items():
        output[f"{name}.weight"] = freeze_quant_linear(model, name, quant)
    for name in norm_names:
        module = model.get_submodule(name)
        module.weight.requires_grad_(False)
        module.weight.data = module.weight.data.to(torch.bfloat16)
        output[f"{name}.weight"] = module.weight.detach().cpu().to(torch.bfloat16)
    return output, {"layer": layer_id, "linears": sorted(quant_linears), "norms": norm_names, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--base-patch", action="append", type=Path)
    parser.add_argument("--include-base-in-output", action="store_true")
    parser.add_argument("--out-patch", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--layers", default="0,1,10,12,27,28,30,31")
    parser.add_argument("--max-train-tokens", type=int, default=3500)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-rows-per-task", type=int)
    parser.add_argument("--shuffle-rows", action="store_true")
    parser.add_argument("--interleave-tasks", action="store_true")
    parser.add_argument("--row-seed", type=int, default=20260816)
    parser.add_argument("--failures")
    parser.add_argument("--mode", choices=("ternary", "binary"), default="ternary")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--steps-per-layer", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--format-boost", type=float, default=5.0)
    parser.add_argument("--hidden-weight", type=float, default=0.0)
    parser.add_argument("--anti-weight", type=float, default=0.1)
    parser.add_argument("--anti-margin", type=float, default=0.05)
    parser.add_argument("--code-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--train-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-projections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection-regex")
    parser.add_argument("--train-norms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact-exact-ternary", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()

    if args.torch_threads < 0:
        raise ValueError("torch-threads must be non-negative")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    cache = torch.load(args.cache, map_location="cpu")
    if cache.get("format") not in {"qwen_onpolicy_survival_cache_v1", "qwen_agentworld_teacher_behavior_v1"}:
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    rows = selected_rows(cache, args)
    hidden_layers = [int(layer) for layer in cache.get("hidden_layers", [])] if args.hidden_weight > 0 else []
    hidden_projection_dim = int(cache.get("hidden_projection_dim") or 0)
    hidden_projection_seed = int(cache.get("hidden_projection_seed") or 0)
    if args.hidden_weight > 0 and any("teacher_hidden_proj" in row for row in rows) and hidden_projection_dim <= 0:
        raise ValueError("projected hidden labels require hidden_projection_dim metadata")

    model = load_model(args.model_dir, device=args.device)
    release_non_text_modules(model)
    model.eval()
    available_layers = {
        int(match.group(1))
        for name, _module in model.named_modules()
        if (match := re.match(r"^model\.(?:language_model\.)?layers\.(\d+)$", name))
    }
    selected_layers = parse_layers(args.layers, available_layers)
    unknown_layers = sorted(set(selected_layers) - available_layers)
    if unknown_layers:
        raise ValueError(f"unknown layer ids: {unknown_layers}; available={sorted(available_layers)}")
    for param in model.parameters():
        param.requires_grad_(False)
    patch_reports = [apply_weight_patch(model, patch) for patch in args.base_patch or []]
    device = next(model.parameters()).device
    format_ids = torch.tensor(cache.get("format_token_ids", []), device=device, dtype=torch.long)
    collapse_ids = torch.tensor(cache.get("collapse_token_ids", []), device=device, dtype=torch.long)

    output: dict[str, torch.Tensor] = {}
    if args.include_base_in_output:
        for patch in args.base_patch or []:
            output.update(load_file(str(patch), device="cpu"))
    reports = []
    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    for layer_id in selected_layers:
        hard, report = train_block(
            model,
            rows,
            layer_id,
            args,
            device,
            hidden_layers,
            format_ids,
            collapse_ids,
            hidden_projection_dim,
            hidden_projection_seed,
        )
        output.update(hard)
        reports.append(report)
        atomic_save_patch(args.out_patch, output)
        atomic_write_json(
            args.out_report,
            {"partial": True, "completed_layers": [r["layer"] for r in reports], "reports": reports},
        )
        torch.cuda.empty_cache()

    result = {
        "cache": str(args.cache),
        "base_patches": [str(path) for path in args.base_patch or []],
        "included_base_in_output": args.include_base_in_output,
        "out_patch": str(args.out_patch),
        "patch_reports": patch_reports,
        "layers": selected_layers,
        "hidden_layers": hidden_layers,
        "mode": args.mode,
        "group_size": args.group_size,
        "artifact_exact_ternary": args.artifact_exact_ternary,
        "steps_per_layer": args.steps_per_layer,
        "lr": args.lr,
        "projection_regex": args.projection_regex,
        "runtime_device": str(device),
        "torch_threads": torch.get_num_threads(),
        "train_rows": [
            {"index": row.get("index"), "task": row.get("task"), "id": row.get("id"), "tokens": int(row["tokens"]), "keep": int(row["keep"])}
            for row in rows
        ],
        "shuffle_rows": args.shuffle_rows,
        "interleave_tasks": args.interleave_tasks,
        "row_seed": args.row_seed,
        "max_rows_per_task": args.max_rows_per_task,
        "reports": reports,
    }
    result["patch_size_bytes"] = args.out_patch.stat().st_size
    result["patch_sha256"] = sha256_file(args.out_patch)
    atomic_write_json(args.out_report, result)
    print(json.dumps({"out_patch": str(args.out_patch), "out_report": str(args.out_report), "layers": selected_layers}, indent=2))


if __name__ == "__main__":
    main()
