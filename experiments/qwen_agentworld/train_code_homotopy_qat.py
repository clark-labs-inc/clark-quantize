"""Magnitude-aware code-homotopy QAT for Qwen AgentWorld survival.

This trains code transition logits and group scales directly, without dense
master weights or LoRA adapters. It is intended as a small one-GPU test of the
"move the codebook itself" hypothesis from the Bonsai/Prism forensics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant.optimal_ternary import quantize_ternary_mse
from train_mlp_triplet_survival_qat import (
    anti_collapse_loss,
    get_parent_and_attr,
    hidden_loss,
    load_model,
    parse_layers,
    register_hidden_hooks,
    row_to_inputs,
    topk_teacher_kl,
)
from weight_patch import apply_weight_patch


def layer_prefix(layer_id: int) -> str:
    return f"model.language_model.layers.{layer_id}"


def mlp_linear_names(layer_id: int) -> list[str]:
    prefix = layer_prefix(layer_id)
    return [
        f"{prefix}.mlp.gate_proj",
        f"{prefix}.mlp.up_proj",
        f"{prefix}.mlp.down_proj",
    ]


def canonical_weight_name(module_name: str) -> str:
    return f"{module_name}.weight"


class CodeHomotopyLinear(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Linear,
        *,
        group_size: int,
        candidate_fraction: float,
        max_candidates: int,
        candidate_strategy: str,
        init_margin: float,
        prior_beta: float,
    ) -> None:
        super().__init__()
        if source.bias is not None:
            raise ValueError("CodeHomotopyLinear currently expects bias=False")
        weight = source.weight.detach().float()
        quant = quantize_ternary_mse(weight, group_size=group_size)
        flat_codes = quant.codes.reshape(-1).to(torch.int8)
        real_numel = weight.numel()
        flat_weight = weight.reshape(-1)

        k = max(1, int(real_numel * candidate_fraction))
        k = min(k, max_candidates, real_numel)
        if candidate_strategy == "bottom_abs":
            score = flat_weight.abs()
        elif candidate_strategy == "boundary":
            score = ternary_boundary_score(flat_weight, flat_codes, quant.scales, group_size, quant.pad)
        else:
            raise ValueError(f"unsupported candidate strategy: {candidate_strategy}")
        candidate_idx = torch.topk(score, k=k, largest=False).indices.sort().values
        old_code = flat_codes.index_select(0, candidate_idx).to(torch.int8)
        base_sign = torch.where(
            flat_weight.index_select(0, candidate_idx) >= 0,
            torch.ones_like(old_code),
            -torch.ones_like(old_code),
        ).to(torch.int8)
        alt1 = torch.where(old_code == 0, base_sign, torch.zeros_like(old_code))
        alt2 = torch.where(old_code == 0, -base_sign, -old_code)
        options = torch.stack([old_code, alt1, alt2], dim=1).to(torch.int8)

        logits = torch.zeros((k, 3), dtype=torch.float32)
        logits[:, 0] = init_margin

        base_abs = flat_weight.index_select(0, candidate_idx).abs().clamp_min(1e-8)
        prior = base_abs.pow(prior_beta)
        prior = prior / prior.mean().clamp_min(1e-8)

        self.original_shape = tuple(weight.shape)
        self.group_size = int(group_size)
        self.pad = int(quant.pad)
        self.register_buffer("base_codes", flat_codes)
        self.register_buffer("candidate_idx", candidate_idx.to(torch.long))
        self.register_buffer("old_code", old_code)
        self.register_buffer("options", options)
        self.register_buffer("prior_weight", prior)
        self.register_buffer("init_log_scales", quant.scales.float().log())
        self.log_scales = torch.nn.Parameter(quant.scales.float().log())
        self.logits = torch.nn.Parameter(logits)
        self.hard_forward = True

    def soft_codes(self) -> torch.Tensor:
        probs = F.softmax(self.logits.float(), dim=-1)
        if not self.hard_forward:
            return (probs * self.options.to(probs.device, dtype=probs.dtype)).sum(dim=-1)
        hard_idx = probs.argmax(dim=-1)
        hard = F.one_hot(hard_idx, num_classes=probs.shape[-1]).to(probs.dtype)
        onehot = hard + (probs - probs.detach())
        return (onehot * self.options.to(probs.device, dtype=probs.dtype)).sum(dim=-1)

    def dequantized_weight(self) -> torch.Tensor:
        base = self.base_codes.to(self.logits.device, dtype=torch.float32)
        new_code = self.soft_codes()
        old = self.old_code.to(self.logits.device, dtype=torch.float32)
        delta = torch.zeros_like(base)
        delta.index_add_(0, self.candidate_idx.to(self.logits.device), new_code - old)
        codes = (base + delta).reshape(-1, self.group_size)
        values = codes * self.log_scales.float().exp().reshape(-1, 1)
        flat = values.reshape(-1)
        if self.pad:
            flat = flat[:-self.pad]
        return flat.reshape(self.original_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight().to(dtype=x.dtype)
        return F.linear(x, weight)

    def transition_prior(self) -> torch.Tensor:
        probs = F.softmax(self.logits.float(), dim=-1)
        expected_change = 1.0 - probs[:, 0]
        return (expected_change * self.prior_weight.to(probs.device)).mean()

    def scale_prior(self) -> torch.Tensor:
        return (self.log_scales.float() - self.init_log_scales.to(self.log_scales.device)).square().mean()

    @torch.no_grad()
    def stats(self) -> dict[str, float]:
        probs = F.softmax(self.logits.float(), dim=-1)
        hard = probs.argmax(dim=-1)
        changed = (hard != 0).float().mean()
        opt = [(hard == idx).float().mean().item() for idx in range(3)]
        scale_ratio = (self.log_scales.float() - self.init_log_scales.to(self.log_scales.device)).exp()
        return {
            "candidates": float(self.candidate_idx.numel()),
            "changed": float(changed.cpu()),
            "keep": float(opt[0]),
            "alt1": float(opt[1]),
            "alt2": float(opt[2]),
            "scale_ratio_mean": float(scale_ratio.mean().cpu()),
            "scale_ratio_max": float(scale_ratio.max().cpu()),
        }

    @torch.no_grad()
    def hard_weight(self) -> torch.Tensor:
        return self.dequantized_weight().detach().cpu().to(torch.bfloat16)

    @torch.no_grad()
    def sparse_code_artifact(self) -> dict[str, torch.Tensor]:
        probs = F.softmax(self.logits.float(), dim=-1)
        option_index = probs.argmax(dim=-1).to(torch.int64)
        chosen_code = self.options.to(option_index.device).gather(1, option_index[:, None]).squeeze(1).to(torch.int8)
        changed = option_index != 0
        return {
            "original_shape": torch.tensor(self.original_shape, dtype=torch.int64),
            "group_size": torch.tensor([self.group_size], dtype=torch.int64),
            "pad": torch.tensor([self.pad], dtype=torch.int64),
            "changed_candidate_idx": self.candidate_idx.detach().cpu()[changed.detach().cpu()],
            "old_code": self.old_code.detach().cpu()[changed.detach().cpu()],
            "new_code": chosen_code.detach().cpu()[changed.detach().cpu()],
            "option_index": option_index.detach().cpu()[changed.detach().cpu()].to(torch.uint8),
            "final_scales_fp16": self.log_scales.detach().cpu().float().exp().to(torch.float16),
            "initial_scales_fp16": self.init_log_scales.detach().cpu().float().exp().to(torch.float16),
        }


def replace_linear(model: torch.nn.Module, name: str, args: argparse.Namespace) -> CodeHomotopyLinear:
    source = model.get_submodule(name)
    hom = CodeHomotopyLinear(
        source,
        group_size=args.group_size,
        candidate_fraction=args.candidate_fraction,
        max_candidates=args.max_candidates_per_tensor,
        candidate_strategy=args.candidate_strategy,
        init_margin=args.init_margin,
        prior_beta=args.prior_beta,
    ).to(next(source.parameters()).device)
    parent, attr = get_parent_and_attr(model, name)
    setattr(parent, attr, hom)
    return hom


@torch.no_grad()
def ternary_boundary_score(
    flat_weight: torch.Tensor,
    flat_codes: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    pad: int,
) -> torch.Tensor:
    """Small score means near the ternary support boundary for its group."""
    if pad:
        padded_weight = torch.cat([flat_weight, flat_weight.new_zeros(pad)])
    else:
        padded_weight = flat_weight
    abs_groups = padded_weight.abs().reshape(-1, group_size)
    code_groups = flat_codes.reshape(-1, group_size)
    nonzero = code_groups != 0
    zero = ~nonzero
    inf = torch.full_like(abs_groups, float("inf"))
    neg_inf = torch.full_like(abs_groups, -float("inf"))
    min_support = torch.where(nonzero, abs_groups, inf).min(dim=1).values
    max_zero = torch.where(zero, abs_groups, neg_inf).max(dim=1).values
    min_support = torch.where(torch.isfinite(min_support), min_support, scales.to(min_support.device))
    max_zero = torch.where(torch.isfinite(max_zero), max_zero, torch.zeros_like(max_zero))
    support_margin = (abs_groups - max_zero.reshape(-1, 1)).abs()
    zero_margin = (min_support.reshape(-1, 1) - abs_groups).abs()
    score = torch.where(nonzero, support_margin, zero_margin).reshape(-1)
    if pad:
        score = score[:-pad]
    return score


def selected_rows(cache: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in cache["rows"] if int(row["tokens"]) <= args.max_train_tokens]
    if args.max_train_rows is not None:
        rows = rows[: args.max_train_rows]
    if not rows:
        raise RuntimeError("no cache rows selected")
    return rows


def aggregate_homotopy_stats(modules: dict[str, CodeHomotopyLinear]) -> dict[str, float]:
    rows = [module.stats() for module in modules.values()]
    total_candidates = sum(row["candidates"] for row in rows)
    if total_candidates <= 0:
        return {}
    return {
        "hom_candidates": total_candidates,
        "hom_changed": sum(row["changed"] * row["candidates"] for row in rows) / total_candidates,
        "hom_keep": sum(row["keep"] * row["candidates"] for row in rows) / total_candidates,
        "hom_alt1": sum(row["alt1"] * row["candidates"] for row in rows) / total_candidates,
        "hom_alt2": sum(row["alt2"] * row["candidates"] for row in rows) / total_candidates,
        "hom_scale_ratio_mean": sum(row["scale_ratio_mean"] for row in rows) / len(rows),
        "hom_scale_ratio_max": max(row["scale_ratio_max"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--base-patch", action="append", type=Path)
    parser.add_argument("--out-patch", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--out-code-artifact", type=Path)
    parser.add_argument("--layers", default="28,29,30")
    parser.add_argument("--max-train-tokens", type=int, default=1300)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--candidate-fraction", type=float, default=0.005)
    parser.add_argument("--candidate-strategy", choices=("bottom_abs", "boundary"), default="bottom_abs")
    parser.add_argument("--max-candidates-per-tensor", type=int, default=250_000)
    parser.add_argument("--init-margin", type=float, default=2.0)
    parser.add_argument("--prior-beta", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--logit-lr", type=float, default=1e-3)
    parser.add_argument("--scale-lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--format-boost", type=float, default=5.0)
    parser.add_argument("--hidden-weight", type=float, default=0.0)
    parser.add_argument("--anti-weight", type=float, default=0.1)
    parser.add_argument("--anti-margin", type=float, default=0.05)
    parser.add_argument("--transition-weight", type=float, default=1e-3)
    parser.add_argument("--scale-weight", type=float, default=0.0)
    parser.add_argument("--soft-forward-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cache = torch.load(args.cache, map_location="cpu")
    if cache.get("format") != "qwen_onpolicy_survival_cache_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    rows = selected_rows(cache, args)
    selected_layers = parse_layers(args.layers, set(range(32)))
    hidden_layers = [layer for layer in cache.get("hidden_layers", []) if layer in selected_layers and args.hidden_weight > 0]

    model = load_model(args.model_dir)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    patch_reports = [apply_weight_patch(model, patch) for patch in args.base_patch or []]
    modules: dict[str, CodeHomotopyLinear] = {}
    for layer_id in selected_layers:
        for name in mlp_linear_names(layer_id):
            modules[name] = replace_linear(model, name, args)
    if not modules:
        raise RuntimeError("no homotopy modules attached")

    logit_params = [module.logits for module in modules.values()]
    scale_params = [module.log_scales for module in modules.values()]
    optimizer = torch.optim.AdamW(
        [
            {"params": logit_params, "lr": args.logit_lr, "weight_decay": 0.0},
            {"params": scale_params, "lr": args.scale_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    device = next(model.parameters()).device
    format_ids = torch.tensor(cache.get("format_token_ids", []), device=device, dtype=torch.long)
    collapse_ids = torch.tensor(cache.get("collapse_token_ids", []), device=device, dtype=torch.long)
    keep_ref = {"keep": 1}
    handles, captured = register_hidden_hooks(model, hidden_layers, keep_ref)
    history = []
    try:
        for step in range(1, args.steps + 1):
            hard_forward = step > args.soft_forward_steps
            for module in modules.values():
                module.hard_forward = hard_forward
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
            )
            anti = anti_collapse_loss(outputs.logits, row["top_values"], row["top_indices"], collapse_ids, margin=args.anti_margin)
            h_loss = hidden_loss(captured, row, hidden_layers, device) if args.hidden_weight > 0 else kl.detach().new_zeros(())
            transition = torch.stack([module.transition_prior() for module in modules.values()]).mean()
            scale = torch.stack([module.scale_prior() for module in modules.values()]).mean()
            loss = kl + args.hidden_weight * h_loss + args.anti_weight * anti + args.transition_weight * transition + args.scale_weight * scale
            if not torch.isfinite(loss):
                print(json.dumps({"error": "nonfinite_loss", "step": step, "loss": float(loss.detach().cpu())}), flush=True)
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(logit_params + scale_params, args.grad_clip)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                record = {
                    "step": step,
                    "task": row.get("task"),
                    "id": row.get("id"),
                    "tokens": int(row["tokens"]),
                    "loss": float(loss.detach().cpu()),
                    "hidden": float(h_loss.detach().cpu()),
                    "anti": float(anti.detach().cpu()),
                    "transition": float(transition.detach().cpu()),
                    "scale": float(scale.detach().cpu()),
                    **stats,
                    **aggregate_homotopy_stats(modules),
                }
                history.append(record)
                print(json.dumps(record), flush=True)
            del outputs, loss, kl, anti, h_loss, transition, scale
            captured.clear()
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    for module in modules.values():
        module.hard_forward = True
    tensors = {canonical_weight_name(name): module.hard_weight() for name, module in modules.items()}
    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out_patch))
    if args.out_code_artifact is not None:
        args.out_code_artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "clark_code_homotopy_sparse_v1",
                "cache": str(args.cache),
                "base_patches": [str(path) for path in args.base_patch or []],
                "dense_gate_patch": str(args.out_patch),
                "layers": selected_layers,
                "modules": {
                    name: module.sparse_code_artifact()
                    for name, module in modules.items()
                },
            },
            args.out_code_artifact,
        )
    result = {
        "cache": str(args.cache),
        "base_patches": [str(path) for path in args.base_patch or []],
        "patch_reports": patch_reports,
        "out_patch": str(args.out_patch),
        "out_code_artifact": str(args.out_code_artifact) if args.out_code_artifact is not None else None,
        "layers": selected_layers,
        "modules": sorted(modules),
        "group_size": args.group_size,
        "candidate_fraction": args.candidate_fraction,
        "candidate_strategy": args.candidate_strategy,
        "max_candidates_per_tensor": args.max_candidates_per_tensor,
        "transition_weight": args.transition_weight,
        "scale_weight": args.scale_weight,
        "soft_forward_steps": args.soft_forward_steps,
        "history": history,
        "final_homotopy": {name: module.stats() for name, module in modules.items()},
    }
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out_patch": str(args.out_patch),
        "out_report": str(args.out_report),
        "out_code_artifact": str(args.out_code_artifact) if args.out_code_artifact is not None else None,
        "modules": len(modules),
    }, indent=2))


if __name__ == "__main__":
    main()
