"""Adjoint blockwise QAT for one-GPU all-2D ternary Qwen.

This trains one active block while the downstream suffix stays frozen. The
active block receives a real behavior gradient through the frozen suffix:

    h_l^Q -> B_l^Q -> S_{l+1:L}^Q -> logits -> teacher KL / format loss

It is the one-A100 approximation to simultaneous all-2D QAT.
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
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_quantize.bonsai_qat8 import BonsaiQAT8
from collect_onpolicy_survival_cache import load_model, parse_layers
from train_all2d_ternary_qat import alpha_quantized_weight, dequantize_ternary_chunked, get_parent_and_attr
from train_mlp_triplet_survival_qat import anti_collapse_loss, topk_teacher_kl
from weight_patch import apply_weight_patch


class TrainQuantLinear(torch.nn.Module):
    def __init__(self, source: torch.nn.Linear, *, group_size: int) -> None:
        super().__init__()
        self.master = torch.nn.Parameter(source.weight.detach().float().clone())
        self.group_size = group_size
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = alpha_quantized_weight(self.master, mode="ternary", group_size=self.group_size, alpha=1.0)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight.to(dtype=x.dtype), bias)

    def codebook_loss(self) -> torch.Tensor:
        with torch.no_grad():
            quantized = dequantize_ternary_chunked(self.master, group_size=self.group_size)
        denom = self.master.detach().float().square().mean().clamp_min(1e-8)
        return (self.master.float() - quantized.float()).square().mean() / denom

    @torch.no_grad()
    def hard_weight(self) -> torch.Tensor:
        return dequantize_ternary_chunked(self.master, group_size=self.group_size).cpu().to(torch.bfloat16)


def layer_name(layer_id: int) -> str:
    return f"model.language_model.layers.{layer_id}"


def layer_module(model: torch.nn.Module, layer_id: int) -> torch.nn.Module:
    return model.get_submodule(layer_name(layer_id))


def replace_layer_linears(model: torch.nn.Module, layer_id: int, group_size: int) -> dict[str, TrainQuantLinear]:
    prefix = layer_name(layer_id)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith(prefix + ".") and isinstance(module, torch.nn.Linear)
    ]
    out: dict[str, TrainQuantLinear] = {}
    for name, module in targets:
        quant = TrainQuantLinear(module, group_size=group_size).to(next(module.parameters()).device)
        parent, attr = get_parent_and_attr(model, name)
        setattr(parent, attr, quant)
        out[name] = quant
    return out


def select_layer_norms(model: torch.nn.Module, layer_id: int) -> dict[str, torch.nn.Parameter]:
    prefix = layer_name(layer_id)
    selected: dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if name.startswith(prefix + ".") and name.endswith(".weight") and param.ndim == 1:
            param.data = param.data.float()
            param.requires_grad_(True)
            selected[name] = param
    return selected


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    return model.get_submodule("model.language_model.layers")


def final_logits(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    keep: int,
) -> torch.Tensor:
    final_norm = model.get_submodule("model.language_model.norm")
    lm_head = model.get_submodule("lm_head")
    norm_dtype = next(final_norm.parameters()).dtype
    normed = final_norm(hidden_states[:, -keep:, :].to(dtype=norm_dtype))
    return lm_head(normed)


def run_suffix(
    model: torch.nn.Module,
    *,
    start_layer: int,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor | None,
    use_checkpoint: bool,
) -> torch.Tensor:
    h = hidden_states
    layers = decoder_layers(model)
    for layer_id in range(start_layer, len(layers)):
        layer = layers[layer_id]

        def layer_forward(x: torch.Tensor, *, layer=layer) -> torch.Tensor:
            out = layer(
                x,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            return out[0] if isinstance(out, tuple) else out

        if use_checkpoint:
            h = checkpoint(layer_forward, h, use_reentrant=False)
        else:
            h = layer_forward(h)
    return h


def train_one_layer(
    model: torch.nn.Module,
    cache: dict[str, Any],
    layer_id: int,
    args: argparse.Namespace,
    device: torch.device,
    format_ids: torch.Tensor,
    collapse_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    quant_linears = replace_layer_linears(model, layer_id, args.group_size)
    norm_params = select_layer_norms(model, layer_id) if args.train_norms else {}
    params: list[torch.nn.Parameter] = [quant.master for quant in quant_linears.values()]
    params.extend(norm_params.values())
    if not params:
        raise RuntimeError(f"no trainable parameters for layer {layer_id}")

    matrix_params = [p for p in params if p.ndim == 2]
    other_params = [p for p in params if p.ndim != 2]
    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append({"params": matrix_params, "role": "matrix_qat8", "lr": args.lr, "betas": (0.9, 0.95)})
    if other_params:
        groups.append({"params": other_params, "role": "adamw", "lr": args.lr, "betas": (0.9, 0.95)})
    optimizer = BonsaiQAT8(groups)

    rows = cache["rows"]
    blocks = cache["blocks"]
    active = layer_module(model, layer_id)
    history = []
    for step in range(1, args.steps_per_layer + 1):
        row_index = (step - 1) % len(rows)
        row = rows[row_index]
        item = blocks[row_index][layer_id]
        keep = int(row["keep"])
        h_in = item["h_in"].unsqueeze(0).to(device)
        cos = item["cos"].unsqueeze(0).to(device)
        sin = item["sin"].unsqueeze(0).to(device)
        position_embeddings = (cos, sin)
        position_ids = item.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.unsqueeze(0).to(device)
        target_block = item["target_last"].unsqueeze(0).to(device).float()

        y = active(
            h_in,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        y = y[0] if isinstance(y, tuple) else y
        y_last = y[:, -keep:, :].float()
        block_loss = (y_last - target_block).square().mean() / target_block.square().mean().clamp_min(1e-8)
        suffix_out = run_suffix(
            model,
            start_layer=layer_id + 1,
            hidden_states=y,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            use_checkpoint=args.suffix_checkpoint,
        )
        logits = final_logits(model, suffix_out, keep=keep)
        logit_loss, stats = topk_teacher_kl(
            logits,
            row["top_values"],
            row["top_indices"],
            temperature=args.temperature,
            format_ids=format_ids,
            format_boost=args.format_boost,
        )
        anti = anti_collapse_loss(logits, row["top_values"], row["top_indices"], collapse_ids, margin=args.anti_margin)
        code_terms = [quant.codebook_loss() for quant in quant_linears.values()]
        code = torch.stack(code_terms).mean() if code_terms else block_loss.detach().new_zeros(())
        loss = args.logit_weight * logit_loss + args.block_weight * block_loss + args.anti_weight * anti + args.code_weight * code
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
                "logit": float(logit_loss.detach().cpu()),
                "block": float(block_loss.detach().cpu()),
                "anti": float(anti.detach().cpu()),
                "code": float(code.detach().cpu()),
                **stats,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        del y, y_last, suffix_out, logits, loss, logit_loss, block_loss, anti, code
        torch.cuda.empty_cache()

    patch: dict[str, torch.Tensor] = {}
    for name, quant in quant_linears.items():
        patch[f"{name}.weight"] = quant.hard_weight().contiguous()
    for name, param in norm_params.items():
        patch[name] = param.detach().cpu().to(torch.bfloat16).contiguous()
    return patch, {"layer": layer_id, "linears": sorted(quant_linears), "norms": sorted(norm_params), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--base-patch", action="append", type=Path)
    parser.add_argument("--out-patch", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--steps-per-layer", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--format-boost", type=float, default=5.0)
    parser.add_argument("--logit-weight", type=float, default=1.0)
    parser.add_argument("--block-weight", type=float, default=0.25)
    parser.add_argument("--anti-weight", type=float, default=0.1)
    parser.add_argument("--anti-margin", type=float, default=0.05)
    parser.add_argument("--code-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--train-norms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suffix-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("format") != "qwen_onpolicy_block_cache_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    layers = parse_layers(args.layers)
    cache["blocks"] = {int(row_index): {int(layer): item for layer, item in layer_items.items()} for row_index, layer_items in cache["blocks"].items()}

    model = load_model(args.model_dir)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    patch_reports = [apply_weight_patch(model, patch) for patch in args.base_patch or []]
    device = next(model.parameters()).device
    format_ids = torch.tensor(cache.get("format_token_ids", []), device=device, dtype=torch.long)
    collapse_ids = torch.tensor(cache.get("collapse_token_ids", []), device=device, dtype=torch.long)

    output: dict[str, torch.Tensor] = {}
    reports = []
    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    for layer_id in layers:
        patch, report = train_one_layer(model, cache, layer_id, args, device, format_ids, collapse_ids)
        output.update(patch)
        reports.append(report)
        save_file(output, str(args.out_patch))
        args.out_report.write_text(json.dumps({"partial": True, "completed_layers": [r["layer"] for r in reports], "reports": reports}, indent=2) + "\n")
        torch.cuda.empty_cache()

    result = {
        "cache": str(args.cache),
        "base_patches": [str(path) for path in args.base_patch or []],
        "patch_reports": patch_reports,
        "out_patch": str(args.out_patch),
        "layers": layers,
        "group_size": args.group_size,
        "steps_per_layer": args.steps_per_layer,
        "lr": args.lr,
        "logit_weight": args.logit_weight,
        "block_weight": args.block_weight,
        "anti_weight": args.anti_weight,
        "code_weight": args.code_weight,
        "reports": reports,
    }
    save_file(output, str(args.out_patch))
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out_patch": str(args.out_patch), "layers": layers}, indent=2))


if __name__ == "__main__":
    main()
