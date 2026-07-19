"""Train Qwen blocks against adjoint-shifted teacher targets.

This is the local optimizer half of Adjoint Block QAT. It consumes a cache from
compute_block_adjoint_cache.py and trains each active block toward:

    B_l^T(h_l^Q) - alpha_lambda * normalize(lambda_{l+1})

The whole model is still hard ternary via the loaded patch, but only the active
block's master weights are trainable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_quantize.bonsai_qat8 import BonsaiQAT8
from collect_onpolicy_survival_cache import load_model, parse_layers
from train_block_gauss_seidel_qat import layer_module, replace_layer_linears, select_layer_norms
from weight_patch import apply_weight_patch


def normalized_adjoint_shift(target: torch.Tensor, adjoint: torch.Tensor, *, alpha_lambda: float) -> tuple[torch.Tensor, dict[str, float]]:
    target = target.detach().float()
    adjoint = adjoint.detach().float()
    target_rms = target.square().mean().sqrt().clamp_min(1e-8)
    adjoint_rms = adjoint.square().mean().sqrt().clamp_min(1e-8)
    normalized = adjoint * (target_rms / adjoint_rms)
    shifted = target - alpha_lambda * normalized
    return shifted, {
        "target_rms": float(target_rms.detach().cpu()),
        "adjoint_rms": float(adjoint_rms.detach().cpu()),
        "shift_rms": float((alpha_lambda * normalized).square().mean().sqrt().detach().cpu()),
    }


def train_one_layer(
    model: torch.nn.Module,
    cache: dict[str, Any],
    layer_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    quant_linears = replace_layer_linears(model, layer_id, args.group_size)
    norm_params = select_layer_norms(model, layer_id) if args.train_norms else {}
    params: list[torch.nn.Parameter] = [quant.master for quant in quant_linears.values()]
    params.extend(norm_params.values())
    if not params:
        raise RuntimeError(f"no trainable parameters for layer {layer_id}")

    matrix_params = [param for param in params if param.ndim == 2]
    other_params = [param for param in params if param.ndim != 2]
    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append({"params": matrix_params, "role": "matrix_qat8", "lr": args.lr, "betas": (0.9, 0.95)})
    if other_params:
        groups.append({"params": other_params, "role": "adamw", "lr": args.lr, "betas": (0.9, 0.95)})
    optimizer = BonsaiQAT8(groups)

    rows = cache["rows"]
    blocks = cache["blocks"]
    layer = layer_module(model, layer_id)
    history = []
    for step in range(1, args.steps_per_layer + 1):
        row_index = (step - 1) % len(rows)
        row = rows[row_index]
        item = blocks[row_index][layer_id]
        keep = int(row["keep"])
        h_in = item["h_in"].unsqueeze(0).to(device)
        cos = item["cos"].unsqueeze(0).to(device)
        sin = item["sin"].unsqueeze(0).to(device)
        position_ids = item.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.unsqueeze(0).to(device)
        target = item["target_last"].unsqueeze(0).to(device)
        adjoint = item["adjoint_last"].unsqueeze(0).to(device)
        shifted_target, shift_stats = normalized_adjoint_shift(target, adjoint, alpha_lambda=args.alpha_lambda)

        out = layer(
            h_in,
            position_embeddings=(cos, sin),
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        pred = out[0] if isinstance(out, tuple) else out
        pred_last = pred[:, -keep:, :].float()
        shifted_loss = (pred_last - shifted_target).square().mean() / target.float().square().mean().clamp_min(1e-8)
        teacher_loss = (pred_last - target.float()).square().mean() / target.float().square().mean().clamp_min(1e-8)
        code_terms = [quant.codebook_loss() for quant in quant_linears.values()]
        code_loss = torch.stack(code_terms).mean() if code_terms else shifted_loss.detach().new_zeros(())
        loss = args.shifted_weight * shifted_loss + args.teacher_weight * teacher_loss + args.code_weight * code_loss
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
                "shifted": float(shifted_loss.detach().cpu()),
                "teacher": float(teacher_loss.detach().cpu()),
                "code": float(code_loss.detach().cpu()),
                **shift_stats,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        del out, pred, pred_last, shifted_target, loss, shifted_loss, teacher_loss, code_loss
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
    parser.add_argument("--steps-per-layer", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--alpha-lambda", type=float, default=0.03)
    parser.add_argument("--shifted-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--code-weight", type=float, default=0.002)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--train-norms", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("format") != "qwen_onpolicy_block_adjoint_cache_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    layers = parse_layers(args.layers)
    cache["blocks"] = {int(row_index): {int(layer): item for layer, item in layer_items.items()} for row_index, layer_items in cache["blocks"].items()}
    for layer_id in layers:
        for row_index in range(len(cache["rows"])):
            item = cache["blocks"].get(row_index, {}).get(layer_id)
            if item is None or "adjoint_last" not in item:
                raise KeyError(f"cache missing adjoint for row {row_index}, layer {layer_id}")

    model = load_model(args.model_dir)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    patch_reports = [apply_weight_patch(model, patch) for patch in args.base_patch or []]
    device = next(model.parameters()).device

    output: dict[str, torch.Tensor] = {}
    reports = []
    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    for layer_id in layers:
        patch, report = train_one_layer(model, cache, layer_id, args, device)
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
        "alpha_lambda": args.alpha_lambda,
        "shifted_weight": args.shifted_weight,
        "teacher_weight": args.teacher_weight,
        "code_weight": args.code_weight,
        "reports": reports,
    }
    save_file(output, str(args.out_patch))
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out_patch": str(args.out_patch), "layers": layers}, indent=2))


if __name__ == "__main__":
    main()
