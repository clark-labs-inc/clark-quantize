"""Compute downstream adjoint messages for blockwise QAT.

This augments a qwen_onpolicy_block_cache_v1 file with lambda_{l+1} messages:

    lambda_{l+1} = d Phi(final_logits) / d h_{l+1}

for each cached student state h_l^Q and selected block l. The resulting cache
lets the block trainer use a shifted teacher target:

    B_l^T(h_l^Q) - alpha * normalized(lambda_{l+1})

without recomputing the downstream suffix on every optimizer step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_onpolicy_survival_cache import load_model, parse_layers
from train_mlp_triplet_survival_qat import anti_collapse_loss, topk_teacher_kl
from weight_patch import apply_weight_patch


def layer_name(layer_id: int) -> str:
    return f"model.language_model.layers.{layer_id}"


def layer_module(model: torch.nn.Module, layer_id: int) -> torch.nn.Module:
    return model.get_submodule(layer_name(layer_id))


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    return model.get_submodule("model.language_model.layers")


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


def final_logits(model: torch.nn.Module, hidden_states: torch.Tensor, *, keep: int) -> torch.Tensor:
    final_norm = model.get_submodule("model.language_model.norm")
    lm_head = model.get_submodule("lm_head")
    norm_dtype = next(final_norm.parameters()).dtype
    normed = final_norm(hidden_states[:, -keep:, :].to(dtype=norm_dtype))
    return lm_head(normed)


def tensor_rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt().detach().cpu())


def compute_one_adjoint(
    model: torch.nn.Module,
    *,
    item: dict[str, Any],
    row: dict[str, Any],
    layer_id: int,
    device: torch.device,
    format_ids: torch.Tensor,
    collapse_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    keep = int(row["keep"])
    h_in = item["h_in"].unsqueeze(0).to(device)
    cos = item["cos"].unsqueeze(0).to(device)
    sin = item["sin"].unsqueeze(0).to(device)
    position_ids = item.get("position_ids")
    if position_ids is not None:
        position_ids = position_ids.unsqueeze(0).to(device)
    position_embeddings = (cos, sin)

    with torch.no_grad():
        y = layer_module(model, layer_id)(
            h_in,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        y = y[0] if isinstance(y, tuple) else y
    y = y.detach().requires_grad_(True)

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
    loss = args.logit_weight * logit_loss + args.anti_weight * anti
    (adjoint,) = torch.autograd.grad(loss, y, retain_graph=False, create_graph=False)
    adjoint_last = adjoint[:, -keep:, :].detach().cpu().to(torch.bfloat16).squeeze(0).contiguous()
    target = item["target_last"]
    report = {
        "layer": layer_id,
        "task": row.get("task"),
        "id": row.get("id"),
        "tokens": int(row["tokens"]),
        "positions": keep,
        "loss": float(loss.detach().cpu()),
        "logit": float(logit_loss.detach().cpu()),
        "anti": float(anti.detach().cpu()),
        "target_rms": tensor_rms(target),
        "adjoint_rms": tensor_rms(adjoint_last),
        **stats,
    }
    del y, suffix_out, logits, loss, logit_loss, anti, adjoint
    torch.cuda.empty_cache()
    return adjoint_last, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-patch", action="append", type=Path)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--format-boost", type=float, default=5.0)
    parser.add_argument("--logit-weight", type=float, default=1.0)
    parser.add_argument("--anti-weight", type=float, default=0.1)
    parser.add_argument("--anti-margin", type=float, default=0.05)
    parser.add_argument("--suffix-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("format") not in {"qwen_onpolicy_block_cache_v1", "qwen_onpolicy_block_adjoint_cache_v1"}:
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    if any("top_values" not in row or "top_indices" not in row for row in cache["rows"]):
        raise ValueError("cache must include teacher top-k logits; recollect with the updated block cache collector")
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

    reports: list[dict[str, Any]] = []
    for row_index, row in enumerate(cache["rows"]):
        for layer_id in layers:
            if layer_id not in cache["blocks"][row_index]:
                raise KeyError(f"row {row_index} has no cached layer {layer_id}")
            item = cache["blocks"][row_index][layer_id]
            adjoint, report = compute_one_adjoint(
                model,
                item=item,
                row=row,
                layer_id=layer_id,
                device=device,
                format_ids=format_ids,
                collapse_ids=collapse_ids,
                args=args,
            )
            item["adjoint_last"] = adjoint
            item["adjoint_report"] = report
            reports.append(report)
            print(json.dumps({"event": "adjoint_labeled", "row": row.get("index"), **report}), flush=True)

    cache["format"] = "qwen_onpolicy_block_adjoint_cache_v1"
    cache["source_cache"] = str(args.cache)
    cache["adjoint_layers"] = layers
    cache["adjoint_base_patches"] = [str(path) for path in args.base_patch or []]
    cache["adjoint_patch_reports"] = patch_reports
    cache["adjoint_loss"] = {
        "logit_weight": args.logit_weight,
        "anti_weight": args.anti_weight,
        "anti_margin": args.anti_margin,
        "format_boost": args.format_boost,
        "temperature": args.temperature,
    }
    cache["adjoint_reports"] = reports

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps(
            {
                "format": cache["format"],
                "source_cache": str(args.cache),
                "rows": len(cache["rows"]),
                "layers": layers,
                "reports": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"out": str(args.out), "rows": len(cache["rows"]), "layers": layers}, indent=2))


if __name__ == "__main__":
    main()
