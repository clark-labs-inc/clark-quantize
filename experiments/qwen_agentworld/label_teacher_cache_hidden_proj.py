"""Attach projected FP16 teacher hidden labels to Qwen teacher-cache shards.

The global all-2D QAT trainer can consume either full ``teacher_hidden`` values
or compact ``teacher_hidden_proj`` anchors. This script adds the compact form
to existing ``qwen_agentworld_teacher_behavior_v1`` shards without rebuilding
top-k logits or multimodal processor payloads.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_onpolicy_survival_cache import parse_layers
from train_mlp_triplet_survival_qat import load_model, register_hidden_hooks, row_to_inputs


def hidden_projection(
    *,
    layer_id: int,
    hidden_dim: int,
    projected_dim: int,
    seed: int,
    device: torch.device,
    cache: dict[tuple[int, int, int], torch.Tensor],
) -> torch.Tensor:
    key = (layer_id, hidden_dim, projected_dim)
    projection = cache.get(key)
    if projection is not None and projection.device == device:
        return projection
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer_id * 1009 + hidden_dim * 9176 + projected_dim)
    projection_cpu = torch.empty(hidden_dim, projected_dim, dtype=torch.float32)
    projection_cpu.bernoulli_(0.5, generator=generator)
    projection_cpu.mul_(2.0).sub_(1.0).div_(math.sqrt(projected_dim))
    projection = projection_cpu.to(device=device)
    cache[key] = projection
    return projection


@torch.no_grad()
def label_rows(
    teacher: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    hidden_layers: list[int],
    projected_dim: int,
    seed: int,
) -> None:
    device = next(teacher.parameters()).device
    keep_ref = {"keep": 1}
    handles, captured = register_hidden_hooks(teacher, hidden_layers, keep_ref)
    projection_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    try:
        for index, row in enumerate(rows):
            keep = int(row["keep"])
            keep_ref["keep"] = keep
            captured.clear()
            teacher(**row_to_inputs(row, device), use_cache=False, logits_to_keep=keep)
            projected: dict[int, torch.Tensor] = {}
            for layer_id in hidden_layers:
                hidden = captured[layer_id].float()
                projection = hidden_projection(
                    layer_id=layer_id,
                    hidden_dim=hidden.shape[-1],
                    projected_dim=projected_dim,
                    seed=seed,
                    device=device,
                    cache=projection_cache,
                )
                hidden_norm = F.layer_norm(hidden, hidden.shape[-1:])
                projected[layer_id] = (hidden_norm @ projection).cpu().to(torch.float16).contiguous()
            row["teacher_hidden_proj"] = projected
            row.pop("teacher_hidden", None)
            print(
                json.dumps(
                    {
                        "event": "hidden_projected",
                        "row": index,
                        "id": row.get("id"),
                        "tokens": int(row["tokens"]),
                        "keep": keep,
                        "layers": hidden_layers,
                        "projected_dim": projected_dim,
                    }
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache-glob", action="append", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hidden-layers", default="0,1,8,12,20,27,30,31")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=20260701)
    parser.add_argument("--limit-shards", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    paths: list[Path] = []
    for pattern in args.cache_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    paths = list(dict.fromkeys(paths))
    if args.limit_shards > 0:
        paths = paths[: args.limit_shards]
    if not paths:
        raise RuntimeError("no cache shards matched")
    hidden_layers = parse_layers(args.hidden_layers)
    if not hidden_layers:
        raise RuntimeError("no hidden layers requested")

    teacher = load_model(args.model_dir)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        cache = torch.load(path, map_location="cpu")
        if cache.get("format") not in {"qwen_agentworld_teacher_behavior_v1", "qwen_onpolicy_survival_cache_v1"}:
            raise ValueError(f"unsupported cache format in {path}: {cache.get('format')!r}")
        label_rows(
            teacher,
            cache["rows"],
            hidden_layers=hidden_layers,
            projected_dim=args.projection_dim,
            seed=args.projection_seed,
        )
        cache["hidden_layers"] = hidden_layers
        cache["hidden_projection_dim"] = args.projection_dim
        cache["hidden_projection_seed"] = args.projection_seed
        cache["hidden_label_source"] = {
            "model_dir": str(args.model_dir),
            "input_cache": str(path),
            "projection": "layer_norm_rademacher",
        }
        out = args.out_dir / path.name
        torch.save(cache, out)
        meta = {key: value for key, value in cache.items() if key != "rows"}
        meta["rows"] = [
            {
                "task": row.get("task"),
                "id": row.get("id"),
                "tokens": int(row["tokens"]),
                "keep": int(row["keep"]),
                "has_teacher_hidden_proj": bool(row.get("teacher_hidden_proj")),
            }
            for row in cache["rows"]
        ]
        out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps({"out": str(out), "rows": len(cache["rows"])}), flush=True)


if __name__ == "__main__":
    main()
