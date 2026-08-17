"""Cache local Qwen MLP input/output pairs from BF16 teacher runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from train_mlp_triplet_survival_qat import load_model


LAYER_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp$")


def parse_layers(value: str) -> set[int] | None:
    if value == "all":
        return None
    layers = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        layers.add(int(part))
    return layers


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path, help="teacher behavior cache with input_ids rows")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layers", default="all", help="comma layer ids or all")
    parser.add_argument("--rows-limit", type=int, default=0)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()

    if args.positions <= 0:
        raise ValueError("positions must be positive")
    if args.torch_threads < 0:
        raise ValueError("torch-threads must be non-negative")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    wanted_layers = parse_layers(args.layers)
    cache = torch.load(args.cache, map_location="cpu")
    if cache.get("format") != "qwen_agentworld_teacher_behavior_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")

    model = load_model(args.model_dir, device=args.device)
    model.eval()

    captured: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(lambda: {"x": [], "y": []})
    current_keep = {"value": args.positions}
    handles = []

    def make_hook(layer_id: int):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            keep = int(current_keep["value"])
            x = inputs[0]
            y = output[0] if isinstance(output, tuple) else output
            captured[layer_id]["x"].append(x.detach()[:, -keep:, :].reshape(-1, x.shape[-1]).cpu().to(torch.bfloat16))
            captured[layer_id]["y"].append(y.detach()[:, -keep:, :].reshape(-1, y.shape[-1]).cpu().to(torch.bfloat16))

        return hook

    for name, module in model.named_modules():
        match = LAYER_RE.match(name)
        if not match:
            continue
        layer_id = int(match.group(1))
        if wanted_layers is not None and layer_id not in wanted_layers:
            continue
        handles.append(module.register_forward_hook(make_hook(layer_id)))

    if not handles:
        raise RuntimeError("no MLP hooks registered")

    device = first_parameter_device(model)
    rows = cache["rows"][: args.rows_limit or None]
    row_meta = []
    try:
        for row in rows:
            keep = min(args.positions, int(row["top_values"].shape[0]))
            current_keep["value"] = keep
            input_ids = row["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0)
            attention_mask = row["attention_mask"].to(device=device, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )
            row_meta.append({"task": row.get("task"), "id": row.get("id"), "tokens": int(row["tokens"]), "positions": keep})
            print(json.dumps(row_meta[-1]), flush=True)
    finally:
        for handle in handles:
            handle.remove()

    layers: dict[int, dict[str, torch.Tensor]] = {}
    for layer_id, tensors in captured.items():
        layers[layer_id] = {
            "x": torch.cat(tensors["x"], dim=0).contiguous(),
            "y": torch.cat(tensors["y"], dim=0).contiguous(),
        }

    payload = {
        "format": "qwen_mlp_activation_cache_v1",
        "model_dir": str(args.model_dir),
        "source_cache": str(args.cache),
        "source_label_semantics": cache.get("label_semantics"),
        "runtime_device": str(device),
        "torch_threads": torch.get_num_threads(),
        "rows": row_meta,
        "layers": layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, args.out)
    print(json.dumps({"out": str(args.out), "rows": len(row_meta), "layers": sorted(layers)}, indent=2))


if __name__ == "__main__":
    main()
