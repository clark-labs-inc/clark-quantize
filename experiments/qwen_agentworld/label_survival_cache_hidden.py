"""Add FP16 teacher hidden-state labels to an existing survival cache.

This preserves the student-generated prefixes already stored in the cache.
Use it when a generation gate is the authoritative candidate and we want
hidden anchors without re-running student generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_onpolicy_survival_cache import load_model, parse_layers, register_hidden_hooks


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def label_hidden(
    teacher: torch.nn.Module,
    rows: list[dict[str, Any]],
    hidden_layers: list[int],
) -> None:
    device = first_parameter_device(teacher)
    keep_ref = {"keep": 1}
    handles, hidden_capture = register_hidden_hooks(teacher, hidden_layers, keep_ref)
    try:
        for row in rows:
            keep = int(row["keep"])
            keep_ref["keep"] = keep
            hidden_capture.clear()
            inputs = {
                "input_ids": row["input_ids"].unsqueeze(0).to(device),
                "attention_mask": row["attention_mask"].unsqueeze(0).to(device),
            }
            teacher(**inputs, use_cache=False, logits_to_keep=keep)
            row["teacher_hidden"] = {
                layer: hidden_capture[layer].contiguous()
                for layer in hidden_layers
            }
            print(
                {
                    "event": "hidden_labeled",
                    "index": row.get("index"),
                    "tokens": int(row["tokens"]),
                    "keep": keep,
                    "hidden_layers": hidden_layers,
                },
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hidden-layers", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cache = torch.load(args.cache, map_location="cpu")
    if cache.get("format") != "qwen_onpolicy_survival_cache_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    hidden_layers = parse_layers(args.hidden_layers)
    if not hidden_layers:
        raise RuntimeError("no hidden layers requested")

    teacher = load_model(args.model_dir)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    label_hidden(teacher, cache["rows"], hidden_layers)
    cache["hidden_layers"] = hidden_layers
    cache["hidden_label_source"] = {
        "model_dir": str(args.model_dir),
        "input_cache": str(args.cache),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    print({"out": str(args.out), "rows": len(cache["rows"]), "hidden_layers": hidden_layers})


if __name__ == "__main__":
    main()
