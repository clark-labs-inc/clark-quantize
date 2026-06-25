from __future__ import annotations

from typing import Iterable, Sequence

import torch


def torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compact_checkpoint(payload):
    if isinstance(payload, dict) and "quant" in payload:
        return payload["quant"]
    return payload


def load_compact_checkpoint(path: str):
    return compact_checkpoint(torch_load_cpu(path))


def item_names(items: Iterable) -> list[str] | None:
    names = []
    for item in items:
        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], str):
            names.append(item[0])
        else:
            return None
    return names


def island_values(ck: dict, island_items: Sequence):
    islands = ck.get("islands", {})
    if isinstance(islands, dict):
        names = item_names(island_items)
        if names is None:
            raise ValueError("checkpoint islands are keyed by name; pass named island parameters")
        missing = [name for name in names if name not in islands]
        if missing:
            raise ValueError(f"checkpoint missing {len(missing)} island tensors, first={missing[0]}")
        return [islands[name] for name in names]

    vals = list(islands)
    if len(vals) != len(island_items):
        raise ValueError(f"island count mismatch {len(vals)} != {len(island_items)}")
    return vals


def validate_quant_names(ck: dict, names: Sequence[str]) -> None:
    expected = ck.get("quant_names")
    if expected and list(expected) != list(names):
        raise ValueError("checkpoint quant module order does not match current Sana model")

