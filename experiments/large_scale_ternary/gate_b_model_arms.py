"""Deterministic model transformations for non-trained Gate B controls."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from clark_quantize.ternary_kd import (
    AbsmeanTernaryLinear,
    ternarize_qwen35,
)


def _parent_and_leaf(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


@torch.no_grad()
def apply_ptq_ternary_dense(
    model: nn.Module, *, dtype: torch.dtype = torch.bfloat16
) -> dict[str, Any]:
    """Apply the exact QAT surgery map at step zero and materialize dense weights.

    Materialization removes float32 master weights and ensures the evaluator
    measures the deployed absmean-dequant function, not an STE wrapper.
    """
    surgery = ternarize_qwen35(model, ptq_embed=True, train_norms=False)
    materialized = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, AbsmeanTernaryLinear):
            continue
        replacement = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight_master.device,
            dtype=dtype,
        )
        replacement.weight.copy_(module.quantized_weight(dtype))
        if module.bias is not None:
            replacement.bias.copy_(module.bias.to(dtype))
        replacement.requires_grad_(False)
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, replacement)
        materialized.append(name)
    expected = sorted(surgery["ternarized"] + surgery["head"])
    if sorted(materialized) != expected:
        raise ValueError(
            f"PTQ materialization mismatch: {materialized} != {expected}"
        )
    return {
        "method": "qwen35_absmean_ptq_dense_v1",
        "group_size": 128,
        "codebook": [-1, 0, 1],
        "ternary_modules": materialized,
        "ptq_embedding": True,
        "continuous_islands": surgery["frozen"],
    }
