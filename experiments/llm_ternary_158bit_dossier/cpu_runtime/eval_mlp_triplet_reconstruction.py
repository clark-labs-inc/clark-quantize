#!/usr/bin/env python3
"""Compare one exact Qwen MLP patch with BF16 and direct ternary PTQ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clark_quantize.qwen_qat import hard_ternary_weight
from experiments.qwen_agentworld.train_mlp_triplet_qat import (
    activation_position_indices,
    layer_tensor_name,
    load_tensor,
    load_weight_patch,
    parse_row_indices,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def normalized_loss_from_weights(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> float:
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("evaluation tensors must be nonempty, equally shaped matrices")
    weights = {name: value.float() for name, value in weights.items()}
    squared_error = 0.0
    squared_target = 0.0
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        xb = x[start:stop].float()
        yb = y[start:stop].float()
        gate = F.linear(xb, weights["gate"])
        up = F.linear(xb, weights["up"])
        prediction = F.linear(F.silu(gate) * up, weights["down"])
        squared_error += float((prediction - yb).square().sum())
        squared_target += float(yb.square().sum())
    return squared_error / max(squared_target, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--activation-cache", required=True, type=Path)
    parser.add_argument("--candidate-patch", required=True, type=Path)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--train-row-indices", required=True)
    parser.add_argument("--validation-row-indices", required=True)
    parser.add_argument("--test-row-indices", required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if min(args.group_size, args.batch_size, args.torch_threads) <= 0:
        raise ValueError("group size, batch size, and torch threads must be positive")
    torch.set_num_threads(args.torch_threads)

    cache = torch.load(args.activation_cache, map_location="cpu")
    if cache.get("format") != "qwen_mlp_activation_cache_v1":
        raise ValueError(f"unsupported activation cache: {cache.get('format')!r}")
    payload = cache["layers"][args.layer]
    rows: list[dict[str, Any]] = cache["rows"]
    split_specs = {
        "train": parse_row_indices(args.train_row_indices, len(rows), allow_empty=False),
        "validation": parse_row_indices(
            args.validation_row_indices, len(rows), allow_empty=False
        ),
        "test": parse_row_indices(args.test_row_indices, len(rows), allow_empty=False),
    }
    selected_sets = {name: set(indices) for name, indices in split_specs.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(selected_sets[left] & selected_sets[right])
        if overlap:
            raise ValueError(f"{left} and {right} rows overlap: {overlap}")

    inventory = json.loads(args.inventory.read_text())
    candidate_patch = load_weight_patch(args.candidate_patch)
    bf16 = {
        projection: load_tensor(
            args.model_dir,
            inventory,
            layer_tensor_name(args.layer, projection),
            {},
        ).to(torch.bfloat16)
        for projection in ("gate", "up", "down")
    }
    candidate = {
        projection: load_tensor(
            args.model_dir,
            inventory,
            layer_tensor_name(args.layer, projection),
            candidate_patch,
        ).to(torch.bfloat16)
        for projection in ("gate", "up", "down")
    }
    direct_ptq = {
        projection: hard_ternary_weight(
            bf16[projection].float(),
            group_size=args.group_size,
            row_batch=128,
        ).to(torch.bfloat16)
        for projection in ("gate", "up", "down")
    }
    candidate_exact_ternary = all(
        torch.equal(
            candidate[projection],
            hard_ternary_weight(
                candidate[projection].float(),
                group_size=args.group_size,
                row_batch=128,
            ).to(torch.bfloat16),
        )
        for projection in ("gate", "up", "down")
    )
    if not candidate_exact_ternary:
        raise RuntimeError("candidate patch is not idempotent under the exact ternary contract")

    split_reports: dict[str, Any] = {}
    for split_name, row_indices in split_specs.items():
        positions = activation_position_indices(rows, row_indices, int(payload["x"].shape[0]))
        x = payload["x"].index_select(0, positions)
        y = payload["y"].index_select(0, positions)
        losses = {
            label: normalized_loss_from_weights(
                x, y, weights, batch_size=args.batch_size
            )
            for label, weights in (
                ("bf16", bf16),
                ("direct_ptq", direct_ptq),
                ("candidate", candidate),
            )
        }
        split_reports[split_name] = {
            "row_indices": row_indices,
            "rows": [rows[index] for index in row_indices],
            "positions": int(positions.numel()),
            "normalized_loss": losses,
            "candidate_relative_improvement_vs_direct_ptq_percent": 100.0
            * (1.0 - losses["candidate"] / losses["direct_ptq"]),
        }

    result = {
        "schema": "qwen_mlp_triplet_reconstruction_eval_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_label": args.candidate_label,
        "candidate_patch": str(args.candidate_patch),
        "candidate_patch_sha256": sha256_file(args.candidate_patch),
        "activation_cache": str(args.activation_cache),
        "activation_cache_sha256": sha256_file(args.activation_cache),
        "inventory_sha256": sha256_file(args.inventory),
        "model_index_sha256": sha256_file(args.model_dir / "model.safetensors.index.json"),
        "layer": args.layer,
        "group_size": args.group_size,
        "candidate_exact_ternary": candidate_exact_ternary,
        "splits": split_reports,
        "independent_test_improves_direct_ptq": split_reports["test"]
        ["normalized_loss"]["candidate"]
        < split_reports["test"]["normalized_loss"]["direct_ptq"],
    }
    result["report_content_sha256"] = canonical_digest(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
