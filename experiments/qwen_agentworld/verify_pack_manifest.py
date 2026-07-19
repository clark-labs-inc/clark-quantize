"""Verify Clark Air pack-smoke artifacts by loading and reconstructing them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant import unpack_bits, unpack_trits_40

TERNARY_RATES = {"ternary40_log8", "ternary40_fp16"}
BINARY_RATES = {"binary_log8", "binary_fp16"}


def load_tensor(model_dir: Path, inventory: dict[str, Any], tensor_name: str) -> torch.Tensor:
    info = inventory["tensor_summary"][tensor_name]
    with safe_open(model_dir / info["shard"], framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def slice_for_row(weight: torch.Tensor, row: dict[str, Any]) -> torch.Tensor:
    if "expert" not in row:
        return weight
    return weight[int(row["expert"])].contiguous()


def reconstruct(row: dict[str, Any]) -> torch.Tensor:
    artifact = np.load(row["artifact"])
    group_size = int(artifact["group_size"][0])
    pad = int(artifact["pad"][0])
    code_shape = tuple(int(x) for x in artifact["shape"].tolist())
    scales = torch.from_numpy(artifact["scales"].astype(np.float32))

    if row["rate"] in TERNARY_RATES:
        codes = torch.from_numpy(unpack_trits_40(artifact["packed"], code_shape, pad)).float()
    elif row["rate"] in BINARY_RATES:
        codes = torch.from_numpy(unpack_bits(artifact["packed"], code_shape, pad, signed=True)).float()
    else:
        raise ValueError(f"unsupported packed rate: {row['rate']}")

    values = (codes.reshape(-1, group_size) * scales.reshape(-1, 1)).reshape(-1)
    target_shape = tuple(int(x) for x in row["shape"])
    return values[: int(np.prod(target_shape))].reshape(target_shape)


def relative_rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    err = (reconstructed.float() - original.float()).square().mean().sqrt()
    denom = original.float().square().mean().sqrt().clamp_min(1e-8)
    return float((err / denom).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inventory", default="/workspace/agentworld_runs/model_inventory.json", type=Path)
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    inventory = json.loads(args.inventory.read_text())

    rows = []
    max_delta = 0.0
    for row in manifest["packed"]:
        original = slice_for_row(load_tensor(args.model_dir, inventory, row["tensor"]), row)
        reconstructed = reconstruct(row)
        rmse = relative_rmse(original, reconstructed)
        delta = abs(rmse - float(row.get("relative_rmse", rmse)))
        max_delta = max(max_delta, delta)
        rows.append(
            {
                "group": row["group"],
                "rate": row["rate"],
                "stored_relative_rmse": row.get("relative_rmse"),
                "verified_relative_rmse": rmse,
                "delta": delta,
                "artifact_bytes": row.get("artifact_bytes"),
                "source_bytes": row.get("source_bytes"),
            }
        )

    result = {
        "manifest": str(args.manifest),
        "verified": len(rows),
        "max_rmse_delta": max_delta,
        "ok": max_delta < 1e-5,
        "rows": rows,
    }
    out = args.out or args.manifest.with_name("verify.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(out), "verified": len(rows), "ok": result["ok"], "max_rmse_delta": max_delta}, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
