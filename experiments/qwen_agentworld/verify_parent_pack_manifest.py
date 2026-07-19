"""Verify parent-tensor mixed-rate pack artifacts."""

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


def load_weight_patch(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    rows: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            rows[key] = f.get_tensor(key)
    return rows


def patch_key_candidates(tensor_name: str) -> list[str]:
    candidates = [tensor_name]
    language_prefix = "model.language_model."
    model_prefix = "model."
    if tensor_name.startswith(language_prefix):
        candidates.append(model_prefix + tensor_name[len(language_prefix) :])
    elif tensor_name.startswith(model_prefix):
        candidates.append(language_prefix + tensor_name[len(model_prefix) :])
    return candidates


def apply_weight_patch(
    parent: torch.Tensor,
    tensor_name: str,
    weight_patch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, str | None]:
    for key in patch_key_candidates(tensor_name):
        patched = weight_patch.get(key)
        if patched is None:
            continue
        if tuple(patched.shape) != tuple(parent.shape):
            raise ValueError(
                f"patch tensor {key} has shape {tuple(patched.shape)}, "
                f"expected {tuple(parent.shape)} for {tensor_name}"
            )
        return patched.contiguous(), key
    return parent, None


def original_slice(parent: torch.Tensor, group: dict[str, Any]) -> torch.Tensor:
    if "expert" not in group:
        return parent.contiguous()
    return parent[int(group["expert"])].contiguous()


def relative_rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    err = (reconstructed.float() - original.float()).square().mean().sqrt()
    denom = original.float().square().mean().sqrt().clamp_min(1e-8)
    return float((err / denom).item())


def group_size_for_artifact(artifact: Any, group: dict[str, Any]) -> int:
    if group.get("group_size") is not None:
        return int(group["group_size"])
    prefix_key = f"{group['prefix']}_group_size"
    if hasattr(artifact, "files") and prefix_key in artifact.files:
        return int(artifact[prefix_key][0])
    return int(artifact["group_size"][0])


def dequantize_lowbit(artifact: Any, group: dict[str, Any], *, signed_bits: bool) -> torch.Tensor:
    prefix = group["prefix"]
    group_size = group_size_for_artifact(artifact, group)
    pad = int(artifact[f"{prefix}_pad"][0])
    code_shape = tuple(int(x) for x in artifact[f"{prefix}_shape"].tolist())
    scales = torch.from_numpy(artifact[f"{prefix}_scales"].astype(np.float32))
    if group["rate"] in TERNARY_RATES:
        codes_np = unpack_trits_40(artifact[f"{prefix}_packed"], code_shape, pad)
    elif group["rate"] in BINARY_RATES:
        codes_np = unpack_bits(artifact[f"{prefix}_packed"], code_shape, pad, signed=signed_bits)
    else:
        raise ValueError(f"not a low-bit rate: {group['rate']}")
    codes = torch.from_numpy(codes_np).float()
    values = (codes.reshape(-1, group_size) * scales.reshape(-1, 1)).reshape(-1)
    target_shape = tuple(int(x) for x in group["shape"])
    return values[: int(np.prod(target_shape))].reshape(target_shape)


def reconstruct_group(artifact: Any, group: dict[str, Any]) -> torch.Tensor:
    prefix = group["prefix"]
    shape = tuple(int(x) for x in group["shape"])
    if group["rate"] == "bf16":
        return torch.from_numpy(artifact[f"{prefix}_bf16"].copy()).view(torch.bfloat16).reshape(shape).float()
    if group["rate"] == "fp8":
        mode = group.get("fp8_mode")
        if mode == "torch.float8_e4m3fn":
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("torch.float8_e4m3fn is required to verify this artifact")
            return torch.from_numpy(artifact[f"{prefix}_fp8"].copy()).view(torch.float8_e4m3fn).reshape(shape).float()
        if mode == "int8_symmetric_fallback":
            return torch.from_numpy(artifact[f"{prefix}_fp8"].copy()).float().reshape(shape) * float(group["fp8_scale"])
        raise ValueError(f"unsupported fp8 mode: {mode}")
    if group["rate"] in TERNARY_RATES | BINARY_RATES:
        return dequantize_lowbit(artifact, group, signed_bits=True)
    raise ValueError(f"unsupported rate: {group['rate']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inventory", default="/workspace/agentworld_runs/model_inventory.json", type=Path)
    parser.add_argument("--model-dir", default="/workspace/models/Qwen-AgentWorld-35B-A3B", type=Path)
    parser.add_argument("--weight-patch", type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    inventory = json.loads(args.inventory.read_text())
    weight_patch_path = args.weight_patch
    if weight_patch_path is None and manifest.get("weight_patch"):
        weight_patch_path = Path(str(manifest["weight_patch"]))
    weight_patch = load_weight_patch(weight_patch_path)
    rows = []
    max_delta = 0.0
    for parent_row in manifest["packed"]:
        parent = load_tensor(args.model_dir, inventory, parent_row["tensor"])
        parent, patch_key = apply_weight_patch(parent, parent_row["tensor"], weight_patch)
        artifact = np.load(parent_row["artifact"])
        for group in parent_row["groups"]:
            original = original_slice(parent, group)
            reconstructed = reconstruct_group(artifact, group)
            rmse = relative_rmse(original, reconstructed)
            stored = float(group.get("relative_rmse", rmse))
            delta = abs(rmse - stored)
            max_delta = max(max_delta, delta)
            rows.append(
                {
                    "tensor": parent_row["tensor"],
                    "group": group["group"],
                    "rate": group["rate"],
                    "stored_relative_rmse": stored,
                    "verified_relative_rmse": rmse,
                    "delta": delta,
                    "source_bytes": group.get("source_bytes"),
                    "artifact": parent_row["artifact"],
                    "weight_patch_key": patch_key,
                }
            )

    result = {
        "manifest": str(args.manifest),
        "weight_patch": str(weight_patch_path) if weight_patch_path else None,
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
