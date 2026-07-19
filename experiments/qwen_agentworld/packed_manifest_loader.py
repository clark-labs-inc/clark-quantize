"""Load Clark Air parent-pack artifacts into a live Qwen model.

This is a behavioral/runtime bridge: it reconstructs selected packed groups
from the parent-pack manifest and copies them into the loaded Transformers
model. It does not yet reduce GPU memory because the destination is still the
normal model parameter storage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake_quant_policy import _parse_csv, resolve_parameter_name
from verify_parent_pack_manifest import reconstruct_group


def parse_rate_scales(values: Iterable[str] | None) -> dict[str, float]:
    scales: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"rate scale must be RATE=FACTOR, got {value!r}")
        rate, factor_text = value.split("=", 1)
        rate = rate.strip()
        if not rate:
            raise ValueError(f"empty rate in scale override {value!r}")
        factor = float(factor_text)
        if factor <= 0:
            raise ValueError(f"scale factor must be positive for {rate!r}: {factor}")
        scales[rate] = factor
    return scales


def _group_allowed(
    group: dict[str, Any],
    *,
    include_experts: bool,
    allowed_rates: set[str] | None,
    allowed_buckets: set[str] | None,
) -> bool:
    if "expert" in group and not include_experts:
        return False
    if allowed_rates is not None and group.get("rate") not in allowed_rates:
        return False
    if allowed_buckets is not None and group.get("bucket") not in allowed_buckets:
        return False
    return True


def iter_manifest_groups(
    manifest: dict[str, Any],
    *,
    include_experts: bool,
    allowed_rates: set[str] | None,
    allowed_buckets: set[str] | None,
    max_groups: int | None,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    yielded = 0
    for parent in manifest["packed"]:
        for group in parent["groups"]:
            if not _group_allowed(
                group,
                include_experts=include_experts,
                allowed_rates=allowed_rates,
                allowed_buckets=allowed_buckets,
            ):
                continue
            yield parent, group
            yielded += 1
            if max_groups is not None and yielded >= max_groups:
                return


def relative_rmse(before: torch.Tensor, after: torch.Tensor) -> float:
    err = (after.float() - before.float()).square().mean().sqrt()
    denom = before.float().square().mean().sqrt().clamp_min(1e-8)
    return float((err / denom).item())


@torch.no_grad()
def apply_packed_manifest(
    model: torch.nn.Module,
    manifest_path: str | Path,
    *,
    max_groups: int | None = None,
    include_experts: bool = False,
    rates: str | None = None,
    buckets: str | None = None,
    scale_by_rate: dict[str, float] | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    params = dict(model.named_parameters())
    allowed_rates = _parse_csv(rates)
    allowed_buckets = _parse_csv(buckets)
    scale_by_rate = scale_by_rate or {}

    applied = []
    missing = []
    artifact_cache: dict[str, Any] = {}
    try:
        for parent, group in iter_manifest_groups(
            manifest,
            include_experts=include_experts,
            allowed_rates=allowed_rates,
            allowed_buckets=allowed_buckets,
            max_groups=max_groups,
        ):
            tensor_name = group.get("tensor") or parent.get("tensor")
            parameter_name = resolve_parameter_name(str(tensor_name), params) if tensor_name else None
            if parameter_name is None:
                missing.append({"group": group.get("group"), "tensor": tensor_name, "reason": "parameter not found"})
                continue

            artifact_path = str(parent["artifact"])
            artifact = artifact_cache.get(artifact_path)
            if artifact is None:
                artifact = np.load(artifact_path)
                artifact_cache[artifact_path] = artifact

            param = params[parameter_name]
            reconstructed = reconstruct_group(artifact, group).to(device=param.device, dtype=param.dtype)
            scale_factor = float(scale_by_rate.get(str(group.get("rate")), 1.0))
            if scale_factor != 1.0:
                reconstructed = reconstructed * scale_factor
            if "expert" in group:
                target = param.data[int(group["expert"])]
            else:
                target = param.data
            before = target.detach().float().cpu()
            target.copy_(reconstructed)
            after = target.detach().float().cpu()
            applied.append(
                {
                    "group": group.get("group"),
                    "tensor": tensor_name,
                    "parameter": parameter_name,
                    "rate": group.get("rate"),
                    "bucket": group.get("bucket"),
                    "expert": group.get("expert"),
                    "params": int(group.get("params", target.numel())),
                    "scale_factor": scale_factor,
                    "artifact": artifact_path,
                    "stored_relative_rmse": group.get("relative_rmse"),
                    "applied_relative_rmse": relative_rmse(before, after),
                }
            )
    finally:
        for artifact in artifact_cache.values():
            artifact.close()

    result = {
        "manifest": str(manifest_path),
        "requested_max_groups": max_groups,
        "include_experts": include_experts,
        "rates": sorted(allowed_rates) if allowed_rates else None,
        "buckets": sorted(allowed_buckets) if allowed_buckets else None,
        "scale_by_rate": scale_by_rate,
        "applied_count": len(applied),
        "missing_count": len(missing),
        "applied": applied,
        "missing": missing,
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    return result
