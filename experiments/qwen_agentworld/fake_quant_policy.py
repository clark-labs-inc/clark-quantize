"""Apply a Clark Air policy to a loaded Qwen model as in-place fake quant.

The result is still stored as normal torch weights, so this does not reduce GPU
memory. It is a behavioral validation path: run the model with weights snapped
to the binary/ternary policy before building the true packed runtime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant import quantize_binary, quantize_ternary_mse


TERNARY_RATES = {"ternary40_log8", "ternary40_fp16"}
BINARY_RATES = {"binary_log8", "binary_fp16"}
PACKED_RATES = {"fp8", *TERNARY_RATES, *BINARY_RATES}


def _parse_csv(values: str | None) -> set[str] | None:
    if not values:
        return None
    return {value.strip() for value in values.split(",") if value.strip()}


def _choice_allowed(
    choice: dict[str, Any],
    *,
    include_experts: bool,
    allowed_rates: set[str] | None,
    allowed_buckets: set[str] | None,
) -> bool:
    meta = choice.get("meta") or {}
    if choice["rate"] not in PACKED_RATES:
        return False
    if "::expert_" in choice["group"] and not include_experts:
        return False
    if allowed_rates is not None and choice["rate"] not in allowed_rates:
        return False
    if allowed_buckets is not None and meta.get("bucket") not in allowed_buckets:
        return False
    return True


def _iter_choices(
    policy: dict[str, Any],
    *,
    include_experts: bool,
    allowed_rates: set[str] | None,
    allowed_buckets: set[str] | None,
    max_groups: int | None,
) -> Iterable[dict[str, Any]]:
    yielded = 0
    for choice in policy["choices"]:
        if not _choice_allowed(
            choice,
            include_experts=include_experts,
            allowed_rates=allowed_rates,
            allowed_buckets=allowed_buckets,
        ):
            continue
        yield choice
        yielded += 1
        if max_groups is not None and yielded >= max_groups:
            return


def _quantized(weight: torch.Tensor, rate: str, group_size: int) -> torch.Tensor:
    if rate == "fp8":
        if hasattr(torch, "float8_e4m3fn"):
            return weight.detach().float().to(torch.float8_e4m3fn).to(device=weight.device, dtype=weight.dtype)
        source = weight.detach().float()
        scale = source.abs().amax().clamp_min(1e-8) / 127.0
        quant = torch.round(source / scale).clamp(-127, 127)
        return (quant * scale).to(device=weight.device, dtype=weight.dtype)
    source = weight.detach().float()
    if rate in TERNARY_RATES:
        return quantize_ternary_mse(source, group_size=group_size).dequantize().to(device=weight.device, dtype=weight.dtype)
    if rate in BINARY_RATES:
        return quantize_binary(source, group_size=group_size).dequantize().to(device=weight.device, dtype=weight.dtype)
    raise ValueError(f"unsupported fake-quant rate: {rate}")


def choice_group_size(choice: dict[str, Any], default_group_size: int) -> int:
    meta = choice.get("meta") or {}
    explicit = choice.get("group_size", meta.get("group_size"))
    size = int(explicit) if explicit is not None else int(default_group_size)
    if size <= 0:
        raise ValueError(f"group size must be positive for {choice.get('group')}: {size}")
    return size


def resolve_parameter_name(tensor_name: str, params: dict[str, torch.nn.Parameter]) -> str | None:
    """Map safetensors policy names to loaded Transformers parameter names."""

    if tensor_name in params:
        return tensor_name
    rewrites = []
    if tensor_name.startswith("model.language_model."):
        rewrites.append("model." + tensor_name[len("model.language_model.") :])
    if tensor_name.startswith("model."):
        rewrites.append("model.language_model." + tensor_name[len("model.") :])
    for candidate in rewrites:
        if candidate in params:
            return candidate
    return None


def _relative_rmse(before: torch.Tensor, after: torch.Tensor) -> float:
    err = (after.float() - before.float()).square().mean().sqrt()
    denom = before.float().square().mean().sqrt().clamp_min(1e-8)
    return float((err / denom).item())


@torch.no_grad()
def apply_fake_quant_policy(
    model: torch.nn.Module,
    policy_path: str | Path,
    *,
    max_groups: int | None = None,
    include_experts: bool = False,
    rates: str | None = None,
    buckets: str | None = None,
    out_path: str | Path | None = None,
    measure_rmse: bool = True,
) -> dict[str, Any]:
    """Apply selected low-bit policy groups to a loaded model in-place."""

    policy = json.loads(Path(policy_path).read_text())
    group_size = int(policy.get("group_size", 112))
    params = dict(model.named_parameters())
    allowed_rates = _parse_csv(rates)
    allowed_buckets = _parse_csv(buckets)

    applied = []
    missing = []
    for choice in _iter_choices(
        policy,
        include_experts=include_experts,
        allowed_rates=allowed_rates,
        allowed_buckets=allowed_buckets,
        max_groups=max_groups,
    ):
        meta = choice.get("meta") or {}
        effective_group_size = choice_group_size(choice, group_size)
        tensor_name = meta.get("tensor")
        parameter_name = resolve_parameter_name(str(tensor_name), params) if tensor_name else None
        if parameter_name is None:
            missing.append({"group": choice["group"], "tensor": tensor_name, "reason": "parameter not found"})
            continue

        param = params[parameter_name]
        if "expert" in meta:
            expert = int(meta["expert"])
            target = param.data[expert]
            before = target.detach().float().cpu() if measure_rmse else None
            target.copy_(_quantized(target, choice["rate"], effective_group_size))
            after = target.detach().float().cpu() if measure_rmse else None
        else:
            before = param.data.detach().float().cpu() if measure_rmse else None
            param.data.copy_(_quantized(param.data, choice["rate"], effective_group_size))
            after = param.data.detach().float().cpu() if measure_rmse else None

        row = {
            "group": choice["group"],
            "tensor": tensor_name,
            "parameter": parameter_name,
            "rate": choice["rate"],
            "group_size": effective_group_size,
            "bucket": meta.get("bucket"),
            "expert": meta.get("expert"),
            "params": int(meta.get("params", param.numel())),
        }
        if measure_rmse and before is not None and after is not None:
            row["relative_rmse"] = _relative_rmse(before, after)
        applied.append(row)

    result = {
        "policy": str(policy_path),
        "group_size": group_size,
        "requested_max_groups": max_groups,
        "include_experts": include_experts,
        "rates": sorted(allowed_rates) if allowed_rates else None,
        "buckets": sorted(allowed_buckets) if allowed_buckets else None,
        "measure_rmse": measure_rmse,
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
