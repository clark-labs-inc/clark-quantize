"""Apply named tensor patches to a loaded Transformers model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from fake_quant_policy import resolve_parameter_name


@torch.no_grad()
def apply_weight_patch(
    model: torch.nn.Module,
    patch_path: str | Path,
    *,
    out_path: str | Path | None = None,
    exclude_regex: str | None = None,
    include_regex: str | None = None,
) -> dict[str, Any]:
    """Copy tensors from a safetensors patch into matching model parameters."""

    patch_path = Path(patch_path)
    params = dict(model.named_parameters())
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    include_re = re.compile(include_regex) if include_regex else None
    applied = []
    missing = []
    skipped = []
    with safe_open(patch_path, framework="pt", device="cpu") as handle:
        for tensor_name in handle.keys():
            if exclude_re is not None and exclude_re.search(tensor_name):
                skipped.append({"tensor": tensor_name, "reason": "exclude_regex"})
                continue
            parameter_name = resolve_parameter_name(tensor_name, params)
            if parameter_name is None:
                missing.append({"tensor": tensor_name, "reason": "parameter not found"})
                continue
            if include_re is not None and not (
                include_re.search(tensor_name) or include_re.search(parameter_name)
            ):
                skipped.append({"tensor": tensor_name, "parameter": parameter_name, "reason": "include_regex"})
                continue
            if exclude_re is not None and exclude_re.search(parameter_name):
                skipped.append({"tensor": tensor_name, "parameter": parameter_name, "reason": "exclude_regex"})
                continue
            param = params[parameter_name]
            tensor = handle.get_tensor(tensor_name)
            if tuple(tensor.shape) != tuple(param.shape):
                missing.append(
                    {
                        "tensor": tensor_name,
                        "parameter": parameter_name,
                        "reason": "shape mismatch",
                        "patch_shape": list(tensor.shape),
                        "parameter_shape": list(param.shape),
                    }
                )
                continue
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            applied.append(
                {
                    "tensor": tensor_name,
                    "parameter": parameter_name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "params": int(tensor.numel()),
                }
            )

    result = {
        "patch": str(patch_path),
        "applied_count": len(applied),
        "missing_count": len(missing),
        "skipped_count": len(skipped),
        "applied": applied,
        "missing": missing,
        "skipped": skipped,
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    return result
