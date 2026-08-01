#!/usr/bin/env python3
"""Materialize the 5/6/7-bit RSTB precision-boundary arms."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


MIXED_ROOT = Path(__file__).parents[1] / "rstb_mixed_precision_20260731"
sys.path.insert(0, str(MIXED_ROOT))
from materialize_rstb_mixed_precision import (  # noqa: E402
    RSTB_KEYS,
    file_sha256,
    load_source,
    tensor_sha256,
)


FORMAT = "backdropkit-swin2sr-rstb-precision-boundary-materialization-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-rstb-precision-boundary-protocol-v1"
ARM_SPECS = {
    "int5_per_output": {"bits": 5, "granularity": "per_output"},
    "int6_per_output": {"bits": 6, "granularity": "per_output"},
    "int5_per_kernel": {"bits": 5, "granularity": "per_kernel"},
    "int7_per_output": {"bits": 7, "granularity": "per_output"},
    "int6_per_kernel": {"bits": 6, "granularity": "per_kernel"},
}


def quantize_weight_nbit(
    weight: torch.Tensor,
    bits: int,
    granularity: str,
) -> tuple[torch.Tensor, dict]:
    if bits not in (5, 6, 7):
        raise ValueError(f"unsupported boundary bit width: {bits}")
    value = weight.detach().cpu().float().contiguous()
    if value.ndim != 4 or tuple(value.shape[2:]) != (3, 3):
        raise ValueError(f"expected a 3x3 convolution, got {tuple(value.shape)}")
    if granularity == "per_output":
        reduction = (1, 2, 3)
        expected_scales = value.shape[0]
    elif granularity == "per_kernel":
        reduction = (2, 3)
        expected_scales = value.shape[0] * value.shape[1]
    else:
        raise ValueError(f"unsupported granularity: {granularity}")
    limit = (1 << (bits - 1)) - 1
    maximum = value.abs().amax(dim=reduction, keepdim=True)
    scale = (maximum / limit).to(torch.float16).to(torch.float32)
    if bool(((maximum > 0) & (scale == 0)).any()):
        raise RuntimeError("a nonzero quantization scale underflowed FP16")
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(value / safe_scale).clamp(-limit, limit)
    recovered = codes * scale
    if scale.numel() != expected_scales:
        raise RuntimeError(f"unexpected scale count: {scale.numel()} != {expected_scales}")
    error = recovered - value
    code_bytes = math.ceil(value.numel() * bits / 8)
    scale_bytes = scale.numel() * 2
    return recovered, {
        "bits": bits,
        "granularity": granularity,
        "values": value.numel(),
        "scales": scale.numel(),
        "code_bytes": code_bytes,
        "scale_bytes": scale_bytes,
        "data_bytes": code_bytes + scale_bytes,
        "code_min": int(codes.min().item()),
        "code_max": int(codes.max().item()),
        "scale_fp16_roundtrip_exact": bool(
            torch.equal(scale, scale.to(torch.float16).to(torch.float32))
        ),
        "max_abs_error": float(error.abs().max().item()),
        "mean_abs_error": float(error.abs().mean().item()),
    }


def materialize(
    protocol_path: Path,
    ternary_receipt_path: Path,
    fp16_receipt_path: Path,
    arm: str,
    out: Path,
) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported precision-boundary protocol")
    if Path(protocol["sources"]["ternary_materialization"]).resolve() != ternary_receipt_path.resolve():
        raise ValueError("ternary source differs from the sealed protocol")
    if Path(protocol["sources"]["fp16_materialization"]).resolve() != fp16_receipt_path.resolve():
        raise ValueError("FP16 source differs from the sealed protocol")
    if arm not in ARM_SPECS or arm not in protocol["arms"]:
        raise ValueError(f"unsupported boundary arm: {arm}")
    ternary_receipt, ternary = load_source(ternary_receipt_path, "ternary")
    fp16_receipt, fp16 = load_source(fp16_receipt_path, "fp16")
    if ternary.keys() != fp16.keys():
        raise RuntimeError("source state dictionaries differ")
    if ternary_receipt["parent_checkpoint_sha256"] != fp16_receipt["parent_checkpoint_sha256"]:
        raise RuntimeError("source arms have different parents")
    undeclared = [
        name for name in ternary
        if name not in RSTB_KEYS and not torch.equal(ternary[name], fp16[name])
    ]
    if undeclared:
        raise RuntimeError(f"FP16 source changed undeclared tensors: {undeclared[:8]}")

    result = {name: value.detach().cpu().clone() for name, value in ternary.items()}
    tensors = []
    for key in RSTB_KEYS:
        recovered, detail = quantize_weight_nbit(fp16[key], **ARM_SPECS[arm])
        result[key] = recovered
        tensors.append({"name": key, **detail})
    changed = [name for name in result if not torch.equal(result[name], ternary[name])]
    if set(changed) != set(RSTB_KEYS):
        raise RuntimeError(f"boundary state changed unexpected tensors: {changed}")

    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "dense-hard.pt"
    torch.save(result, state_path)
    data_bytes = sum(item["data_bytes"] for item in tensors)
    ternary_bytes = int(protocol["storage"]["ternary_rstb_data_bytes"])
    receipt = {
        "format": FORMAT,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": file_sha256(protocol_path),
        "arm": arm,
        "quantization": ARM_SPECS[arm],
        "sources": {
            "ternary_materialization_sha256": file_sha256(ternary_receipt_path),
            "fp16_materialization_sha256": file_sha256(fp16_receipt_path),
        },
        "tensors": tensors,
        "changed_tensors": changed,
        "storage": {
            "data_bytes": data_bytes,
            "ternary_rstb_data_bytes": ternary_bytes,
            "incremental_data_bytes": data_bytes - ternary_bytes,
        },
        "dense_state": {
            "path": str(state_path.resolve()),
            "file_sha256": file_sha256(state_path),
            "tensor_sha256": tensor_sha256(result),
            "tensors": len(result),
            "values": sum(value.numel() for value in result.values()),
        },
    }
    manifest = out / "materialization.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ternary-materialization", type=Path, required=True)
    parser.add_argument("--fp16-materialization", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_SPECS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        args.protocol,
        args.ternary_materialization,
        args.fp16_materialization,
        args.arm,
        args.out,
    ), indent=2))


if __name__ == "__main__":
    main()
