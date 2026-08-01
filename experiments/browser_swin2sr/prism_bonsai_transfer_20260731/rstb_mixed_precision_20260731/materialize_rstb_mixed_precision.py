#!/usr/bin/env python3
"""Materialize deployment-faithful low-bit RSTB fusion convolutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


FORMAT = "backdropkit-swin2sr-rstb-mixed-precision-materialization-v1"
SOURCE_FORMAT = "backdropkit-swin2sr-retained-precision-materialization-v1"
RSTB_KEYS = tuple(
    f"swin2sr.encoder.stages.{stage}.conv.weight"
    for stage in range(4)
)
ARM_SPECS = {
    "int4_per_output": {"bits": 4, "granularity": "per_output"},
    "int4_per_kernel": {"bits": 4, "granularity": "per_kernel"},
    "int8_per_output": {"bits": 8, "granularity": "per_output"},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _scale_shape(weight: torch.Tensor, granularity: str) -> tuple[int, ...]:
    if weight.ndim != 4 or tuple(weight.shape[2:]) != (3, 3):
        raise ValueError(f"expected a 3x3 convolution, got {tuple(weight.shape)}")
    if granularity == "per_output":
        return (weight.shape[0], 1, 1, 1)
    if granularity == "per_kernel":
        return (weight.shape[0], weight.shape[1], 1, 1)
    raise ValueError(f"unsupported granularity: {granularity}")


def quantize_weight(
    weight: torch.Tensor,
    bits: int,
    granularity: str,
) -> tuple[torch.Tensor, dict]:
    """Symmetric quantization with exact FP16 deployment scales."""

    if bits not in (4, 8):
        raise ValueError(f"unsupported bit width: {bits}")
    value = weight.detach().cpu().float().contiguous()
    scale_shape = _scale_shape(value, granularity)
    reduction = (1, 2, 3) if granularity == "per_output" else (2, 3)
    limit = (1 << (bits - 1)) - 1
    maximum = value.abs().amax(dim=reduction, keepdim=True)
    scale = (maximum / limit).to(torch.float16).to(torch.float32)
    if bool(((maximum > 0) & (scale == 0)).any()):
        raise RuntimeError("a nonzero quantization scale underflowed FP16")
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(value / safe_scale).clamp(-limit, limit)
    recovered = codes * scale
    if tuple(scale.shape) != scale_shape:
        raise RuntimeError(f"unexpected scale shape: {tuple(scale.shape)} != {scale_shape}")
    if float(codes.min()) < -limit or float(codes.max()) > limit:
        raise RuntimeError("quantized code exceeds the declared signed range")
    code_bytes = math.ceil(value.numel() * bits / 8)
    scale_bytes = scale.numel() * 2
    error = recovered - value
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


def load_source(receipt_path: Path, expected_precision: str) -> tuple[dict, dict]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != SOURCE_FORMAT:
        raise ValueError("source has the wrong materialization format")
    training = receipt.get("retained_precision_training", {})
    if training.get("island") != "rstb_fusion_fp16":
        raise ValueError("source is not the RSTB fusion island")
    if training.get("precision") != expected_precision:
        raise ValueError(f"source precision is not {expected_precision}")
    state_path = Path(receipt["dense_state"]["path"])
    if file_sha256(state_path) != receipt["dense_state"]["file_sha256"]:
        raise RuntimeError("source dense state changed after materialization")
    state = torch.load(state_path, map_location="cpu")
    return receipt, state


def materialize(
    protocol_path: Path,
    ternary_receipt_path: Path,
    fp16_receipt_path: Path,
    arm: str,
    out: Path,
) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != "backdropkit-swin2sr-rstb-mixed-precision-protocol-v1":
        raise ValueError("unsupported mixed-precision protocol")
    expected = protocol["sources"]
    if Path(expected["ternary_materialization"]).resolve() != ternary_receipt_path.resolve():
        raise ValueError("ternary source differs from the sealed protocol")
    if Path(expected["fp16_materialization"]).resolve() != fp16_receipt_path.resolve():
        raise ValueError("FP16 source differs from the sealed protocol")
    if arm not in ARM_SPECS or arm not in protocol["arms"]:
        raise ValueError(f"unsupported arm: {arm}")
    ternary_receipt, ternary = load_source(ternary_receipt_path, "ternary")
    fp16_receipt, fp16 = load_source(fp16_receipt_path, "fp16")
    if ternary.keys() != fp16.keys():
        raise RuntimeError("source state dictionaries differ")
    if ternary_receipt["parent_checkpoint_sha256"] != fp16_receipt["parent_checkpoint_sha256"]:
        raise RuntimeError("source arms have different parents")
    if ternary_receipt["retained_precision_training"]["seed"] != fp16_receipt["retained_precision_training"]["seed"]:
        raise RuntimeError("source arms have different seeds")

    undeclared = [
        name for name in ternary
        if name not in RSTB_KEYS and not torch.equal(ternary[name], fp16[name])
    ]
    if undeclared:
        raise RuntimeError(f"FP16 source changed undeclared tensors: {undeclared[:8]}")

    spec = ARM_SPECS[arm]
    result = {name: value.detach().cpu().clone() for name, value in ternary.items()}
    tensors = []
    for key in RSTB_KEYS:
        recovered, detail = quantize_weight(fp16[key], **spec)
        result[key] = recovered
        tensors.append({"name": key, **detail})
    changed = [name for name in result if not torch.equal(result[name], ternary[name])]
    if set(changed) != set(RSTB_KEYS):
        raise RuntimeError(f"mixed-precision state changed unexpected tensors: {changed}")

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
        "quantization": spec,
        "sources": {
            "ternary_materialization": str(ternary_receipt_path.resolve()),
            "ternary_materialization_sha256": file_sha256(ternary_receipt_path),
            "fp16_materialization": str(fp16_receipt_path.resolve()),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ternary-materialization", type=Path, required=True)
    parser.add_argument("--fp16-materialization", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_SPECS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = materialize(
        args.protocol,
        args.ternary_materialization,
        args.fp16_materialization,
        args.arm,
        args.out,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
