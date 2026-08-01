#!/usr/bin/env python3
"""Verify a Clark-packed artifact against its source training state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Swin2SRForImageSuperResolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clark_air.quant.packing import unpack_bits, unpack_trits_20
from evaluate_stratified import load_hard_student
from rotation_control import block_hadamard
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


DTYPES = {
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "int64": np.dtype("<i8"),
    "int32": np.dtype("<i4"),
    "uint32": np.dtype("<u4"),
    "uint8": np.dtype("u1"),
}
MAX_ABSOLUTE_TOLERANCE = 1.0 / 255.0
MAX_MEAN_TOLERANCE = 1e-3
MINIMUM_PSNR = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_array(blob: bytes, record: dict) -> np.ndarray:
    dtype = DTYPES[record["dtype"]]
    start = int(record["offset"])
    end = start + int(record["bytes"])
    return np.frombuffer(blob[start:end], dtype=dtype).copy()


def unpack_unsigned(blob: bytes, record: dict, values: int, bits: int) -> np.ndarray:
    start = int(record["offset"])
    payload = blob[start:start + int(record["bytes"])]
    output = np.zeros(values, dtype=np.int16)
    for index in range(values):
        bit_offset = index * bits
        raw = 0
        for bit in range(bits):
            source = bit_offset + bit
            raw |= ((payload[source // 8] >> (source % 8)) & 1) << bit
        output[index] = raw
    return output


def dequantize(
    codes: np.ndarray,
    scales: np.ndarray,
    shape: tuple[int, ...],
    group_size: int,
    offsets=None,
) -> np.ndarray:
    flat = codes.astype(np.float32).reshape(-1)
    pad = (-flat.size) % group_size
    if pad:
        flat = np.pad(flat, (0, pad))
    groups = (
        flat.reshape(-1, group_size)
        * scales.astype(np.float32).reshape(-1, 1)
    )
    if offsets is not None:
        groups = groups + offsets.astype(np.float32).reshape(-1, 1)
    values = groups.reshape(-1)
    if pad:
        values = values[:-pad]
    return values.reshape(shape)


def decode_state(artifact: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    manifest = json.loads((artifact / "model.json").read_text())
    blob = (artifact / "model.bin").read_bytes()
    state = {}
    default_transform = manifest.get("input_transform", {"kind": "none"})
    for name, tensor_record in manifest["tensors"].items():
        shape = tuple(tensor_record["shape"])
        kind = tensor_record["kind"]
        if kind == "raw":
            values = read_array(blob, tensor_record["data"]).reshape(shape)
        elif kind == "symmetric_int":
            bits = int(tensor_record["bit_width"])
            if tensor_record.get("granularity") != "per_output":
                raise ValueError("unsupported symmetric integer granularity")
            unsigned = unpack_unsigned(
                blob,
                tensor_record["codes"],
                int(np.prod(shape)),
                bits,
            )
            codes = unsigned - int(tensor_record["zero_point"])
            scales = read_array(blob, tensor_record["scales"]).astype(np.float32)
            if scales.size != shape[0]:
                raise ValueError("symmetric integer scale count changed")
            values = (
                codes.reshape(shape[0], -1).astype(np.float32)
                * scales[:, None]
            ).reshape(shape)
        else:
            packed = read_array(blob, tensor_record["codes"])
            if kind == "ternary":
                codes = unpack_trits_20(packed, shape, tensor_record["pad"])
            elif kind == "binary":
                codes = unpack_bits(packed, shape, tensor_record["pad"])
            else:
                raise ValueError(f"unsupported tensor kind: {kind}")
            scales = read_array(blob, tensor_record["scales"])
            offsets = (
                read_array(blob, tensor_record["offsets"])
                if "offsets" in tensor_record
                else None
            )
            values = dequantize(
                codes,
                scales,
                shape,
                tensor_record.get("group_size", manifest["group_size"]),
                offsets,
            )
            transform = tensor_record.get("input_transform", default_transform)
            if transform["kind"] == "block_hadamard":
                values = block_hadamard(
                    torch.from_numpy(values),
                    int(transform["block_size"]),
                ).numpy()
        state[name] = torch.from_numpy(values)
    return manifest, state


def psnr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    mse = torch.mean((actual.float() - expected.float()).square()).item()
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


def main() -> None:
    args = parse_args()
    manifest, state = decode_state(args.artifact)
    packed_model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    packed_model.load_state_dict(state, strict=True)
    packed_model = packed_model.to(args.device).eval()
    reference_model = load_hard_student(
        args.reference_state,
        args.device,
        int(manifest["group_size"]),
        "optimal_mse",
        "none",
        4,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.rand((1, 3, 64, 64), generator=generator).to(args.device)
    with torch.inference_mode():
        reference = reference_model(inputs).reconstruction
        actual = packed_model(inputs).reconstruction
    delta = (actual.float() - reference.float()).abs()
    output_psnr = psnr(actual, reference)
    finite = bool(torch.isfinite(actual).all().item())
    passed = (
        finite
        and float(delta.max().item()) <= MAX_ABSOLUTE_TOLERANCE
        and float(delta.mean().item()) <= MAX_MEAN_TOLERANCE
        and output_psnr >= MINIMUM_PSNR
    )
    report = {
        "schema_version": 1,
        "artifact": str(args.artifact),
        "reference_state": str(args.reference_state),
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "mode": manifest["mode"],
        "group_size": manifest["group_size"],
        "input_transform": manifest["input_transform"],
        "artifact_binary_sha256": sha256(args.artifact / "model.bin"),
        "artifact_manifest_sha256": sha256(args.artifact / "model.json"),
        "reference_state_sha256": sha256(args.reference_state),
        "input_shape": list(inputs.shape),
        "output_shape": list(actual.shape),
        "max_absolute_error": float(delta.max().item()),
        "mean_absolute_error": float(delta.mean().item()),
        "psnr_vs_reference": output_psnr,
        "finite": finite,
        "tolerances": {
            "max_absolute_error": MAX_ABSOLUTE_TOLERANCE,
            "mean_absolute_error": MAX_MEAN_TOLERANCE,
            "minimum_psnr": MINIMUM_PSNR,
        },
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
