#!/usr/bin/env python3
"""Export a trained Swin2SR state as an aligned Clark Air browser artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Swin2SRForImageSuperResolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clark_air.quant.binary import quantize_binary
from clark_air.quant.optimal_ternary import quantize_ternary_mse
from clark_air.quant.packing import pack_trits_20
from artifact_ledger import build_artifact_ledger
from full_surface_provenance import file_sha256
from materialize_surface_states import load_materialized_state
from quantizer_controls import quantize_ternary_absmean
from rewrite_onnx_ternary import recover_ternary_groups
from scale_codebook_state import (
    MATERIALIZATION_FORMAT as SCALE_CODEBOOK_MATERIALIZATION_FORMAT,
    load_scale_codebook_materialized_state,
)
from surface_schedule import SCOPES, surface_targets


MODEL_ID = "caidas/swin2SR-lightweight-x2-64"
MODEL_REVISION = "07e8e61492ce6daf4f34c651622960faf110c17f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("materialization", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def selected_weights(model: torch.nn.Module, scope: str) -> set[str]:
    return {
        f"{target.name}.weight"
        for target in surface_targets(model, scope)
    }


def selected_categories(model: torch.nn.Module, scope: str) -> dict[str, str]:
    return {
        f"{target.name}.weight": target.category
        for target in surface_targets(model, scope)
    }


def aligned_write(output, payload: bytes) -> tuple[int, int]:
    padding = (-output.tell()) % 4
    if padding:
        output.write(b"\0" * padding)
    offset = output.tell()
    output.write(payload)
    return offset, len(payload)


def pack_unsigned(values: np.ndarray, bit_width: int) -> bytes:
    """Pack unsigned codes least-significant-bit first into a byte stream."""

    flat = np.asarray(values, dtype=np.uint16).reshape(-1)
    if bit_width < 1 or bit_width > 8:
        raise ValueError("packed unsigned width must be between one and eight bits")
    if flat.size and int(flat.max()) >= 1 << bit_width:
        raise ValueError("packed unsigned value exceeds its bit width")
    output = bytearray((flat.size * bit_width + 7) // 8)
    for index, value in enumerate(flat):
        raw = int(value)
        bit_offset = index * bit_width
        for bit in range(bit_width):
            output[(bit_offset + bit) // 8] |= (
                ((raw >> bit) & 1) << ((bit_offset + bit) % 8)
            )
    return bytes(output)


def packed_scale_codebook(scales: np.ndarray, bit_width: int) -> tuple[np.ndarray, bytes]:
    stored = np.asarray(scales, dtype=np.float16).reshape(-1)
    centers, indices = np.unique(stored, return_inverse=True)
    if centers.size > 1 << bit_width:
        raise ValueError(
            f"hard scale state has {centers.size} levels, exceeding q{bit_width}"
        )
    return centers, pack_unsigned(indices, bit_width)


def quantize_tensor(
    tensor: torch.Tensor,
    *,
    mode: str,
    method: str,
    group_size: int,
):
    if mode == "binary":
        return quantize_binary(tensor.float(), group_size)
    if method == "optimal_mse":
        return quantize_ternary_mse(tensor.float(), group_size)
    if method == "absmean":
        return quantize_ternary_absmean(tensor.float(), group_size)
    raise ValueError(f"unsupported export quantizer: {method}")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    parameter_values = sum(parameter.numel() for parameter in model.parameters())
    materialization_header = json.loads(args.materialization.read_text())
    materialization_loader = (
        load_scale_codebook_materialized_state
        if materialization_header.get("format") == SCALE_CODEBOOK_MATERIALIZATION_FORMAT
        else load_materialized_state
    )
    materialization, state = materialization_loader(
        args.materialization, "runtime_state"
    )
    quantization = materialization["quantization"]
    if quantization.get("mode") != "ternary":
        raise ValueError("browser materialization must be ternary")
    scope = quantization["scope"]
    if scope not in SCOPES:
        raise ValueError(f"unsupported materialized scope: {scope}")
    model.load_state_dict(state, strict=True)
    surface_contract = materialization["surfaces"]
    scale_codebook_enabled = "scale_codebook" in materialization
    affine_record = materialization.get("affine_ternary")
    if scale_codebook_enabled and affine_record:
        raise ValueError("scale-codebook and affine-ternary payloads cannot be combined")
    affine_tensors = None
    if affine_record:
        affine_path = Path(affine_record["path"])
        if file_sha256(affine_path) != affine_record["file_sha256"]:
            raise ValueError("affine-ternary payload identity mismatch")
        affine_payload = torch.load(
            affine_path, map_location="cpu", weights_only=False
        )
        if affine_payload.get("format") != affine_record["format"]:
            raise ValueError("unsupported affine-ternary payload")
        affine_tensors = affine_payload["tensors"]
    if scale_codebook_enabled and not materialization.get(
        "all_scale_codebooks_materialized_hard"
    ):
        raise ValueError("scale-codebook artifact is not fully hard")
    selected = set(surface_contract)
    expected = selected_weights(model, scope)
    retained_precision = materialization.get("retained_precision_island")
    retained_weights = set()
    if retained_precision is not None:
        if retained_precision.get("precision") != "fp16":
            raise ValueError("unsupported retained-precision browser island")
        if retained_precision.get("storage_dtype") != "float16":
            raise ValueError("retained-precision browser island must store FP16")
        retained_weights = set(retained_precision.get("weights", []))
        if not retained_weights or not retained_weights <= expected:
            raise ValueError("retained-precision island is outside the target scope")
        retained_values = sum(state[name].numel() for name in retained_weights)
        if retained_values != int(retained_precision["weight_values"]):
            raise ValueError("retained-precision island value count changed")
        if any(
            not torch.equal(
                state[name].float(),
                state[name].to(torch.float16).to(torch.float32),
            )
            for name in retained_weights
        ):
            raise ValueError("retained-precision island is not an exact FP16 roundtrip")
        expected -= retained_weights
    if selected != expected:
        raise ValueError("materialization surface contract does not match model")
    categories = {
        name: contract["category"]
        for name, contract in surface_contract.items()
    }
    if affine_tensors is not None and set(affine_tensors) != selected:
        raise ValueError("affine-ternary payload does not match selected surfaces")
    projected = {}
    for name, contract in surface_contract.items():
        if affine_tensors is None:
            codes, scales, recovery_error = recover_ternary_groups(
                state[name].detach().cpu().numpy(),
                int(contract["group_size"]),
            )
            projected[name] = {
                "codes": codes,
                "scales": scales,
                "offsets": None,
                "recovery_error": recovery_error,
            }
            continue
        record = affine_tensors[name]
        codes = record["codes"].detach().cpu().numpy().astype(np.int8)
        scales = record["scales"].detach().cpu().numpy().astype(np.float32)
        offsets = record["offsets"].detach().cpu().numpy().astype(np.float32)
        shape = tuple(record["shape"])
        group_size = int(record["group_size"])
        if shape != tuple(state[name].shape) or group_size != int(contract["group_size"]):
            raise ValueError(f"affine-ternary shape contract changed: {name}")
        values = codes.astype(np.float32).reshape(-1)
        pad = int(record["pad"])
        if pad:
            values = np.pad(values, (0, pad))
        expected_groups = values.size // group_size
        if scales.size != expected_groups or offsets.size != expected_groups:
            raise ValueError(f"affine-ternary group count changed: {name}")
        recovered = (
            values.reshape(-1, group_size) * scales[:, None]
            + offsets[:, None]
        ).reshape(-1)
        if pad:
            recovered = recovered[:-pad]
        recovered = recovered.reshape(shape)
        recovery_error = float(np.max(np.abs(
            recovered - state[name].detach().cpu().numpy().astype(np.float32)
        )))
        if recovery_error > 1e-6:
            raise ValueError(f"affine-ternary payload changed runtime state: {name}")
        projected[name] = {
            "codes": codes,
            "scales": scales,
            "offsets": offsets,
            "recovery_error": recovery_error,
        }
    manifest = {
        "format": (
            "clark-air-browser-v3" if scale_codebook_enabled
            else "clark-air-browser-v2"
        ),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "mode": "ternary",
        "method": quantization["method"],
        "scope": scope,
        "group_size": int(quantization["group_size"]),
        "materialization_manifest": str(args.materialization),
        "materialization_manifest_sha256": file_sha256(args.materialization),
        "runtime_state_tensor_sha256": materialization[
            "runtime_state"
        ]["tensor_sha256"],
        # Legacy core artifacts used a global transform. New artifacts make
        # shaping tensor-local so packed convolutions and position-bias MLPs
        # are never transformed accidentally.
        "input_transform": {"kind": "none"},
        "tensors": {},
    }
    if retained_precision is not None:
        manifest["retained_precision_island"] = retained_precision
    binary_path = args.out / "model.bin"
    quantized_values = 0
    total_values = 0
    code_bytes_total = 0
    scale_bytes_total = 0
    offset_bytes_total = 0
    raw_bytes_total = 0
    with binary_path.open("wb") as output:
        for name, value in state.items():
            tensor = value.detach().cpu()
            total_values += tensor.numel()
            if name in selected:
                projection = projected[name]
                codes = projection["codes"]
                scales = projection["scales"]
                offsets = projection["offsets"]
                recovery_error = projection["recovery_error"]
                packed, _, pad = pack_trits_20(codes)
                code_dtype = "uint32"
                kind = "ternary"
                code_offset, code_bytes = aligned_write(output, packed.tobytes())
                scale_record = {}
                if scale_codebook_enabled:
                    target_bits = int(surface_contract[name]["target_bits"])
                    centers, scale_indices = packed_scale_codebook(
                        scales, target_bits
                    )
                    center_offset, center_bytes = aligned_write(
                        output, centers.tobytes()
                    )
                    index_offset, index_bytes = aligned_write(
                        output, scale_indices
                    )
                    scale_bytes = center_bytes + index_bytes
                    scale_record = {
                        "scale_codec": "codebook_fp16",
                        "scale_count": int(scales.size),
                        "scale_index_bits": target_bits,
                        "scale_codebook": {
                            "dtype": "float16",
                            "offset": center_offset,
                            "bytes": center_bytes,
                        },
                        "scale_indices": {
                            "dtype": "packed_uint",
                            "offset": index_offset,
                            "bytes": index_bytes,
                        },
                    }
                else:
                    scales_f16 = scales.astype(np.float16)
                    scale_offset, scale_bytes = aligned_write(
                        output,
                        scales_f16.tobytes(),
                    )
                    scale_record = {
                        "scales": {
                            "dtype": "float16",
                            "offset": scale_offset,
                            "bytes": scale_bytes,
                        }
                    }
                offset_record = {}
                offset_bytes = 0
                if offsets is not None:
                    offsets_f16 = offsets.astype(np.float16)
                    offset_offset, offset_bytes = aligned_write(
                        output,
                        offsets_f16.tobytes(),
                    )
                    offset_record = {
                        "offsets": {
                            "dtype": "float16",
                            "offset": offset_offset,
                            "bytes": offset_bytes,
                        }
                    }
                quantized_values += tensor.numel()
                code_bytes_total += code_bytes
                scale_bytes_total += scale_bytes
                offset_bytes_total += offset_bytes
                manifest["tensors"][name] = {
                    "kind": kind,
                    "shape": list(tensor.shape),
                    "pad": pad,
                    "group_size": int(surface_contract[name]["group_size"]),
                    "method": surface_contract[name].get(
                        "method",
                        (
                            quantization["method"]
                            if tensor.ndim == 2
                            else materialization["surface_training"][
                                "conv_quantizer_method"
                            ]
                        ),
                    ),
                    "hard_ternary_recovery_max_error": recovery_error,
                    "codes": {"dtype": code_dtype, "offset": code_offset, "bytes": code_bytes},
                    **scale_record,
                    **offset_record,
                }
                if (
                    surface_contract[name]["rotation"] == "hadamard"
                    and categories[name] == "core_linear"
                ):
                    manifest["tensors"][name]["input_transform"] = {
                        "kind": "block_hadamard",
                        "block_size": int(
                            surface_contract[name]["rotation_block"]
                        ),
                    }
            else:
                if tensor.is_floating_point():
                    raw = tensor.half().numpy()
                    dtype = "float16"
                else:
                    raw = tensor.numpy()
                    dtype = str(raw.dtype)
                offset, byte_count = aligned_write(output, raw.tobytes())
                raw_bytes_total += byte_count
                manifest["tensors"][name] = {
                    "kind": "raw",
                    "shape": list(tensor.shape),
                    "data": {"dtype": dtype, "offset": offset, "bytes": byte_count},
                }
    binary_bytes = binary_path.stat().st_size
    manifest["storage_ledger"] = build_artifact_ledger(
        mode="ternary",
        binary_bytes=binary_bytes,
        parameter_values=parameter_values,
        state_values=total_values,
        quantized_values=quantized_values,
        code_bytes=code_bytes_total,
        scale_bytes=scale_bytes_total,
        offset_bytes=offset_bytes_total,
        raw_bytes=raw_bytes_total,
    )
    manifest_path = args.out / "model.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(json.dumps({
        "binary_bytes": binary_bytes,
        "manifest_bytes": manifest_path.stat().st_size,
        "total_bytes": binary_bytes + manifest_path.stat().st_size,
        "storage_ledger": manifest["storage_ledger"],
        "quantized_tensors": len(selected),
    }, indent=2))


if __name__ == "__main__":
    main()
