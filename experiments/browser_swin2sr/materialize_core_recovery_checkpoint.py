#!/usr/bin/env python3
"""Materialize a legacy core-only recovery checkpoint for the packed runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clark_air.quant.optimal_ternary import _pad_flat, quantize_ternary_mse
from artifact_state import hard_state_dict
from full_surface_provenance import file_sha256
from materialize_surface_states import FORMAT, state_receipt, state_sha256
from quantizer_controls import quantize_ternary_absmean
from train_swin2sr_recovery import load_recovery_model


RECOVERY_FORMAT = "backdropkit-swin2sr-recovery-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--evaluated-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-device", default="cpu")
    return parser.parse_args()


def load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and torch.is_tensor(value)
        for name, value in state.items()
    ):
        raise ValueError("evaluated state is not a tensor state dictionary")
    return state


def materialize(
    checkpoint: Path,
    evaluated_state_path: Path,
    out: Path,
    source_device: str,
) -> dict:
    payload = torch.load(checkpoint, map_location=source_device, weights_only=False)
    recovery = payload.get("recovery", {})
    if recovery.get("format") != RECOVERY_FORMAT:
        raise ValueError("checkpoint is not a core-recovery training state")
    quantization = payload.get("quantization", {})
    if quantization.get("mode") != "ternary":
        raise ValueError("core-recovery browser export requires ternary weights")
    if source_device not in {"cpu", "mps"}:
        raise ValueError("core-recovery materialization supports CPU or MPS")
    names = payload.get("quantized_modules", [])
    if len(names) != 144 or len(set(names)) != 144:
        raise RuntimeError("checkpoint does not cover the exact 144 core surfaces")

    evaluated_state = load_tensor_state(evaluated_state_path)
    dense_state = {
        name: value.detach().cpu().clone()
        for name, value in evaluated_state.items()
    }
    dense_tensor_sha256 = state_sha256(dense_state)
    evaluated_tensor_sha256 = state_sha256(evaluated_state)
    runtime_state = {
        name: value.detach().cpu().clone()
        for name, value in evaluated_state.items()
    }
    rotation = quantization["rotation"]
    rotation_block = int(quantization["rotation_block"])
    if rotation != "hadamard":
        raise ValueError("core-recovery runtime reconstruction requires Hadamard")
    load_args = SimpleNamespace(
        device=source_device,
        group_size=int(quantization["group_size"]),
        scalar_decomposition=bool(recovery.get("scalar_decomposition")),
    )
    student, quantizers, _group_size = load_recovery_model(load_args, payload)
    reconstructed_names = [name for name, _quantizer, _module in quantizers]
    if reconstructed_names != names:
        raise RuntimeError("reconstructed core surface ordering changed")
    reconstructed_dense = hard_state_dict(student, quantizers)
    if state_sha256(reconstructed_dense) != evaluated_tensor_sha256:
        raise RuntimeError(
            "source device does not reproduce the exact evaluated hard state"
        )
    affine_tensors = {}
    maximum_affine_error = 0.0
    for name, quantizer, module in quantizers:
        weight_name = f"{name}.weight"
        if weight_name not in evaluated_state:
            raise RuntimeError(f"evaluated core weight is missing: {weight_name}")
        master = module.parametrizations.weight.original.detach()
        base = (
            quantize_ternary_mse(master, int(quantizer.group_size))
            if quantizer.method == "optimal_mse"
            else quantize_ternary_absmean(master, int(quantizer.group_size))
        )
        scale_multiplier = quantizer.log_scale_multiplier.detach().exp()
        offset_multiplier = quantizer.offset_multiplier.detach()
        scales = base.scales * scale_multiplier
        offsets = base.scales * offset_multiplier
        padded_codes, _pad = _pad_flat(
            base.codes.to(dtype=master.dtype), int(quantizer.group_size)
        )
        projected = (
            padded_codes.reshape(-1, int(quantizer.group_size))
            * scales[:, None]
            + offsets[:, None]
        ).reshape(-1)
        if base.pad:
            projected = projected[:-base.pad]
        projected = projected.reshape_as(master)
        observed = module.weight.detach()
        error = float((projected - observed).abs().max().item())
        maximum_affine_error = max(maximum_affine_error, error)
        if error > 1e-7:
            raise RuntimeError(f"affine ternary reconstruction changed: {weight_name}")
        runtime_state[weight_name] = projected.cpu()
        affine_tensors[weight_name] = {
            "codes": base.codes.detach().cpu(),
            "scales": scales.detach().cpu(),
            "offsets": offsets.detach().cpu(),
            "shape": list(master.shape),
            "group_size": int(quantizer.group_size),
            "pad": int(base.pad),
        }
    if dense_tensor_sha256 != evaluated_tensor_sha256:
        raise RuntimeError("materialized dense state differs from the evaluated state")

    out.mkdir(parents=True, exist_ok=True)
    dense_path = out / "dense-hard.pt"
    runtime_path = out / "runtime-hard.pt"
    affine_path = out / "affine-ternary.pt"
    torch.save(dense_state, dense_path)
    torch.save(runtime_state, runtime_path)
    torch.save({
        "format": "backdropkit-swin2sr-affine-ternary-payload-v1",
        "tensors": affine_tensors,
    }, affine_path)
    surfaces = {
        f"{name}.weight": {
            "category": "core_linear",
            "method": quantization["method"],
            "group_size": int(quantization["group_size"]),
            "rotation": rotation,
            "rotation_block": rotation_block,
            "affine_offsets": True,
        }
        for name in names
    }
    manifest = {
        "format": FORMAT,
        "interpretation": (
            "All 144 transformer core matrices are hard ternary. Convolution, "
            "position-bias, normalization, and remaining tensors stay raw FP16 "
            "in the packed artifact. dense_state is byte-independent but tensor-"
            "identical to the state used for the 3,040-sample evaluation."
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_device": source_device,
        "surface_step": int(payload["step"]),
        "all_target_surfaces_materialized_hard": True,
        "quantization": {
            **quantization,
            "scope": "core",
        },
        "surface_training": {
            **recovery,
            "conv_quantizer_method": "not_applicable",
        },
        "surfaces": surfaces,
        "affine_ternary": {
            "format": "backdropkit-swin2sr-affine-ternary-payload-v1",
            "path": str(affine_path.resolve()),
            "file_sha256": file_sha256(affine_path),
            "tensors": len(affine_tensors),
            "maximum_reconstruction_error": maximum_affine_error,
        },
        "evaluated_state": {
            "path": str(evaluated_state_path.resolve()),
            "file_sha256": file_sha256(evaluated_state_path),
            "tensor_sha256": evaluated_tensor_sha256,
            "tensors": len(evaluated_state),
            "values": sum(value.numel() for value in evaluated_state.values()),
        },
        "dense_state": state_receipt(dense_path, dense_state),
        "runtime_state": state_receipt(runtime_path, runtime_state),
    }
    manifest_path = out / "materialization.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**manifest, "manifest": str(manifest_path.resolve())}


def main() -> None:
    args = parse_args()
    result = materialize(
        args.checkpoint,
        args.evaluated_state,
        args.out,
        args.source_device,
    )
    print(json.dumps({
        "manifest": result["manifest"],
        "surfaces": len(result["surfaces"]),
        "dense_tensor_sha256": result["dense_state"]["tensor_sha256"],
        "evaluated_tensor_sha256": result["evaluated_state"]["tensor_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
