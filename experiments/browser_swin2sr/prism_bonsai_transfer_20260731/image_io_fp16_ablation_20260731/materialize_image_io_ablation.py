#!/usr/bin/env python3
"""Materialize one single-convolution ablation of the image-I/O FP16 island."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


FORMAT = "backdropkit-swin2sr-image-io-fp16-ablation-v1"
SOURCE_FORMAT = "backdropkit-swin2sr-retained-precision-materialization-v1"
KEYS = {
    "input": "swin2sr.first_convolution.weight",
    "reconstruction": "upsample.conv.weight",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_receipt(path: Path, precision: str) -> tuple[dict, dict[str, torch.Tensor]]:
    receipt = json.loads(path.read_text())
    if receipt.get("format") != SOURCE_FORMAT:
        raise ValueError("unsupported retained-precision materialization")
    config = receipt["retained_precision_training"]
    if config.get("island") != "image_io_fp16" or config.get("precision") != precision:
        raise ValueError(f"source is not the image-I/O {precision} arm")
    state_path = Path(receipt["dense_state"]["path"])
    if file_sha256(state_path) != receipt["dense_state"]["file_sha256"]:
        raise ValueError(f"{precision} state file hash changed")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if state_sha256(state) != receipt["dense_state"]["tensor_sha256"]:
        raise ValueError(f"{precision} tensor hash changed")
    return receipt, state


def matched_config(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in {"precision", "initialization"}
    }


def ablate_states(
    ternary: dict[str, torch.Tensor],
    fp16: dict[str, torch.Tensor],
    choice: str,
) -> tuple[dict[str, torch.Tensor], dict]:
    if choice not in KEYS:
        raise ValueError(f"unknown image-I/O ablation: {choice}")
    if ternary.keys() != fp16.keys():
        raise ValueError("paired state keys differ")
    changed = {
        name for name in ternary
        if not torch.equal(ternary[name], fp16[name])
    }
    if changed != set(KEYS.values()):
        raise ValueError(f"paired arms differ outside the exact image-I/O pair: {sorted(changed)}")
    selected = KEYS[choice]
    output = {
        name: tensor.detach().cpu().clone()
        for name, tensor in ternary.items()
    }
    output[selected] = fp16[selected].detach().cpu().clone()
    output_changed = {
        name for name in ternary
        if not torch.equal(ternary[name], output[name])
    }
    if output_changed != {selected}:
        raise RuntimeError("ablation changed an undeclared tensor")
    return output, {
        "choice": choice,
        "retained_fp16_tensor": selected,
        "retained_fp16_values": output[selected].numel(),
        "changed_from_ternary": sorted(output_changed),
    }


def projected_delta_bytes(values: int, group_size: int = 2) -> dict:
    ternary_codes = math.ceil(values / 20) * 4
    ternary_scales = math.ceil(values / group_size) * 2
    fp16 = values * 2
    return {
        "ternary_code_bytes": ternary_codes,
        "ternary_scale_bytes": ternary_scales,
        "retained_fp16_bytes": fp16,
        "projected_data_byte_delta": fp16 - ternary_codes - ternary_scales,
    }


def materialize(
    ternary_path: Path,
    fp16_path: Path,
    choice: str,
    out: Path,
) -> dict:
    ternary_receipt, ternary = load_receipt(ternary_path, "ternary")
    fp16_receipt, fp16 = load_receipt(fp16_path, "fp16")
    if ternary_receipt["parent_checkpoint_sha256"] != fp16_receipt["parent_checkpoint_sha256"]:
        raise ValueError("paired arms have different parents")
    if matched_config(ternary_receipt["retained_precision_training"]) != matched_config(
        fp16_receipt["retained_precision_training"]
    ):
        raise ValueError("paired arm configuration differs beyond precision")
    state, intervention = ablate_states(ternary, fp16, choice)
    storage = projected_delta_bytes(intervention["retained_fp16_values"])
    expected_delta = 1_296 if choice == "input" else 5_184
    if storage["projected_data_byte_delta"] != expected_delta:
        raise RuntimeError("image-I/O storage geometry changed")
    out.mkdir(parents=True, exist_ok=False)
    state_path = out / "dense-hard.pt"
    torch.save(state, state_path)
    receipt = {
        "format": FORMAT,
        "interpretation": (
            "The same-seed hard-ternary image-I/O arm with exactly one declared "
            "convolution replaced by its paired trained FP16 value."
        ),
        "seed": ternary_receipt["retained_precision_training"]["seed"],
        "parent_checkpoint_sha256": ternary_receipt["parent_checkpoint_sha256"],
        "sources": {
            "ternary": {
                "materialization": str(ternary_path.resolve()),
                "sha256": file_sha256(ternary_path),
                "state_sha256": ternary_receipt["dense_state"]["tensor_sha256"],
            },
            "fp16": {
                "materialization": str(fp16_path.resolve()),
                "sha256": file_sha256(fp16_path),
                "state_sha256": fp16_receipt["dense_state"]["tensor_sha256"],
            },
        },
        "intervention": intervention,
        "storage": storage,
        "dense_state": {
            "path": str(state_path.resolve()),
            "file_sha256": file_sha256(state_path),
            "tensor_sha256": state_sha256(state),
            "tensors": len(state),
            "values": sum(tensor.numel() for tensor in state.values()),
        },
    }
    manifest = out / "materialization.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ternary-materialization", type=Path, required=True)
    parser.add_argument("--fp16-materialization", type=Path, required=True)
    parser.add_argument("--choice", choices=sorted(KEYS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(
        args.ternary_materialization,
        args.fp16_materialization,
        args.choice,
        args.out,
    )
    print(json.dumps({
        "choice": receipt["intervention"]["choice"],
        "storage": receipt["storage"],
        "dense_state": receipt["dense_state"],
    }, indent=2))


if __name__ == "__main__":
    main()
