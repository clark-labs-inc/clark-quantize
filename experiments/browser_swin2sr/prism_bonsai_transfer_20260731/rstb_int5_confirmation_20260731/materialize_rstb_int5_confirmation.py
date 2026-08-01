#!/usr/bin/env python3
"""Apply the selected 5-bit per-kernel RSTB geometry to confirmation seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


BOUNDARY_ROOT = Path(__file__).parents[1] / "rstb_precision_boundary_20260731"
sys.path.insert(0, str(BOUNDARY_ROOT))
from materialize_rstb_precision_boundary import (  # noqa: E402
    RSTB_KEYS,
    file_sha256,
    load_source,
    quantize_weight_nbit,
    tensor_sha256,
)


FORMAT = "backdropkit-swin2sr-rstb-int5-confirmation-materialization-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-rstb-int5-confirmation-protocol-v1"
BITS = 5
GRANULARITY = "per_kernel"


def verify_source_pair(
    ternary_receipt: dict,
    fp16_receipt: dict,
    expected_seed: int,
) -> None:
    ternary_training = ternary_receipt["retained_precision_training"]
    fp16_training = fp16_receipt["retained_precision_training"]
    if int(ternary_training["seed"]) != expected_seed:
        raise RuntimeError("ternary source seed differs from the sealed seed")
    if int(fp16_training["seed"]) != expected_seed:
        raise RuntimeError("FP16 source seed differs from the sealed seed")
    if ternary_receipt["parent_checkpoint_sha256"] != fp16_receipt["parent_checkpoint_sha256"]:
        raise RuntimeError("source arms have different parents")


def materialize(
    protocol_path: Path,
    seed: int,
    ternary_receipt_path: Path,
    fp16_receipt_path: Path,
    out: Path,
) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported RSTB int5 confirmation protocol")
    if protocol.get("selected_geometry") != {
        "bits": BITS,
        "granularity": GRANULARITY,
    }:
        raise ValueError("confirmation geometry differs from the selected boundary")
    seed_contract = protocol["seeds"].get(str(seed))
    if seed_contract is None:
        raise ValueError(f"seed {seed} is not authorized by the sealed protocol")
    if Path(seed_contract["ternary_materialization"]).resolve() != ternary_receipt_path.resolve():
        raise ValueError("ternary source differs from the sealed protocol")
    if Path(seed_contract["fp16_materialization"]).resolve() != fp16_receipt_path.resolve():
        raise ValueError("FP16 source differs from the sealed protocol")

    ternary_receipt, ternary = load_source(ternary_receipt_path, "ternary")
    fp16_receipt, fp16 = load_source(fp16_receipt_path, "fp16")
    verify_source_pair(ternary_receipt, fp16_receipt, seed)
    if ternary.keys() != fp16.keys():
        raise RuntimeError("source state dictionaries differ")
    undeclared = [
        name for name in ternary
        if name not in RSTB_KEYS and not torch.equal(ternary[name], fp16[name])
    ]
    if undeclared:
        raise RuntimeError(f"FP16 source changed undeclared tensors: {undeclared[:8]}")

    result = {name: value.detach().cpu().clone() for name, value in ternary.items()}
    tensors = []
    for key in RSTB_KEYS:
        recovered, detail = quantize_weight_nbit(
            fp16[key], bits=BITS, granularity=GRANULARITY
        )
        result[key] = recovered
        tensors.append({"name": key, **detail})
    changed = [name for name in result if not torch.equal(result[name], ternary[name])]
    if set(changed) != set(RSTB_KEYS):
        raise RuntimeError(f"confirmation state changed unexpected tensors: {changed}")

    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "dense-hard.pt"
    torch.save(result, state_path)
    data_bytes = sum(item["data_bytes"] for item in tensors)
    ternary_bytes = int(protocol["storage"]["ternary_rstb_data_bytes"])
    increment = data_bytes - ternary_bytes
    if increment != int(protocol["storage"]["expected_incremental_data_bytes"]):
        raise RuntimeError("confirmation storage differs from the sealed geometry")
    receipt = {
        "format": FORMAT,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": file_sha256(protocol_path),
        "seed": seed,
        "quantization": {"bits": BITS, "granularity": GRANULARITY},
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
            "incremental_data_bytes": increment,
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ternary-materialization", type=Path, required=True)
    parser.add_argument("--fp16-materialization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        args.protocol,
        args.seed,
        args.ternary_materialization,
        args.fp16_materialization,
        args.out,
    ), indent=2))


if __name__ == "__main__":
    main()
