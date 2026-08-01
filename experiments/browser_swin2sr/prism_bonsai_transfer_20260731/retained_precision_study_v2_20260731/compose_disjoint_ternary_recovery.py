#!/usr/bin/env python3
"""Compose two preregistered, disjoint hard-ternary recovery islands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


FORMAT = "backdropkit-swin2sr-disjoint-ternary-composition-v1"
SOURCE_FORMAT = "backdropkit-swin2sr-retained-precision-materialization-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-disjoint-ternary-composition-protocol-v1"
IMAGE_KEYS = (
    "swin2sr.first_convolution.weight",
    "upsample.conv.weight",
)
RSTB_KEYS = tuple(
    f"swin2sr.encoder.stages.{stage}.conv.weight"
    for stage in range(4)
)


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


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise ValueError(f"not a tensor state dictionary: {path}")
    return state


def load_source(receipt_path: Path, island: str) -> tuple[dict, dict[str, torch.Tensor]]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != SOURCE_FORMAT:
        raise ValueError(f"unsupported source receipt: {receipt_path}")
    config = receipt["retained_precision_training"]
    if config.get("island") != island:
        raise ValueError(f"source island differs from {island}")
    if config.get("precision") != "ternary":
        raise ValueError(f"source island is not ternary: {island}")
    state_path = Path(receipt["dense_state"]["path"])
    if file_sha256(state_path) != receipt["dense_state"]["file_sha256"]:
        raise ValueError(f"source state file hash changed: {island}")
    state = load_state(state_path)
    if state_sha256(state) != receipt["dense_state"]["tensor_sha256"]:
        raise ValueError(f"source tensor hash changed: {island}")
    return receipt, state


def verify_protocol_inputs(
    protocol_path: Path,
    parent_path: Path,
    image_receipt_path: Path,
    rstb_receipt_path: Path,
    out: Path,
) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported composition protocol")
    expected_parent = Path(protocol["parent"]["state"])
    if parent_path.resolve() != expected_parent.resolve():
        raise ValueError("parent state path differs from sealed protocol")
    if file_sha256(parent_path) != protocol["parent"]["state_file_sha256"]:
        raise ValueError("parent state hash differs from sealed protocol")
    expected_image = Path(protocol["sources"]["image_io_fp16"]["materialization"])
    if image_receipt_path.resolve() != expected_image.resolve():
        raise ValueError("image materialization path differs from sealed protocol")
    if (
        file_sha256(image_receipt_path)
        != protocol["sources"]["image_io_fp16"]["materialization_sha256_at_seal"]
    ):
        raise ValueError("image materialization hash differs from sealed protocol")
    expected_rstb = Path(protocol["sources"]["rstb_fusion_fp16"]["materialization"])
    if rstb_receipt_path.resolve() != expected_rstb.resolve():
        raise ValueError("RSTB materialization path differs from sealed protocol")
    expected_out = Path(protocol["composition_command"][-1])
    if out.resolve() != expected_out.resolve():
        raise ValueError("composition output path differs from sealed protocol")
    source_study = Path(protocol["source_study"])
    if file_sha256(source_study) != protocol["source_study_sha256"]:
        raise ValueError("source study hash differs from sealed protocol")
    return protocol


def changed_keys(
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> set[str]:
    if parent.keys() != candidate.keys():
        raise ValueError("source and parent state keys differ")
    return {
        name
        for name in parent
        if not torch.equal(parent[name], candidate[name])
    }


def compose_states(
    parent: dict[str, torch.Tensor],
    image: dict[str, torch.Tensor],
    rstb: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict]:
    image_changes = changed_keys(parent, image)
    rstb_changes = changed_keys(parent, rstb)
    if not image_changes <= set(IMAGE_KEYS):
        raise ValueError(
            f"image source changed undeclared tensors: {sorted(image_changes - set(IMAGE_KEYS))}"
        )
    if not rstb_changes <= set(RSTB_KEYS):
        raise ValueError(
            f"RSTB source changed undeclared tensors: {sorted(rstb_changes - set(RSTB_KEYS))}"
        )
    missing = (set(IMAGE_KEYS) | set(RSTB_KEYS)) - parent.keys()
    if missing:
        raise ValueError(f"parent is missing declared tensors: {sorted(missing)}")
    if not image_changes:
        raise ValueError("image source made no exact tensor changes")
    if not rstb_changes:
        raise ValueError("RSTB source made no exact tensor changes")

    composed = {
        name: tensor.detach().cpu().clone()
        for name, tensor in parent.items()
    }
    for name in IMAGE_KEYS:
        composed[name] = image[name].detach().cpu().clone()
    for name in RSTB_KEYS:
        composed[name] = rstb[name].detach().cpu().clone()
    receipt = {
        "image_changed_keys": sorted(image_changes),
        "rstb_changed_keys": sorted(rstb_changes),
        "composed_keys": sorted(set(IMAGE_KEYS) | set(RSTB_KEYS)),
        "changed_from_parent": sorted(changed_keys(parent, composed)),
    }
    if set(receipt["changed_from_parent"]) - (set(IMAGE_KEYS) | set(RSTB_KEYS)):
        raise RuntimeError("composition changed a tensor outside the declared union")
    return composed, receipt


def compose(
    protocol_path: Path,
    parent_path: Path,
    image_receipt_path: Path,
    rstb_receipt_path: Path,
    out: Path,
) -> dict:
    protocol = verify_protocol_inputs(
        protocol_path,
        parent_path,
        image_receipt_path,
        rstb_receipt_path,
        out,
    )
    parent = load_state(parent_path)
    image_receipt, image = load_source(image_receipt_path, "image_io_fp16")
    rstb_receipt, rstb = load_source(rstb_receipt_path, "rstb_fusion_fp16")
    if image_receipt["parent_checkpoint_sha256"] != rstb_receipt["parent_checkpoint_sha256"]:
        raise ValueError("source arms do not share a parent checkpoint")
    if (
        image_receipt["retained_precision_training"]["seed"]
        != rstb_receipt["retained_precision_training"]["seed"]
    ):
        raise ValueError("source arms do not share a seed")
    if image_receipt["retained_precision_training"]["seed"] != protocol["seed"]:
        raise ValueError("source seed differs from sealed protocol")
    if image_receipt["parent_checkpoint_sha256"] != protocol["parent"]["checkpoint_sha256"]:
        raise ValueError("source checkpoint differs from sealed protocol parent")
    composed, changes = compose_states(parent, image, rstb)
    out.mkdir(parents=True, exist_ok=False)
    state_path = out / "dense-hard.pt"
    torch.save(composed, state_path)
    receipt = {
        "format": FORMAT,
        "interpretation": (
            "Deterministic union of independently trained, disjoint hard-ternary "
            "image-I/O and RSTB-fusion tensors; no retained FP16 weights and no "
            "additional parameter or scale storage."
        ),
        "seed": image_receipt["retained_precision_training"]["seed"],
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": file_sha256(protocol_path),
        "parent_checkpoint_sha256": image_receipt["parent_checkpoint_sha256"],
        "parent_state": {
            "path": str(parent_path.resolve()),
            "file_sha256": file_sha256(parent_path),
            "tensor_sha256": state_sha256(parent),
        },
        "sources": {
            "image_io_fp16": {
                "receipt": str(image_receipt_path.resolve()),
                "receipt_sha256": file_sha256(image_receipt_path),
                "state_sha256": image_receipt["dense_state"]["tensor_sha256"],
            },
            "rstb_fusion_fp16": {
                "receipt": str(rstb_receipt_path.resolve()),
                "receipt_sha256": file_sha256(rstb_receipt_path),
                "state_sha256": rstb_receipt["dense_state"]["tensor_sha256"],
            },
        },
        "changes": changes,
        "storage_delta_bytes": 0,
        "dense_state": {
            "path": str(state_path.resolve()),
            "file_sha256": file_sha256(state_path),
            "tensor_sha256": state_sha256(composed),
            "tensors": len(composed),
            "values": sum(tensor.numel() for tensor in composed.values()),
        },
    }
    manifest = out / "composition.json"
    manifest.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-state", type=Path, required=True)
    parser.add_argument("--image-materialization", type=Path, required=True)
    parser.add_argument("--rstb-materialization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = compose(
        args.protocol,
        args.parent_state,
        args.image_materialization,
        args.rstb_materialization,
        args.out,
    )
    print(json.dumps(receipt["dense_state"], indent=2))


if __name__ == "__main__":
    main()
