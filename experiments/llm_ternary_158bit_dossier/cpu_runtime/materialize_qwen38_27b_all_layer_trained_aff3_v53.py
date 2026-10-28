#!/usr/bin/env python3
"""Derive all 64 mmap-ready AFF3_X16 images from the passed v44 algebra."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np

from clark_quantize.qwen38_aff3_runtime_image import (
    Aff3RuntimeImage,
    build_from_compressed_artifact,
)
from clark_quantize.qwen38_affine_butterfly import deterministic_givens_pairings


MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TRAINING_SCHEMA = "qwen38_27b_all_layer_mixed_training_receipt_v44"
TRAINING_LAYER_SCHEMA = "qwen38_27b_all_layer_mixed_trained_layer_v44"
DIRECT_SCHEMA = "qwen38_27b_all_layer_mixed_direct_receipt_v43"
DIRECT_LAYER_SCHEMA = "qwen38_27b_all_layer_mixed_direct_layer_v43"
TRAINED_MANIFEST_SCHEMA = "qwen38_affine_butterfly_qat_site_v1"
DIRECT_MANIFEST_SCHEMA = "qwen38_affine_butterfly_site_v1"
LAYER_RECEIPT_SCHEMA = "qwen38_27b_trained_aff3_runtime_layer_v53"
RECEIPT_SCHEMA = "qwen38_27b_all_layer_trained_aff3_runtime_v53"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def safe_child(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("AFF3 artifact reference must be relative")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("AFF3 artifact reference escapes its root") from error
    return resolved


def validate_parent_receipts(
    training: dict[str, Any],
    direct: dict[str, Any],
    *,
    direct_receipt_sha256: str,
) -> None:
    trained_layers = training.get("layers", [])
    direct_layers = direct.get("layers", [])
    if (
        training.get("schema") != TRAINING_SCHEMA
        or direct.get("schema") != DIRECT_SCHEMA
        or training.get("model") != MODEL
        or direct.get("model") != MODEL
        or training.get("revision") != REVISION
        or direct.get("revision") != REVISION
        or training.get("direct_receipt_sha256") != direct_receipt_sha256
        or training.get("gate", {}).get("passed") is not True
        or direct.get("gate", {}).get("passed") is not True
        or len(trained_layers) != 64
        or len(direct_layers) != 64
        or [item.get("layer") for item in trained_layers] != list(range(64))
        or [item.get("layer") for item in direct_layers] != list(range(64))
        or not all(item.get("passed") is True for item in trained_layers)
        or not all(item.get("passed") is True for item in direct_layers)
        or training.get("complete") is not True
        or direct.get("complete") is not True
        or training.get("terminal") is not True
        or direct.get("terminal") is not True
    ):
        raise ValueError("v53 parent receipt identity or denominator changed")


def load_angles(root: Path, descriptor: dict[str, Any]) -> np.ndarray:
    path = safe_child(root, str(descriptor.get("path", "")))
    shape = tuple(int(value) for value in descriptor.get("shape", []))
    if (
        len(shape) != 2
        or descriptor.get("dtype") != "float16"
        or not path.is_file()
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise ValueError("trained AFF3 angle artifact changed")
    values = np.fromfile(path, dtype="<f2").reshape(shape)
    if not bool(np.isfinite(values.astype(np.float32)).all()):
        raise ValueError("trained AFF3 angles are non-finite")
    return values


def validate_layer_sources(
    *,
    layer: int,
    trained_result: dict[str, Any],
    direct_result: dict[str, Any],
    trained_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    training: dict[str, Any],
    direct: dict[str, Any],
) -> None:
    trained_tensors = trained_manifest.get("tensors", [])
    direct_tensors = direct_manifest.get("tensors", [])
    if (
        trained_result.get("schema") != TRAINING_LAYER_SCHEMA
        or direct_result.get("schema") != DIRECT_LAYER_SCHEMA
        or trained_result.get("layer") != layer
        or direct_result.get("layer") != layer
        or trained_result.get("repository_commit")
        != training.get("repository_commit")
        or direct_result.get("repository_commit") != direct.get("repository_commit")
        or trained_result.get("preregistration_sha256")
        != training.get("preregistration_sha256")
        or direct_result.get("preregistration_sha256")
        != direct.get("preregistration_sha256")
        or trained_result.get("passed") is not True
        or direct_result.get("passed") is not True
        or trained_result.get("complete") is not True
        or direct_result.get("complete") is not True
        or trained_result.get("terminal") is not True
        or direct_result.get("terminal") is not True
        or trained_manifest.get("schema") != TRAINED_MANIFEST_SCHEMA
        or direct_manifest.get("schema") != DIRECT_MANIFEST_SCHEMA
        or trained_manifest.get("layer") != layer
        or direct_manifest.get("layer") != layer
        or trained_manifest.get("site") != "attention_output"
        or direct_manifest.get("site") != "attention_output"
        or trained_manifest.get("tensor_names") != direct_manifest.get("tensor_names")
        or len(trained_tensors) != 1
        or len(direct_tensors) != 1
        or trained_tensors[0].get("tensor") != direct_tensors[0].get("tensor")
        or trained_tensors[0].get("shape") != direct_tensors[0].get("shape")
        or trained_manifest.get("repository_commit")
        != training.get("repository_commit")
        or trained_manifest.get("preregistration_sha256")
        != training.get("preregistration_sha256")
        or direct_manifest.get("repository_commit") != direct.get("repository_commit")
        or direct_manifest.get("preregistration_sha256")
        != direct.get("preregistration_sha256")
        or trained_manifest.get("complete") is not True
        or direct_manifest.get("complete") is not True
        or trained_manifest.get("terminal") is not True
        or direct_manifest.get("terminal") is not True
        or int(direct_manifest.get("input_dimension", -1))
        != int(trained_tensors[0]["shape"][1])
    ):
        raise ValueError(f"v53 layer {layer} AFF3 source identity changed")


def validate_existing_layer(
    image_path: Path,
    receipt_path: Path,
    *,
    layer: int,
    training_receipt_sha256: str,
    direct_receipt_sha256: str,
) -> dict[str, Any]:
    if not image_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"partial v53 layer {layer} materialization")
    receipt = json.loads(receipt_path.read_text())
    with Aff3RuntimeImage(image_path) as image:
        image.verify_all_sections()
        if (
            receipt.get("schema") != LAYER_RECEIPT_SCHEMA
            or receipt.get("layer") != layer
            or receipt.get("training_receipt_sha256")
            != training_receipt_sha256
            or receipt.get("direct_receipt_sha256") != direct_receipt_sha256
            or receipt.get("runtime_image", {}).get("bytes") != image.file_bytes
            or receipt.get("runtime_image", {}).get("root_sha256")
            != image.root_sha256
            or receipt.get("runtime_image", {}).get("sha256")
            != sha256_file(image_path)
            or receipt.get("complete") is not True
            or receipt.get("terminal") is not True
        ):
            raise ValueError(f"existing v53 layer {layer} changed")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-receipt", required=True, type=Path)
    parser.add_argument("--direct-receipt", required=True, type=Path)
    parser.add_argument("--trained-artifact-root", required=True, type=Path)
    parser.add_argument("--direct-artifact-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    training_receipt_sha = sha256_file(args.training_receipt)
    direct_receipt_sha = sha256_file(args.direct_receipt)
    training = json.loads(args.training_receipt.read_text())
    direct = json.loads(args.direct_receipt.read_text())
    validate_parent_receipts(
        training, direct, direct_receipt_sha256=direct_receipt_sha
    )
    if os.statvfs(args.output_root.parent).f_bavail * os.statvfs(
        args.output_root.parent
    ).f_frsize < 20 * 1024**3:
        raise SystemExit("infrastructure-invalid: v53 disk floor failed")
    repository_commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if args.receipt.exists():
        existing = json.loads(args.receipt.read_text())
        if (
            existing.get("schema") != RECEIPT_SCHEMA
            or existing.get("repository_commit") != repository_commit
            or existing.get("training_receipt_sha256") != training_receipt_sha
            or existing.get("direct_receipt_sha256") != direct_receipt_sha
            or len(existing.get("layers", [])) != 64
            or existing.get("complete") is not True
            or existing.get("terminal") is not True
        ):
            raise ValueError("existing v53 terminal receipt changed")
        for layer in range(64):
            validate_existing_layer(
                args.output_root / f"layer-{layer:02d}.aff3x16",
                args.output_root / f"layer-{layer:02d}.json",
                layer=layer,
                training_receipt_sha256=training_receipt_sha,
                direct_receipt_sha256=direct_receipt_sha,
            )
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for layer, (trained_result, direct_result) in enumerate(
        zip(training["layers"], direct["layers"])
    ):
        image_path = args.output_root / f"layer-{layer:02d}.aff3x16"
        layer_receipt_path = args.output_root / f"layer-{layer:02d}.json"
        if image_path.exists() or layer_receipt_path.exists():
            records.append(
                validate_existing_layer(
                    image_path,
                    layer_receipt_path,
                    layer=layer,
                    training_receipt_sha256=training_receipt_sha,
                    direct_receipt_sha256=direct_receipt_sha,
                )
            )
            continue
        manifests = trained_result.get("site_manifests", [])
        if len(manifests) != 1:
            raise ValueError(f"v53 layer {layer} trained manifest denominator changed")
        trained_manifest_path = safe_child(
            args.trained_artifact_root, str(manifests[0])
        )
        direct_manifest_path = safe_child(
            args.direct_artifact_root / "aff3-base",
            str(direct_result.get("aff3_manifest", "")),
        )
        trained_manifest = json.loads(trained_manifest_path.read_text())
        direct_manifest = json.loads(direct_manifest_path.read_text())
        validate_layer_sources(
            layer=layer,
            trained_result=trained_result,
            direct_result=direct_result,
            trained_manifest=trained_manifest,
            direct_manifest=direct_manifest,
            training=training,
            direct=direct,
        )
        angles = load_angles(args.trained_artifact_root, trained_manifest["angles"])
        pairings = deterministic_givens_pairings(
            int(direct_manifest["input_dimension"]),
            stages=int(angles.shape[0]),
            seed=int(direct_manifest["pairing_seed"]),
        ).numpy()
        image_receipt = build_from_compressed_artifact(
            artifact_root=args.trained_artifact_root,
            descriptor=trained_manifest["tensors"][0],
            angles=angles,
            pairings=pairings,
            layer=layer,
            path=image_path,
        )
        record = {
            "schema": LAYER_RECEIPT_SCHEMA,
            "model": MODEL,
            "revision": REVISION,
            "repository_commit": repository_commit,
            "layer": layer,
            "training_receipt_sha256": training_receipt_sha,
            "direct_receipt_sha256": direct_receipt_sha,
            "trained_manifest_sha256": sha256_file(trained_manifest_path),
            "direct_manifest_sha256": sha256_file(direct_manifest_path),
            "runtime_image": {
                "path": image_path.name,
                "bytes": image_path.stat().st_size,
                "sha256": sha256_file(image_path),
                "root_sha256": image_receipt["root_sha256"],
                "shape": image_receipt["shape"],
            },
            "persistent_dense_weight_bytes": 0,
            "complete": True,
            "terminal": True,
        }
        atomic_json(record, layer_receipt_path)
        records.append(record)
        print(json.dumps({"event": "v53_aff3_layer_terminal", "layer": layer}), flush=True)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "model": MODEL,
        "revision": REVISION,
        "repository_commit": repository_commit,
        "training_receipt_sha256": training_receipt_sha,
        "direct_receipt_sha256": direct_receipt_sha,
        "layers": records,
        "layer_count": len(records),
        "total_runtime_image_bytes": sum(
            item["runtime_image"]["bytes"] for item in records
        ),
        "persistent_dense_weight_bytes": 0,
        "all_images_authenticated": len(records) == 64,
        "claim_boundary": (
            "Exact mmap-ready derivation of the passed v44 AFF3 algebra only; "
            "this is not whole-model quality, throughput, energy, novelty, or "
            "scientific-breakthrough evidence."
        ),
        "complete": True,
        "terminal": True,
    }
    if len(records) != 64:
        raise AssertionError("v53 AFF3 runtime denominator changed")
    atomic_json(receipt, args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
