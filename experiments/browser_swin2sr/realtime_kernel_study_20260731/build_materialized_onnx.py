#!/usr/bin/env python3
"""Build a decoder-free ONNX control from the released Ternary v2 state."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import onnx
from safetensors.torch import load_file
from transformers import Swin2SRForImageSuperResolution


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
RELEASE = EXPERIMENT / "huggingface/clark-swin2sr-lightweight-x2-1.58bit"
BASE_ONNX = Path(
    "/Users/stan/.cache/huggingface/hub/"
    "models--Xenova--swin2SR-lightweight-x2-64/snapshots/"
    "54ccc3f9ed678912a9c47dc14f6c478855380a2b/onnx/model.onnx"
)
OUTPUT = HERE / "materialized_dense/model.onnx"

sys.path.insert(0, str(EXPERIMENT))
from build_portable_onnx_artifact import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    map_state_to_onnx,
    replace_initializers,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    teacher_state = {
        name: tensor.detach().clone() for name, tensor in teacher.state_dict().items()
    }
    candidate_state = load_file(RELEASE / "model.safetensors")
    if set(candidate_state) != set(teacher_state):
        raise ValueError("released state does not exactly match the teacher state surface")
    model = onnx.load_model(BASE_ONNX)
    mapping = map_state_to_onnx(teacher_state, list(model.graph.initializer))
    replace_initializers(model, candidate_state, mapping)
    onnx.checker.check_model(model, full_check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, OUTPUT)
    package = OUTPUT.with_suffix(".onnx.gz")
    with OUTPUT.open("rb") as source, package.open("wb") as target:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, fileobj=target, mtime=0
        ) as compressed:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                compressed.write(chunk)
    receipt = {
        "kind": "decoder-free materialized control",
        "source_state": str(RELEASE / "model.safetensors"),
        "source_state_sha256": digest(RELEASE / "model.safetensors"),
        "base_onnx": str(BASE_ONNX),
        "base_onnx_sha256": digest(BASE_ONNX),
        "model": str(OUTPUT),
        "model_bytes": OUTPUT.stat().st_size,
        "model_sha256": digest(OUTPUT),
        "package": str(package),
        "package_bytes": package.stat().st_size,
        "package_sha256": digest(package),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "operator_counts": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
    }
    (OUTPUT.parent / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
