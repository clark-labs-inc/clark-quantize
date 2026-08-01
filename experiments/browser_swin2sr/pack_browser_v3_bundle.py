#!/usr/bin/env python3
"""Build one deterministic gzip transport from a browser-v3 artifact pair."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path


FORMAT = "backdropkit-packed-browser-v3-bundle-v2"
SUPPORTED_ARTIFACT_FORMATS = {
    "clark-air-browser-v2",
    "clark-air-browser-v3",
    "clark-air-browser-v4",
}
MAGIC_V1 = b"BDKPV3\x00\x01"
MAGIC_V2 = b"BDKPV3\x00\x02"
HEADER_V1 = struct.Struct("<8sII")
HEADER_V2 = struct.Struct("<8sIIII")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def artifact_payload(artifact: Path) -> tuple[bytes, bytes, dict]:
    manifest_path = artifact / "model.json"
    binary_path = artifact / "model.bin"
    manifest_bytes = manifest_path.read_bytes()
    binary = binary_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") not in SUPPORTED_ARTIFACT_FORMATS:
        raise ValueError("packed transport requires a browser-v2 or browser-v3 artifact")
    ledger = manifest.get("storage_ledger") or {}
    if int(ledger.get("binary_bytes", -1)) != len(binary):
        raise ValueError("browser-v3 binary size differs from its storage ledger")
    return manifest_bytes, binary, manifest


def raw_bundle(
    manifest: bytes,
    binary: bytes,
    graph: bytes = b"",
    layout: bytes = b"",
) -> bytes:
    if any(len(value) >= 2**32 for value in (manifest, binary, graph, layout)):
        raise ValueError("browser-v3 bundle component exceeds uint32 length")
    if bool(graph) != bool(layout):
        raise ValueError("external ONNX graph and layout must be bundled together")
    if graph:
        return (
            HEADER_V2.pack(
                MAGIC_V2,
                len(manifest),
                len(binary),
                len(graph),
                len(layout),
            )
            + manifest
            + binary
            + graph
            + layout
        )
    return HEADER_V1.pack(MAGIC_V1, len(manifest), len(binary)) + manifest + binary


def unpack_raw_bundle(payload: bytes) -> tuple[bytes, bytes, bytes, bytes]:
    if len(payload) < HEADER_V1.size:
        raise ValueError("browser-v3 bundle is truncated")
    magic = payload[:8]
    if magic == MAGIC_V1:
        _, manifest_bytes, binary_bytes = HEADER_V1.unpack_from(payload)
        graph_bytes = layout_bytes = 0
        header_bytes = HEADER_V1.size
    elif magic == MAGIC_V2:
        if len(payload) < HEADER_V2.size:
            raise ValueError("browser-v3 WebGPU bundle is truncated")
        (
            _,
            manifest_bytes,
            binary_bytes,
            graph_bytes,
            layout_bytes,
        ) = HEADER_V2.unpack_from(payload)
        header_bytes = HEADER_V2.size
    else:
        raise ValueError("browser-v3 bundle magic differs")
    expected = header_bytes + manifest_bytes + binary_bytes + graph_bytes + layout_bytes
    if len(payload) != expected:
        raise ValueError("browser-v3 bundle length differs")
    manifest_end = header_bytes + manifest_bytes
    binary_end = manifest_end + binary_bytes
    graph_end = binary_end + graph_bytes
    return (
        payload[header_bytes:manifest_end],
        payload[manifest_end:binary_end],
        payload[binary_end:graph_end],
        payload[graph_end:],
    )


def build_bundle(
    artifact: Path,
    out: Path,
    ledger_path: Path,
    graph: Path | None = None,
    layout: Path | None = None,
) -> dict:
    manifest, binary, model = artifact_payload(artifact)
    if bool(graph) != bool(layout):
        raise ValueError("external ONNX graph and layout must be supplied together")
    graph_bytes = graph.read_bytes() if graph else b""
    layout_bytes = layout.read_bytes() if layout else b""
    if layout:
        layout_payload = json.loads(layout_bytes)
        if (
            layout_payload.get("format")
            != "backdropkit-swin2sr-onnx-external-layout-v1"
            or layout_payload.get("graph_sha256") != sha256_bytes(graph_bytes)
            or int(layout_payload.get("graph_bytes", -1)) != len(graph_bytes)
        ):
            raise ValueError("external ONNX graph differs from its layout")
    raw = raw_bundle(manifest, binary, graph_bytes, layout_bytes)
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(compressed)
    receipt = {
        "format": FORMAT,
        "artifact_format": model["format"],
        "artifact_manifest": str((artifact / "model.json").resolve()),
        "artifact_manifest_bytes": len(manifest),
        "artifact_manifest_sha256": sha256_bytes(manifest),
        "artifact_binary": str((artifact / "model.bin").resolve()),
        "artifact_binary_bytes": len(binary),
        "artifact_binary_sha256": sha256_bytes(binary),
        "external_graph": str(graph.resolve()) if graph else None,
        "external_graph_bytes": len(graph_bytes),
        "external_graph_sha256": sha256_bytes(graph_bytes) if graph else None,
        "external_layout": str(layout.resolve()) if layout else None,
        "external_layout_bytes": len(layout_bytes),
        "external_layout_sha256": sha256_bytes(layout_bytes) if layout else None,
        "expanded_bundle_bytes": len(raw),
        "expanded_bundle_sha256": sha256_bytes(raw),
        "runtime_package": str(out.resolve()),
        "runtime_package_bytes": len(compressed),
        "runtime_package_sha256": sha256_bytes(compressed),
        "runtime_package_compression": "gzip",
        "packed_trunk_bits_per_quantized_weight": model["storage_ledger"][
            "packed_trunk_bits_per_quantized_weight"
        ],
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--layout", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_bundle(
        args.artifact,
        args.out,
        args.ledger,
        args.graph,
        args.layout,
    ), indent=2))


if __name__ == "__main__":
    main()
