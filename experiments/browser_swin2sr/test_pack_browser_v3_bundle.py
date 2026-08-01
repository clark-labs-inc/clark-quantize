import gzip
import json
from pathlib import Path

from pack_browser_v3_bundle import (
    FORMAT,
    build_bundle,
    raw_bundle,
    unpack_raw_bundle,
)


def artifact_fixture(root: Path, artifact_format="clark-air-browser-v3") -> Path:
    root.mkdir()
    binary = bytes(range(31))
    manifest = {
        "format": artifact_format,
        "storage_ledger": {
            "binary_bytes": len(binary),
            "packed_trunk_bits_per_quantized_weight": 1.64465,
        },
    }
    (root / "model.json").write_text(json.dumps(manifest, sort_keys=True))
    (root / "model.bin").write_bytes(binary)
    return root


def test_raw_bundle_round_trips_exact_component_bytes():
    manifest = b'{"format":"clark-air-browser-v3"}'
    binary = b"\x00\x01\xffpayload"
    assert unpack_raw_bundle(raw_bundle(manifest, binary)) == (
        manifest, binary, b"", b""
    )


def test_transport_is_deterministic_and_receipted(tmp_path):
    artifact = artifact_fixture(tmp_path / "artifact")
    first = tmp_path / "first.gz"
    second = tmp_path / "second.gz"
    first_receipt = build_bundle(artifact, first, tmp_path / "first.json")
    second_receipt = build_bundle(artifact, second, tmp_path / "second.json")
    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["format"] == FORMAT
    assert first_receipt["runtime_package_compression"] == "gzip"
    manifest, binary, graph, layout = unpack_raw_bundle(
        gzip.decompress(first.read_bytes())
    )
    assert manifest == (artifact / "model.json").read_bytes()
    assert binary == (artifact / "model.bin").read_bytes()
    assert graph == layout == b""


def test_transport_accepts_browser_v2_mixed_precision_artifact(tmp_path):
    artifact = artifact_fixture(
        tmp_path / "artifact-v2",
        artifact_format="clark-air-browser-v2",
    )
    receipt = build_bundle(
        artifact,
        tmp_path / "mixed.gz",
        tmp_path / "mixed.json",
    )
    assert receipt["artifact_format"] == "clark-air-browser-v2"


def test_webgpu_bundle_round_trips_graph_and_layout(tmp_path):
    artifact = artifact_fixture(tmp_path / "artifact")
    graph = tmp_path / "graph.onnx"
    graph.write_bytes(b"weightless graph")
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({
        "format": "backdropkit-swin2sr-onnx-external-layout-v1",
        "graph_bytes": graph.stat().st_size,
        "graph_sha256": __import__("hashlib").sha256(graph.read_bytes()).hexdigest(),
    }))
    receipt = build_bundle(
        artifact,
        tmp_path / "webgpu.gz",
        tmp_path / "webgpu.json",
        graph,
        layout,
    )
    components = unpack_raw_bundle(gzip.decompress(
        Path(receipt["runtime_package"]).read_bytes()
    ))
    assert components[2] == graph.read_bytes()
    assert components[3] == layout.read_bytes()


def test_transport_rejects_a_binary_ledger_mismatch(tmp_path):
    artifact = artifact_fixture(tmp_path / "artifact")
    manifest = json.loads((artifact / "model.json").read_text())
    manifest["storage_ledger"]["binary_bytes"] += 1
    (artifact / "model.json").write_text(json.dumps(manifest))
    try:
        build_bundle(artifact, tmp_path / "bad.gz", tmp_path / "bad.json")
    except ValueError as error:
        assert "binary size" in str(error)
    else:
        raise AssertionError("a mismatched artifact ledger must be rejected")
