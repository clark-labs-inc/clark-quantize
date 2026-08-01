import json
from pathlib import Path

from full_surface_provenance import file_sha256
from prepare_scale_final_candidate import FINAL_BITS, FORMAT, prepare
from scale_codebook_state import MATERIALIZATION_FORMAT


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def fixture(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    dense = tmp_path / "dense.pt"
    runtime = tmp_path / "runtime.pt"
    dense.write_bytes(b"dense")
    runtime.write_bytes(b"runtime")
    materialization = write_json(tmp_path / "materialization.json", {
        "format": MATERIALIZATION_FORMAT,
        "checkpoint_sha256": file_sha256(checkpoint),
        "all_scale_codebooks_materialized_hard": True,
        "dense_state": {
            "path": str(dense),
            "file_sha256": file_sha256(dense),
            "tensor_sha256": "dense-tensor",
        },
        "runtime_state": {
            "path": str(runtime),
            "file_sha256": file_sha256(runtime),
            "tensor_sha256": "runtime-tensor",
        },
    })
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    binary = artifact_root / "model.bin"
    binary.write_bytes(bytes(range(32)))
    artifact_manifest = write_json(artifact_root / "model.json", {
        "format": "clark-air-browser-v3",
        "materialization_manifest_sha256": file_sha256(materialization),
        "storage_ledger": {
            "binary_bytes": binary.stat().st_size,
            "packed_trunk_bits_per_quantized_weight": 1.64,
        },
    })
    bundle = write_json(tmp_path / "development-bundle.json", {"fold": "dev"})
    q4 = write_json(tmp_path / "development-q4.json", {"fold": "dev"})
    policy = write_json(tmp_path / "policy.json", {
        "reference": {
            "model": "q4-model",
            "revision": "q4-revision",
            "model_sha256": "q4-state",
            "artifact_bytes": 5_676_042,
            "bundle_manifest_sha256": "final-bundle",
            "samples": 2,
        },
        "artifact_gate": {"maximum_fraction_of_reference": 0.5},
    })
    study = write_json(tmp_path / "study.json", {
        "bundle": str(bundle),
        "bundle_sha256": file_sha256(bundle),
        "q4_reference": str(q4),
        "q4_reference_sha256": file_sha256(q4),
        "promotion_policy": str(policy),
        "promotion_policy_sha256": file_sha256(policy),
        "arms": {"uniform_q1": {"bits_by_category": FINAL_BITS}},
        "size_rule": {"uniform_q1_max_packed_trunk_bits": 1.66},
    })
    return write_json(tmp_path / "decision.json", {
        "format": "backdropkit-swin2sr-scale-codebook-qat-decision-v1",
        "study_manifest": str(study),
        "study_manifest_sha256": file_sha256(study),
        "approximately_1_58_endpoint_eligible": True,
        "endpoint_candidate": "uniform_q1",
        "candidates": {
            "uniform_q1": {
                "endpoint_eligible": True,
                "size_passed": True,
                "endpoint_bits_passed": True,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "materialization": str(materialization),
                "materialization_sha256": file_sha256(materialization),
                "artifact_manifest": str(artifact_manifest),
                "artifact_manifest_sha256": file_sha256(artifact_manifest),
                "artifact_binary": str(binary),
                "artifact_binary_sha256": file_sha256(binary),
                "storage_ledger": {
                    "packed_trunk_bits_per_quantized_weight": 1.64,
                },
                "quality_vs_q4": {"passed": True},
            },
        },
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    })


def graph_fixture(tmp_path: Path) -> tuple[Path, Path]:
    graph = tmp_path / "graph.onnx"
    graph.write_bytes(b"weightless graph")
    layout = write_json(tmp_path / "layout.json", {
        "format": "backdropkit-swin2sr-onnx-external-layout-v1",
        "graph_bytes": graph.stat().st_size,
        "graph_sha256": file_sha256(graph),
    })
    return graph, layout


def successor_policy(
    decision: Path,
    tmp_path: Path,
    *,
    quality_change: bool = False,
) -> Path:
    decision_payload = json.loads(decision.read_text())
    study = json.loads(Path(decision_payload["study_manifest"]).read_text())
    development_path = Path(study["promotion_policy"])
    development = json.loads(development_path.read_text())
    successor = dict(development)
    successor["amendment"] = {
        "supersedes": development_path.name,
        "superseded_policy_sha256": study["promotion_policy_sha256"],
        "quality_contract_unchanged": True,
    }
    successor["runtime_harness_sha256"] = {"packed-runtime": "new-hash"}
    if quality_change:
        successor["artifact_gate"] = {"maximum_fraction_of_reference": 0.75}
    return write_json(tmp_path / "successor-policy.json", successor)


def test_prepare_scale_candidate_binds_q1_quality_size_and_transport(tmp_path):
    decision = fixture(tmp_path)
    graph, layout = graph_fixture(tmp_path)
    candidate = prepare(decision, tmp_path / "final", graph, layout)
    assert candidate["format"] == FORMAT
    assert candidate["final_evaluation_eligible"]
    assert candidate["artifact_size_gate"]["passed"]
    assert candidate["endpoint_bits"]["passed"]
    package = candidate["runtime_package"]
    assert package["runtime_package_compression"] == "gzip"
    assert Path(package["runtime_package"]).is_file()
    assert Path(candidate["candidate_manifest"]).is_file()


def test_prepare_scale_candidate_rejects_non_q1_study_arm(tmp_path):
    decision = fixture(tmp_path)
    decision_payload = json.loads(decision.read_text())
    study = Path(decision_payload["study_manifest"])
    study_payload = json.loads(study.read_text())
    study_payload["arms"]["uniform_q1"]["bits_by_category"]["image_io_conv"] = 4
    write_json(study, study_payload)
    decision_payload["study_manifest_sha256"] = file_sha256(study)
    write_json(decision, decision_payload)
    try:
        graph, layout = graph_fixture(tmp_path)
        prepare(decision, tmp_path / "final", graph, layout)
    except ValueError as error:
        assert "not uniform q1" in str(error)
    else:
        raise AssertionError("a mixed-bit bridge must not enter final evaluation")


def test_prepare_scale_candidate_accepts_runtime_only_policy_successor(tmp_path):
    decision = fixture(tmp_path)
    graph, layout = graph_fixture(tmp_path)
    successor = successor_policy(decision, tmp_path)

    candidate = prepare(
        decision,
        tmp_path / "final",
        graph,
        layout,
        successor,
    )

    decision_payload = json.loads(decision.read_text())
    study = json.loads(Path(decision_payload["study_manifest"]).read_text())
    assert candidate["development_promotion_policy"] == study["promotion_policy"]
    assert candidate["promotion_policy"] == str(successor)
    assert candidate["promotion_policy_sha256"] == file_sha256(successor)
    assert candidate["final_evaluation_eligible"]


def test_prepare_scale_candidate_rejects_policy_successor_quality_change(tmp_path):
    decision = fixture(tmp_path)
    graph, layout = graph_fixture(tmp_path)
    successor = successor_policy(decision, tmp_path, quality_change=True)

    try:
        prepare(decision, tmp_path / "final", graph, layout, successor)
    except ValueError as error:
        assert "more than runtime harness identity" in str(error)
    else:
        raise AssertionError("a runtime amendment must not weaken quality gates")
