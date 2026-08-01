import json
from pathlib import Path

from evaluate_scale_final_candidate import (
    candidate_report,
    selected_candidate,
    validate_candidate_artifacts,
    validate_development_selection,
)
from prepare_scale_final_candidate import prepare
from test_prepare_scale_final_candidate import (
    fixture,
    graph_fixture,
    successor_policy,
    write_json,
)


def prepared(tmp_path: Path):
    decision = fixture(tmp_path)
    graph, layout = graph_fixture(tmp_path)
    result = prepare(decision, tmp_path / "candidate", graph, layout)
    manifest = Path(result["candidate_manifest"])
    candidate = selected_candidate(manifest)
    study = json.loads(Path(candidate["scale_study_manifest"]).read_text())
    policy = Path(study["promotion_policy"])
    final_bundle = write_json(tmp_path / "final-bundle.json", {
        "manifest_sha256": "final-bundle",
        "samples": 2,
    })
    final_q4 = write_json(tmp_path / "final-q4.json", {
        "model": "q4-model",
        "model_revision": "q4-revision",
        "state_sha256": "q4-state",
        "artifact_bytes": 5_676_042,
        "bundle_manifest_sha256": "final-bundle",
        "device": "cpu",
    })
    return candidate, policy, final_bundle, final_q4


def test_scale_final_chain_validates_before_opening_final_data(tmp_path):
    candidate, policy, final_bundle_path, final_q4_path = prepared(tmp_path)
    validate_candidate_artifacts(candidate)
    final_bundle = json.loads(final_bundle_path.read_text())
    final_q4 = json.loads(final_q4_path.read_text())
    validate_development_selection(
        candidate,
        policy,
        final_bundle_path,
        final_bundle,
        final_q4_path,
        final_q4,
    )
    metrics = {
        "state_sha256": candidate["hard_state"]["sha256"],
        "bundle_manifest_sha256": "final-bundle",
        "device": "cpu",
        "samples": [{"sha256": "a"}, {"sha256": "b"}],
        "overall": {},
    }
    report = candidate_report(metrics, candidate, final_bundle, final_q4)
    assert report["runtime_artifact_compression"] == "gzip"
    assert report["runtime_artifact_sha256"] == candidate["runtime_package"][
        "runtime_package_sha256"
    ]
    assert report["hard_state_sha256"] == candidate["hard_state"]["sha256"]


def test_scale_final_chain_rejects_tampered_transport(tmp_path):
    candidate, _policy, _bundle, _q4 = prepared(tmp_path)
    package = Path(candidate["runtime_package"]["runtime_package"])
    package.write_bytes(package.read_bytes() + b"tamper")
    try:
        validate_candidate_artifacts(candidate)
    except ValueError as error:
        assert "runtime package identity" in str(error)
    else:
        raise AssertionError("a modified runtime package must be rejected")


def test_scale_final_chain_accepts_runtime_only_policy_successor(tmp_path):
    decision = fixture(tmp_path)
    graph, layout = graph_fixture(tmp_path)
    policy = successor_policy(decision, tmp_path)
    result = prepare(
        decision,
        tmp_path / "candidate",
        graph,
        layout,
        policy,
    )
    candidate = selected_candidate(Path(result["candidate_manifest"]))
    final_bundle_path = write_json(tmp_path / "final-bundle.json", {
        "manifest_sha256": "final-bundle",
        "samples": 2,
    })
    final_q4_path = write_json(tmp_path / "final-q4.json", {
        "model": "q4-model",
        "model_revision": "q4-revision",
        "state_sha256": "q4-state",
        "artifact_bytes": 5_676_042,
        "bundle_manifest_sha256": "final-bundle",
        "device": "cpu",
    })

    validate_development_selection(
        candidate,
        policy,
        final_bundle_path,
        json.loads(final_bundle_path.read_text()),
        final_q4_path,
        json.loads(final_q4_path.read_text()),
    )
