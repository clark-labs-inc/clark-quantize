import json
from pathlib import Path

from decide_scale_final_promotion import (
    validate_final_chain,
    validate_runtime_fixture,
)
from evaluate_scale_final_candidate import (
    FORMAT as EVALUATION_FORMAT,
    PRECOMMIT_FORMAT,
)
from full_surface_provenance import file_sha256
from prepare_scale_final_candidate import prepare
from prepare_scale_runtime_fixture import FORMAT as FIXTURE_FORMAT, SOURCE_FILES
from test_prepare_scale_final_candidate import fixture, graph_fixture, write_json


def final_chain(tmp_path: Path):
    decision = fixture(tmp_path)
    graph, layout = graph_fixture(tmp_path)
    result = prepare(decision, tmp_path / "candidate", graph, layout)
    candidate_path = Path(result["candidate_manifest"])
    candidate = json.loads(candidate_path.read_text())
    study = json.loads(Path(candidate["scale_study_manifest"]).read_text())
    policy = Path(study["promotion_policy"])
    reference = write_json(tmp_path / "final-q4.json", {"reference": True})
    package = candidate["runtime_package"]
    precommit = write_json(tmp_path / "final-precommit.json", {
        "format": PRECOMMIT_FORMAT,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "scale_decision": candidate["scale_decision"],
        "scale_decision_sha256": candidate["scale_decision_sha256"],
        "lane": candidate["lane"],
        "hard_state_sha256": candidate["hard_state"]["sha256"],
        "runtime_package_sha256": package["runtime_package_sha256"],
        "promotion_policy": str(policy),
        "promotion_policy_sha256": file_sha256(policy),
        "final_q4": str(reference),
        "final_q4_sha256": file_sha256(reference),
    })
    report = write_json(tmp_path / "candidate-metrics.json", {
        "selection_source": "development",
        "selection_lane": candidate["lane"],
        "hard_state_sha256": candidate["hard_state"]["sha256"],
        "materialization_sha256": candidate["materialization"]["sha256"],
        "runtime_state_sha256": candidate["runtime_state"]["sha256"],
        "runtime_artifact_sha256": package["runtime_package_sha256"],
        "runtime_artifact_bytes": package["runtime_package_bytes"],
        "runtime_artifact_compression": package["runtime_package_compression"],
        "runtime_expanded_artifact_sha256": package["expanded_bundle_sha256"],
        "runtime_expanded_artifact_bytes": package["expanded_bundle_bytes"],
        "samples": [{"sha256": "a"}],
        "overall": {"quality": 1},
    })
    receipt = write_json(tmp_path / "final-evaluation-receipt.json", {
        "format": EVALUATION_FORMAT,
        "precommit": str(precommit),
        "precommit_sha256": file_sha256(precommit),
        "candidate_metrics": str(report),
        "candidate_metrics_sha256": file_sha256(report),
        "samples": 1,
        "overall": {"quality": 1},
    })
    return candidate_path, candidate, receipt, policy, reference


def test_packed_q1_final_chain_binds_report_to_candidate(tmp_path):
    candidate_path, candidate, receipt, policy, reference = final_chain(tmp_path)
    loaded, report = validate_final_chain(
        candidate_path, receipt, policy, reference
    )
    assert loaded["lane"] == candidate["lane"]
    assert report["hard_state_sha256"] == candidate["hard_state"]["sha256"]


def test_packed_q1_runtime_fixture_binds_package_and_hard_state(tmp_path):
    candidate_path, candidate, _receipt, _policy, _reference = final_chain(tmp_path)
    package = candidate["runtime_package"]
    repository = Path(__file__).resolve().parents[2]
    fixture_path = write_json(tmp_path / "runtime-fixture.json", {
        "format": FIXTURE_FORMAT,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "hard_state_sha256": candidate["hard_state"]["sha256"],
        "model": package["runtime_package"],
        "model_sha256": package["runtime_package_sha256"],
        "model_bytes": package["runtime_package_bytes"],
        "model_compression": package["runtime_package_compression"],
        "expanded_model_sha256": package["expanded_bundle_sha256"],
        "expanded_model_bytes": package["expanded_bundle_bytes"],
        "artifact_manifest_sha256": candidate["artifact_manifest"]["sha256"],
        "artifact_binary_sha256": candidate["artifact_binary"]["sha256"],
        "input_shape": [1, 3, 128, 128],
        "source_files": {
            relative: file_sha256(repository / relative)
            for relative in SOURCE_FILES
        },
    })
    runtime_path = write_json(tmp_path / "runtime.json", {
        "model_sha256": package["runtime_package_sha256"],
        "fixture_sha256": file_sha256(fixture_path),
        "fixture_hard_state_sha256": candidate["hard_state"]["sha256"],
        "receipts": [],
    })
    validate_runtime_fixture(
        candidate_path,
        candidate,
        fixture_path,
        runtime_path,
        {"runtime_input_shape": [1, 3, 128, 128]},
    )
