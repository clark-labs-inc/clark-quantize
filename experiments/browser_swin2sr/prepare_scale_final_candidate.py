#!/usr/bin/env python3
"""Precommit one development-selected q1 scale artifact before final data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_surface_provenance import file_sha256
from pack_browser_v3_bundle import build_bundle
from scale_codebook_state import MATERIALIZATION_FORMAT


FORMAT = "backdropkit-swin2sr-scale-final-candidate-v1"
DECISION_FORMATS = {
    "backdropkit-swin2sr-scale-codebook-qat-decision-v1",
    "backdropkit-swin2sr-staged-scale-codebook-decision-v1",
}
FINAL_BITS = {
    "core_linear": 1,
    "image_io_conv": 1,
    "position_bias_linear": 1,
    "residual_body_conv": 1,
}
SOURCE_FILES = (
    "experiments/browser_swin2sr/full_surface_provenance.py",
    "experiments/browser_swin2sr/pack_browser_v3_bundle.py",
    "experiments/browser_swin2sr/prepare_scale_final_candidate.py",
    "experiments/browser_swin2sr/scale_codebook_state.py",
)


def file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def size_gate(policy: dict, candidate_bytes: int) -> dict:
    reference_bytes = int(policy["reference"]["artifact_bytes"])
    maximum = float(policy["artifact_gate"]["maximum_fraction_of_reference"])
    fraction = candidate_bytes / reference_bytes
    return {
        "reference_bytes": reference_bytes,
        "candidate_bytes": candidate_bytes,
        "candidate_fraction_of_reference": fraction,
        "maximum_fraction_of_reference": maximum,
        "passed": bool(candidate_bytes > 0 and fraction <= maximum),
    }


def promotion_policy_for_candidate(
    study: dict,
    successor_path: Path | None,
) -> tuple[Path, dict]:
    development_path = Path(study["promotion_policy"])
    if file_sha256(development_path) != study["promotion_policy_sha256"]:
        raise ValueError("scale study input changed: promotion_policy")
    development = json.loads(development_path.read_text())
    if successor_path is None or successor_path == development_path:
        return development_path, development

    successor = json.loads(successor_path.read_text())
    amendment = successor.get("amendment") or {}
    if amendment.get("supersedes") != development_path.name:
        raise ValueError("promotion-policy successor does not supersede study policy")
    if amendment.get("superseded_policy_sha256") != study[
        "promotion_policy_sha256"
    ]:
        raise ValueError("promotion-policy successor binds another study policy")
    if amendment.get("quality_contract_unchanged") is not True:
        raise ValueError("promotion-policy successor changes the quality contract")

    development_contract = dict(development)
    successor_contract = dict(successor)
    development_contract.pop("amendment", None)
    successor_contract.pop("amendment", None)
    if "runtime_harness_sha256" in development_contract:
        successor_contract["runtime_harness_sha256"] = development_contract[
            "runtime_harness_sha256"
        ]
    else:
        successor_contract.pop("runtime_harness_sha256", None)
    if successor_contract != development_contract:
        raise ValueError(
            "promotion-policy successor changes more than runtime harness identity"
        )
    return successor_path, successor


def selected_evidence(decision_path: Path) -> tuple[dict, dict, str, dict, dict]:
    decision = json.loads(decision_path.read_text())
    if decision.get("format") not in DECISION_FORMATS:
        raise ValueError("unsupported scale-codebook decision")
    if not decision.get("approximately_1_58_endpoint_eligible"):
        raise ValueError("scale decision produced no eligible q1 endpoint")
    if decision.get("model_promotion_authorized") or not decision.get(
        "final_heldout_locked"
    ):
        raise ValueError("scale decision violates final-evaluation isolation")
    lane = decision.get("endpoint_candidate")
    candidate = (decision.get("candidates") or {}).get(lane)
    if (
        not candidate
        or not candidate.get("endpoint_eligible")
        or not candidate.get("size_passed")
        or not candidate.get("endpoint_bits_passed")
        or not (candidate.get("quality_vs_q4") or {}).get("passed")
    ):
        raise ValueError("scale endpoint candidate is not development eligible")
    study_path = Path(decision["study_manifest"])
    if file_sha256(study_path) != decision["study_manifest_sha256"]:
        raise ValueError("scale study identity changed")
    study = json.loads(study_path.read_text())
    configured = (study.get("arms") or {}).get(lane)
    if not configured or configured.get("bits_by_category") != FINAL_BITS:
        raise ValueError("selected scale endpoint is not uniform q1")
    for key in ("bundle", "q4_reference", "promotion_policy"):
        if file_sha256(Path(study[key])) != study[f"{key}_sha256"]:
            raise ValueError(f"scale study input changed: {key}")
    return decision, study, lane, candidate, configured


def prepare(
    decision_path: Path,
    out: Path,
    graph: Path | None = None,
    layout: Path | None = None,
    promotion_policy: Path | None = None,
) -> dict:
    decision, study, lane, candidate, _configured = selected_evidence(decision_path)
    checkpoint = Path(candidate["checkpoint"])
    materialization_path = Path(candidate["materialization"])
    if file_sha256(checkpoint) != candidate["checkpoint_sha256"]:
        raise ValueError("selected scale checkpoint identity changed")
    if file_sha256(materialization_path) != candidate["materialization_sha256"]:
        raise ValueError("selected scale materialization identity changed")
    materialization = json.loads(materialization_path.read_text())
    if (
        materialization.get("format") != MATERIALIZATION_FORMAT
        or not materialization.get("all_scale_codebooks_materialized_hard")
    ):
        raise ValueError("selected scale materialization is not fully hard")
    if materialization.get("checkpoint_sha256") != candidate["checkpoint_sha256"]:
        raise ValueError("scale materialization uses another checkpoint")
    artifact_manifest = Path(candidate["artifact_manifest"])
    artifact_binary = Path(candidate["artifact_binary"])
    if file_sha256(artifact_manifest) != candidate["artifact_manifest_sha256"]:
        raise ValueError("selected browser-v3 manifest identity changed")
    if file_sha256(artifact_binary) != candidate["artifact_binary_sha256"]:
        raise ValueError("selected browser-v3 binary identity changed")
    artifact = json.loads(artifact_manifest.read_text())
    if artifact.get("format") != "clark-air-browser-v3":
        raise ValueError("selected artifact is not browser-v3")
    if artifact.get("materialization_manifest_sha256") != candidate[
        "materialization_sha256"
    ]:
        raise ValueError("browser-v3 artifact uses another materialization")
    dense = materialization["dense_state"]
    runtime = materialization["runtime_state"]
    for record, label in ((dense, "dense"), (runtime, "runtime")):
        if file_sha256(Path(record["path"])) != record["file_sha256"]:
            raise ValueError(f"scale {label} state identity changed")
    out.mkdir(parents=True, exist_ok=True)
    package = build_bundle(
        artifact_manifest.parent,
        out / "model.v3.gz",
        out / "model.v3.ledger.json",
        graph,
        layout,
    )
    policy_path, policy = promotion_policy_for_candidate(study, promotion_policy)
    artifact_size = size_gate(policy, int(package["runtime_package_bytes"]))
    trunk_bits = float(candidate["storage_ledger"][
        "packed_trunk_bits_per_quantized_weight"
    ])
    bits_passed = trunk_bits <= float(
        study["size_rule"]["uniform_q1_max_packed_trunk_bits"]
    )
    repository = Path(__file__).resolve().parents[2]
    record = {
        "format": FORMAT,
        "selection_source": "development",
        "scale_decision": str(decision_path),
        "scale_decision_sha256": file_sha256(decision_path),
        "scale_study_manifest": decision["study_manifest"],
        "scale_study_manifest_sha256": decision["study_manifest_sha256"],
        "development_bundle": study["bundle"],
        "development_bundle_sha256": study["bundle_sha256"],
        "development_q4": study["q4_reference"],
        "development_q4_sha256": study["q4_reference_sha256"],
        "development_promotion_policy": study["promotion_policy"],
        "development_promotion_policy_sha256": study["promotion_policy_sha256"],
        "promotion_policy": str(policy_path),
        "promotion_policy_sha256": file_sha256(policy_path),
        "lane": lane,
        "checkpoint": file_record(checkpoint),
        "materialization": file_record(materialization_path),
        "hard_state": {
            "path": dense["path"],
            "bytes": Path(dense["path"]).stat().st_size,
            "sha256": dense["file_sha256"],
            "tensor_sha256": dense["tensor_sha256"],
        },
        "runtime_state": {
            "path": runtime["path"],
            "bytes": Path(runtime["path"]).stat().st_size,
            "sha256": runtime["file_sha256"],
            "tensor_sha256": runtime["tensor_sha256"],
        },
        "artifact_manifest": file_record(artifact_manifest),
        "artifact_binary": file_record(artifact_binary),
        "runtime_package": package,
        "artifact_size_gate": artifact_size,
        "endpoint_bits": {
            "packed_trunk_bits_per_quantized_weight": trunk_bits,
            "maximum": study["size_rule"]["uniform_q1_max_packed_trunk_bits"],
            "passed": bits_passed,
        },
        "development_quality_gate": candidate["quality_vs_q4"],
        "webgpu_external_graph_ready": bool(graph and layout),
        "source_files": {
            relative: file_sha256(repository / relative)
            for relative in SOURCE_FILES
        },
        "final_evaluation_eligible": bool(
            candidate["quality_vs_q4"]["passed"]
            and artifact_size["passed"]
            and bits_passed
            and graph
            and layout
        ),
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    path = out / "final-candidate.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("existing scale final candidate differs")
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return {**record, "candidate_manifest": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, required=True)
    args = parser.parse_args()
    record = prepare(
        args.scale_decision,
        args.out,
        args.graph,
        args.layout,
        args.promotion_policy,
    )
    print(json.dumps({
        "candidate_manifest": record["candidate_manifest"],
        "lane": record["lane"],
        "runtime_package_bytes": record["runtime_package"]["runtime_package_bytes"],
        "final_evaluation_eligible": record["final_evaluation_eligible"],
    }, indent=2))


if __name__ == "__main__":
    main()
