#!/usr/bin/env python3
"""Build the one development-selected optimizer artifact before final data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from build_portable_onnx_artifact import build_artifact
from final_artifact_contract import file_sha256, validate_artifacts


FORMAT = "backdropkit-swin2sr-optimizer-transfer-decision-v1"
CANDIDATE_FORMAT = "backdropkit-swin2sr-final-candidate-v1"
SOURCE_FILES = (
    "clark_air/quant/binary.py",
    "clark_air/quant/optimal_ternary.py",
    "clark_quantize/bonsai_muon.py",
    "clark_quantize/bonsai_qat8.py",
    "clark_quantize/qat_optimizer_variants.py",
    "clark_quantize/ternary_kd.py",
    "experiments/browser_swin2sr/artifact_state.py",
    "experiments/browser_swin2sr/build_portable_onnx_artifact.py",
    "experiments/browser_swin2sr/data_pipeline.py",
    "experiments/browser_swin2sr/evaluate_hard_started_checkpoints.py",
    "experiments/browser_swin2sr/final_artifact_contract.py",
    "experiments/browser_swin2sr/full_surface_optimizer.py",
    "experiments/browser_swin2sr/full_surface_provenance.py",
    "experiments/browser_swin2sr/island_policy.py",
    "experiments/browser_swin2sr/materialize_surface_states.py",
    "experiments/browser_swin2sr/mixed_precision_artifact.py",
    "experiments/browser_swin2sr/onnx_packed_bridge.py",
    "experiments/browser_swin2sr/optimizer_controls.py",
    "experiments/browser_swin2sr/prepare_optimizer_final_candidate.py",
    "experiments/browser_swin2sr/qat_objective.py",
    "experiments/browser_swin2sr/quantizer_controls.py",
    "experiments/browser_swin2sr/recovery_controls.py",
    "experiments/browser_swin2sr/retained_precision_islands.py",
    "experiments/browser_swin2sr/rewrite_onnx_ternary.py",
    "experiments/browser_swin2sr/rotation_control.py",
    "experiments/browser_swin2sr/run_state.py",
    "experiments/browser_swin2sr/surface_artifact_state.py",
    "experiments/browser_swin2sr/surface_curriculum.py",
    "experiments/browser_swin2sr/surface_model_state.py",
    "experiments/browser_swin2sr/surface_quantization.py",
    "experiments/browser_swin2sr/surface_run_state.py",
    "experiments/browser_swin2sr/surface_schedule.py",
    "experiments/browser_swin2sr/train_swin2sr_qat.py",
    "experiments/browser_swin2sr/training_data_contract.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-decision", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, required=True)
    parser.add_argument("--base-onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def selected_optimizer_evidence(
    decision_path: Path,
    policy_path: Path,
) -> tuple[dict, dict, dict]:
    decision = json.loads(decision_path.read_text())
    if decision.get("format") != FORMAT:
        raise RuntimeError("unsupported optimizer decision")
    if not decision.get("optimizer_recipe_selected"):
        raise RuntimeError("optimizer decision selected no candidate")
    if decision.get("model_promotion_authorized"):
        raise RuntimeError("optimizer decision improperly authorizes promotion")
    if not decision.get("final_heldout_locked"):
        raise RuntimeError("optimizer decision does not lock final heldout")
    if file_sha256(policy_path) != decision["promotion_policy_sha256"]:
        raise RuntimeError("optimizer decision promotion policy differs")
    study_path = Path(decision["study_manifest"])
    if file_sha256(study_path) != decision["study_manifest_sha256"]:
        raise RuntimeError("optimizer study identity changed")
    study = json.loads(study_path.read_text())
    if study.get("selection_source") != "development":
        raise RuntimeError("optimizer study is not development-only")
    selected = decision["selected_optimizer"]
    matching = [
        candidate
        for candidate in decision.get("candidates", [])
        if candidate.get("optimizer_arm") == selected
    ]
    if len(matching) != 1:
        raise RuntimeError("optimizer decision has no unique selected arm")
    candidate = matching[0]
    if (
        not candidate.get("eligible")
        or not (
            candidate.get("development_quality_gate") or {}
        ).get("passed")
    ):
        raise RuntimeError("selected optimizer failed development quality")
    evidence = candidate["canonical_evidence"]
    expected = {
        "checkpoint": decision["selected_checkpoint"],
        "checkpoint_sha256": decision["selected_checkpoint_sha256"],
        "materialization": decision["selected_materialization"],
        "materialization_sha256": decision[
            "selected_materialization_sha256"
        ],
        "dense_state": decision["selected_dense_state"],
        "dense_state_sha256": decision["selected_dense_state_sha256"],
        "runtime_state": decision["selected_runtime_state"],
        "runtime_state_sha256": decision["selected_runtime_state_sha256"],
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise RuntimeError("selected optimizer canonical evidence differs")
    for key in (
        "checkpoint",
        "materialization",
        "dense_state",
        "runtime_state",
    ):
        expected_hash = evidence[
            f"{key}_sha256"
            if key != "dense_state"
            else "dense_state_sha256"
        ]
        if file_sha256(Path(evidence[key])) != expected_hash:
            raise RuntimeError(f"selected optimizer {key} identity changed")
    return decision, study, candidate


def artifact_size_gate(policy: dict, artifact_bytes: int) -> dict:
    reference_bytes = int(policy["reference"]["artifact_bytes"])
    maximum_fraction = float(
        policy["artifact_gate"]["maximum_fraction_of_reference"]
    )
    if reference_bytes <= 0:
        raise RuntimeError("artifact reference bytes must be positive")
    if not 0 < maximum_fraction < 1:
        raise RuntimeError(
            "artifact maximum fraction must be between zero and one"
        )
    if artifact_bytes <= 0:
        raise RuntimeError("candidate artifact bytes must be positive")
    fraction = artifact_bytes / reference_bytes
    return {
        "reference_bytes": reference_bytes,
        "candidate_bytes": artifact_bytes,
        "candidate_fraction_of_reference": fraction,
        "maximum_fraction_of_reference": maximum_fraction,
        "passed": bool(
            artifact_bytes < reference_bytes
            and fraction <= maximum_fraction
        ),
    }


def write_or_verify(path: Path, stable: dict) -> dict:
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = {
            key: value
            for key, value in existing.items()
            if key != "created_at_unix"
        }
        if comparable != stable:
            raise RuntimeError(f"existing receipt differs: {path}")
        return existing
    receipt = {**stable, "created_at_unix": time.time()}
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def candidate_record(
    *,
    decision_path: Path,
    decision: dict,
    study: dict,
    selected: dict,
    policy_path: Path,
    materialization: dict,
    artifact: dict,
    precommit_path: Path,
    precommit: dict,
) -> dict:
    evidence = selected["canonical_evidence"]
    portable = Path(artifact["runtime_package"]).with_suffix("")
    if portable.suffix != ".onnx":
        raise RuntimeError("runtime package does not identify portable ONNX")
    ledger = Path(artifact["ledger"])
    size = artifact_size_gate(
        json.loads(policy_path.read_text()),
        int(artifact["runtime_package_bytes"]),
    )
    record = {
        "format": CANDIDATE_FORMAT,
        "selection_source": "development",
        "optimizer_decision": str(decision_path),
        "optimizer_decision_sha256": file_sha256(decision_path),
        "optimizer_study_manifest": decision["study_manifest"],
        "optimizer_study_manifest_sha256": decision[
            "study_manifest_sha256"
        ],
        "development_bundle": study["bundle"],
        "development_bundle_sha256": study["bundle_sha256"],
        "development_q4": study["q4_reference"],
        "development_q4_sha256": study["q4_reference_sha256"],
        "promotion_policy": str(policy_path),
        "promotion_policy_sha256": file_sha256(policy_path),
        "artifact_precommit": str(precommit_path),
        "artifact_precommit_sha256": file_sha256(precommit_path),
        "base_onnx": precommit["base_onnx"],
        "base_onnx_sha256": precommit["base_onnx_sha256"],
        "lane": decision["selected_optimizer"],
        "checkpoint": evidence["checkpoint"],
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "materialization": evidence["materialization"],
        "materialization_sha256": evidence["materialization_sha256"],
        "hard_state": evidence["dense_state"],
        "hard_state_sha256": evidence["dense_state_sha256"],
        "runtime_state": evidence["runtime_state"],
        "runtime_state_sha256": evidence["runtime_state_sha256"],
        "portable_onnx": str(portable),
        "portable_onnx_bytes": int(artifact["model_bytes"]),
        "portable_onnx_sha256": artifact["model_sha256"],
        "portable_ledger": str(ledger),
        "portable_ledger_sha256": file_sha256(ledger),
        "runtime_package": artifact["runtime_package"],
        "runtime_package_bytes": int(artifact["runtime_package_bytes"]),
        "runtime_package_sha256": artifact["runtime_package_sha256"],
        "runtime_package_compression": artifact[
            "runtime_package_compression"
        ],
        "quantized_tensors": int(artifact["tensor_count"]),
        "quantized_values": int(artifact["values"]),
        "weight_bits_per_value_including_scales": float(
            artifact["weight_bits_per_value_including_scales"]
        ),
        "artifact_size_gate": size,
        "development_quality_gate": selected[
            "development_quality_gate"
        ],
        "final_evaluation_eligible": bool(
            size["passed"]
            and selected["development_quality_gate"]["passed"]
        ),
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    validate_artifacts(record)
    if materialization["dense_state"]["file_sha256"] != record[
        "hard_state_sha256"
    ]:
        raise RuntimeError("candidate dense state differs from materialization")
    return record


def validate_existing_candidate(
    candidate: dict,
    *,
    decision_path: Path,
    decision: dict,
    study: dict,
    selected: dict,
    policy_path: Path,
    precommit_path: Path,
    precommit: dict,
) -> None:
    evidence = selected["canonical_evidence"]
    expected = {
        "format": CANDIDATE_FORMAT,
        "selection_source": "development",
        "optimizer_decision": str(decision_path),
        "optimizer_decision_sha256": file_sha256(decision_path),
        "optimizer_study_manifest": decision["study_manifest"],
        "optimizer_study_manifest_sha256": decision[
            "study_manifest_sha256"
        ],
        "development_bundle": study["bundle"],
        "development_bundle_sha256": study["bundle_sha256"],
        "development_q4": study["q4_reference"],
        "development_q4_sha256": study["q4_reference_sha256"],
        "promotion_policy": str(policy_path),
        "promotion_policy_sha256": file_sha256(policy_path),
        "artifact_precommit": str(precommit_path),
        "artifact_precommit_sha256": file_sha256(precommit_path),
        "base_onnx": precommit["base_onnx"],
        "base_onnx_sha256": precommit["base_onnx_sha256"],
        "lane": decision["selected_optimizer"],
        "checkpoint": evidence["checkpoint"],
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "materialization": evidence["materialization"],
        "materialization_sha256": evidence["materialization_sha256"],
        "hard_state": evidence["dense_state"],
        "hard_state_sha256": evidence["dense_state_sha256"],
        "runtime_state": evidence["runtime_state"],
        "runtime_state_sha256": evidence["runtime_state_sha256"],
        "development_quality_gate": selected[
            "development_quality_gate"
        ],
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }
    if any(candidate.get(key) != value for key, value in expected.items()):
        raise RuntimeError("existing final candidate lineage differs")
    policy = json.loads(policy_path.read_text())
    expected_size = artifact_size_gate(
        policy,
        int(candidate["runtime_package_bytes"]),
    )
    if candidate.get("artifact_size_gate") != expected_size:
        raise RuntimeError("existing final candidate size gate differs")
    eligible = bool(
        expected_size["passed"]
        and selected["development_quality_gate"]["passed"]
    )
    if candidate.get("final_evaluation_eligible") != eligible:
        raise RuntimeError("existing final candidate eligibility differs")
    validate_artifacts(candidate)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidate_path = args.out / "final-candidate.json"
    decision, study, selected = selected_optimizer_evidence(
        args.optimizer_decision,
        args.promotion_policy,
    )
    materialization_path = Path(
        selected["canonical_evidence"]["materialization"]
    )
    materialization = json.loads(materialization_path.read_text())
    repository = Path(__file__).resolve().parents[2]
    sources = {
        name: file_sha256(repository / name)
        for name in SOURCE_FILES
    }
    portable_path = args.out / "model.onnx"
    stable_precommit = {
        "format": "backdropkit-swin2sr-artifact-precommit-v1",
        "optimizer_decision": str(args.optimizer_decision),
        "optimizer_decision_sha256": file_sha256(args.optimizer_decision),
        "promotion_policy": str(args.promotion_policy),
        "promotion_policy_sha256": file_sha256(args.promotion_policy),
        "base_onnx": str(args.base_onnx),
        "base_onnx_sha256": file_sha256(args.base_onnx),
        "materialization": str(materialization_path),
        "materialization_sha256": file_sha256(materialization_path),
        "portable_onnx": str(portable_path),
        "source_files": sources,
        "final_heldout_locked": True,
    }
    precommit_path = args.out / "artifact-precommit.json"
    precommit = write_or_verify(precommit_path, stable_precommit)
    if candidate_path.exists():
        candidate = json.loads(candidate_path.read_text())
        validate_existing_candidate(
            candidate,
            decision_path=args.optimizer_decision,
            decision=decision,
            study=study,
            selected=selected,
            policy_path=args.promotion_policy,
            precommit_path=precommit_path,
            precommit=precommit,
        )
        print(json.dumps(candidate, indent=2))
        return
    artifact = build_artifact(
        materialization_path=materialization_path,
        base_onnx_path=args.base_onnx,
        output_path=portable_path,
        scope="all_matrix_and_conv",
    )
    candidate = candidate_record(
        decision_path=args.optimizer_decision,
        decision=decision,
        study=study,
        selected=selected,
        policy_path=args.promotion_policy,
        materialization=materialization,
        artifact=artifact,
        precommit_path=precommit_path,
        precommit=precommit,
    )
    candidate_path.write_text(json.dumps(candidate, indent=2) + "\n")
    print(json.dumps({
        "candidate": str(candidate_path),
        "selected_optimizer": candidate["lane"],
        "runtime_package_bytes": candidate["runtime_package_bytes"],
        "candidate_fraction_of_q4": candidate["artifact_size_gate"][
            "candidate_fraction_of_reference"
        ],
        "weight_bits_per_value_including_scales": candidate[
            "weight_bits_per_value_including_scales"
        ],
        "final_evaluation_eligible": candidate[
            "final_evaluation_eligible"
        ],
        "final_heldout_locked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
