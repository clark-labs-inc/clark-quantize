#!/usr/bin/env python3
"""Freeze a base-only whole-model gate after a terminal v45 negative result."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


REPO = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
V45_SCHEMA = "qwen38_27b_all_layer_sensitivity_residual_receipt_v45"
V45_INCIDENT_SCHEMA = (
    "qwen38_27b_all_layer_sensitivity_residual_protocol_incident_v45"
)
V60_SCHEMA = "qwen38_27b_trained_compact_whole_model_gate_preregistration_v60"
V60_SHA256 = "cdf92c6746520e62bb834b463eed0ac16a627cc4c788b70136a5728d21d2930c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_v45_negative(receipt: dict[str, Any]) -> list[str]:
    layers = receipt.get("layers", [])
    gate = receipt.get("gate", {})
    if (
        receipt.get("schema") != V45_SCHEMA
        or receipt.get("model") != MODEL_ID
        or receipt.get("revision") != MODEL_REVISION
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
        or len(layers) != 64
        or [layer.get("layer") for layer in layers] != list(range(64))
        or not all(layer.get("complete") is True for layer in layers)
        or not all(layer.get("terminal") is True for layer in layers)
        or gate.get("required_layers") != 64
        or gate.get("observed_layers") != 64
        or gate.get("all_layers_terminal") is not True
        or gate.get("passed") is not False
    ):
        raise ValueError("v45 is not one complete terminal negative denominator")
    failures = []
    for key in (
        "all_active_layers_calibration_not_regressed",
        "all_active_layers_validation_not_regressed",
        "all_bypass_layers_exact",
        "all_final_validation_nmse_within_maximum",
        "all_pilot_modes_exact",
        "all_active_pilot_payloads_exact",
        "pilot_bypass_layer_has_no_payload",
        "residual_payload_within_maximum",
    ):
        if gate.get(key) is not True:
            failures.append(key)
    if (
        float(gate.get("mean_active_relative_validation_nmse_reduction", 0.0))
        < float(
            gate.get(
                "minimum_mean_active_relative_validation_nmse_reduction",
                float("inf"),
            )
        )
    ):
        failures.append("mean_active_relative_validation_nmse_reduction")
    if not failures:
        raise ValueError("v45 reports failure without a frozen gate failure")
    return failures


def validate_v45_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    if outcome.get("schema") == V45_SCHEMA:
        return {
            "classification": "terminal_negative_receipt",
            "failed_frozen_gates": validate_v45_negative(outcome),
            "repository_commit": outcome["repository_commit"],
            "preregistration_sha256": outcome["preregistration_sha256"],
            "training_receipt_sha256": outcome["training_receipt_sha256"],
        }
    partial = outcome.get("partial_scientific_result", {})
    evidence = outcome.get("evidence", {})
    decision = outcome.get("decision", {})
    if (
        outcome.get("schema") != V45_INCIDENT_SCHEMA
        or outcome.get("model") != MODEL_ID
        or outcome.get("revision") != MODEL_REVISION
        or outcome.get("classification")
        != "terminal_protocol_negative_no_v45_receipt"
        or outcome.get("infrastructure_valid") is not True
        or outcome.get("blocked_layer") != 32
        or outcome.get("complete") is not True
        or outcome.get("terminal") is not True
        or partial.get("terminal_layers") != 32
        or partial.get("active_residual_layers") != 32
        or partial.get("frozen_mean_reduction_gate_already_missed") is not True
        or partial.get("frozen_no_validation_regression_gate_already_missed")
        is not True
        or evidence.get("v45_terminal_receipt_present") is not False
        or evidence.get("v45_completion_marker_present") is not False
        or decision.get("v45_residual_rejected") is not True
        or decision.get("original_v60_residual_candidate_admission_failed")
        is not True
        or decision.get("do_not_materialize_or_execute_v45_payloads") is not True
    ):
        raise ValueError("v45 protocol-negative incident identity changed")
    return {
        "classification": outcome["classification"],
        "failed_frozen_gates": [
            "all_active_layers_validation_not_regressed",
            "mean_active_relative_validation_nmse_reduction",
            "pilot_sensitivity_mode_reproduction_protocol",
        ],
        "repository_commit": None,
        "preregistration_sha256": evidence["preregistration_sha256"],
        "training_receipt_sha256": None,
    }


def validate_v60(preregistration: dict[str, Any]) -> None:
    admission = preregistration.get("candidate_admission", {})
    performance = preregistration.get("performance_protocol", {})
    quality = preregistration.get("quality_gates", {})
    if (
        preregistration.get("schema") != V60_SCHEMA
        or preregistration.get("status")
        != "frozen_before_candidate_materialization_or_whole_model_observation"
        or preregistration.get("scope", {}).get("active_model_only") != MODEL_ID
        or admission.get("v44_all_layer_training_must_pass") is not True
        or admission.get("v45_sensitivity_residual_must_pass") is not True
        or admission.get("v58_native_lut_must_pass_before_v59_is_used") is not True
        or admission.get("v59_llamacpp_integration_must_compile_and_pass_native_parity")
        is not True
        or admission.get("high_precision_weight_materialization_forbidden") is not True
        or quality.get("maximum_candidate_over_q4_wikitext_ppl_ratio") != 1.05
        or performance.get("minimum_candidate_weighted_tokens_per_second") != 2.75
        or performance.get("minimum_candidate_over_q4_weighted_decode_speedup")
        != 1.4
    ):
        raise ValueError("v60 frozen whole-model denominator changed")


def build_amendment(
    *,
    v45_receipt: dict[str, Any],
    v45_receipt_sha256: str,
    v60_preregistration: dict[str, Any],
    generator_sha256: str,
    repository_commit: str,
    frozen_at_utc: str,
) -> dict[str, Any]:
    v45_validation = validate_v45_outcome(v45_receipt)
    validate_v60(v60_preregistration)
    return {
        "schema": "qwen38_27b_trained_compact_base_only_gate_amendment_v61",
        "experiment_id": "qwen38_27b_trained_compact_base_only_gate_v61",
        "goal_id": "qwen38_27b_amd_cpu_breakthrough_v1",
        "status": "frozen_after_v45_negative_before_candidate_materialization_or_whole_model_observation",
        "frozen_at_utc": frozen_at_utc,
        "objective": (
            "Qualify one separate trained CQ3 plus AFF3 Qwen3.8-27B candidate "
            "with the failed v45 post-block residual disabled, under every frozen "
            "v60 capability and conventional-throughput threshold."
        ),
        "scope": {
            "active_model_only": MODEL_ID,
            "revision": MODEL_REVISION,
            "host": "quant",
            "cpu": "AMD EPYC Milan",
            "physical_cores": 4,
            "qwen_4b_compute_forbidden": True,
        },
        "parent_v60": {
            "schema": V60_SCHEMA,
            "sha256": V60_SHA256,
            "all_manifests_prompts_quality_thresholds_and_performance_thresholds_unchanged": True,
        },
        "terminal_v45_outcome": {
            "schema": v45_receipt["schema"],
            "outcome_sha256": v45_receipt_sha256,
            **v45_validation,
            "original_v60_residual_candidate_rejected": True,
        },
        "candidate_identity": {
            "trained_cq3_and_aff3_source": "terminal passed v44 payloads",
            "post_block_residual_payload_bytes": 0,
            "QWEN38_RESIDUAL_DIR_must_be_unset": True,
            "QWEN38_RESIDUAL_LAYERS_must_be_unset": True,
            "v45_payload_materialization_forbidden": True,
            "high_precision_weight_materialization_forbidden": True,
            "v58_native_lut_must_pass_before_v59_is_used": True,
            "v59_llamacpp_integration_must_compile_and_pass_native_parity": True,
        },
        "inherited_frozen_gates": {
            "cute_correct_minimum": 7,
            "math500_correct_minimum": 4,
            "code_correct_minimum": 5,
            "mmlu_correct_minimum": 51,
            "candidate_over_q4_wikitext_ppl_ratio_maximum": 1.05,
            "candidate_weighted_tokens_per_second_minimum": 2.75,
            "candidate_over_q4_weighted_decode_speedup_minimum": 1.4,
            "candidate_weighted_tokens_per_second_stretch": 3.2,
            "free_generation_required": True,
            "resident_memory_and_energy_receipts_required": True,
        },
        "decision_rules": {
            "this_is_a_separate_candidate_not_a_rescue_of_v45": True,
            "failed_v45_residual_must_never_execute": True,
            "no_v60_threshold_or_denominator_changes": True,
            "reject_on_any_inherited_v60_gate_failure": True,
            "optional_mtp_cannot_rescue_conventional_failure": True,
        },
        "frozen_inputs": {
            "generator_sha256": generator_sha256,
            "generator_repository_commit": repository_commit,
            "v60_sha256": V60_SHA256,
            "v45_outcome_sha256": v45_receipt_sha256,
        },
        "claim_boundary": (
            "A pass can qualify only the exact residual-free trained compact "
            "Qwen3.8-27B runtime on the frozen four-core Milan node. It does not "
            "erase the v45 negative, establish 100 tokens per second, or establish novelty."
        ),
    }


def atomic_no_overwrite(payload: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite a frozen v61 amendment")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v45-outcome", "--v45-receipt", dest="v45_outcome", type=Path, required=True
    )
    parser.add_argument(
        "--v60-preregistration",
        type=Path,
        default=RUNTIME
        / "qwen38_27b_trained_compact_whole_model_gate_preregistration_v60.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNTIME
        / "qwen38_27b_trained_compact_base_only_gate_amendment_v61.json",
    )
    args = parser.parse_args()
    if sha256_file(args.v60_preregistration) != V60_SHA256:
        raise ValueError("v60 preregistration bytes changed")
    tracked = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if tracked:
        raise RuntimeError("v61 freeze requires a clean tracked worktree")
    receipt = json.loads(args.v45_outcome.read_text())
    v60 = json.loads(args.v60_preregistration.read_text())
    payload = build_amendment(
        v45_receipt=receipt,
        v45_receipt_sha256=sha256_file(args.v45_outcome),
        v60_preregistration=v60,
        generator_sha256=sha256_file(Path(__file__)),
        repository_commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        frozen_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    atomic_no_overwrite(payload, args.output)
    print(
        json.dumps(
            {
                "event": "v61_base_only_gate_amendment_frozen",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "v45_outcome_sha256": payload["frozen_inputs"]["v45_outcome_sha256"],
                "terminal": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
