"""Checkpoint-bound capability gates for staged global Qwen QAT."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


GATE_REQUEST_SCHEMA = "clark_qwen35_qat_capability_gate_request_v1"
AUTHORIZATION_SCHEMA = "clark_qwen35_qat_continuation_authorization_v1"
ALLOWED_GATE_B_TASKS = {"mmlu", "gsm8k"}


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_sealed_json(
    value: Mapping[str, Any], *, digest_field: str = "report_content_sha256"
) -> None:
    claimed = value.get(digest_field)
    body = {key: item for key, item in value.items() if key != digest_field}
    if not isinstance(claimed, str) or canonical_digest(body) != claimed:
        raise ValueError(f"sealed JSON hash mismatch for {digest_field}")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def make_gate_request(
    *,
    step: int,
    tokens: int,
    checkpoint_path: Path,
    checkpoint_state_sha256: str,
    checkpoint_manifest_sha256: str,
) -> dict[str, Any]:
    request = {
        "schema": GATE_REQUEST_SCHEMA,
        "step": int(step),
        "tokens": int(tokens),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "policy": {
            "purpose": "catastrophic-forgetting gate before a longer QAT stage",
            "minimum_control_correct_retention": 0.60,
            "allowed_quick_tasks": sorted(ALLOWED_GATE_B_TASKS),
            "full_promotion_gate": "separate and stricter",
        },
        "status": "capability_evaluation_required",
    }
    request["report_content_sha256"] = canonical_digest(request)
    return request


def _task_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("Gate B report results are missing")
    mapped = {result.get("task"): result for result in results}
    if None in mapped or len(mapped) != len(results):
        raise ValueError("Gate B report task names are missing or duplicated")
    return mapped


def authorize_gate_b_continuation(
    request: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_retention: float = 0.60,
) -> dict[str, Any]:
    if not 0.0 < minimum_retention <= 1.0:
        raise ValueError("minimum retention must be in (0, 1]")
    verify_sealed_json(request)
    verify_sealed_json(baseline)
    verify_sealed_json(candidate)
    if request.get("schema") != GATE_REQUEST_SCHEMA:
        raise ValueError("unsupported QAT capability gate request")
    if baseline.get("arm") != "bf16" or candidate.get("arm") != "qat_ternary":
        raise ValueError("continuation requires bf16 control and qat_ternary candidate")
    for field in ("id", "revision"):
        if baseline.get("model", {}).get(field) != candidate.get("model", {}).get(field):
            raise ValueError(f"Gate B model {field} differs")
    if (
        baseline.get("manifest", {}).get("content_sha256")
        != candidate.get("manifest", {}).get("content_sha256")
    ):
        raise ValueError("Gate B reports use different sealed sample manifests")
    candidate_checkpoint = candidate.get("checkpoint")
    if not isinstance(candidate_checkpoint, Mapping):
        raise ValueError("candidate Gate B report has no checkpoint provenance")
    candidate_state = candidate_checkpoint.get("state_sha256")
    if candidate_state != request.get("checkpoint_state_sha256"):
        raise ValueError("candidate report is not bound to the requested checkpoint")
    if candidate_checkpoint.get("manifest_sha256") != request.get("checkpoint_manifest_sha256"):
        raise ValueError("candidate report checkpoint manifest does not match the request")

    baseline_tasks = _task_map(baseline)
    candidate_tasks = _task_map(candidate)
    if baseline_tasks.keys() != candidate_tasks.keys():
        raise ValueError("Gate B reports contain different task sets")
    if not baseline_tasks or not set(baseline_tasks) <= ALLOWED_GATE_B_TASKS:
        raise ValueError("continuation receipt contains no supported quick task")

    gates = []
    for task in sorted(baseline_tasks):
        control = baseline_tasks[task]
        treatment = candidate_tasks[task]
        if control.get("n") != treatment.get("n") or not control.get("n"):
            raise ValueError(f"{task}: denominator mismatch or empty task")
        control_samples = control.get("samples", [])
        candidate_samples = treatment.get("samples", [])
        if len(control_samples) != control["n"] or len(candidate_samples) != treatment["n"]:
            raise ValueError(f"{task}: sample count does not match denominator")
        control_ids = [row.get("sample_id") for row in control_samples]
        candidate_ids = [row.get("sample_id") for row in candidate_samples]
        if (
            None in control_ids
            or len(set(control_ids)) != len(control_ids)
            or set(control_ids) != set(candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
        ):
            raise ValueError(f"{task}: paired sample ids differ or are duplicated")
        control_correct = sum(bool(row.get("correct")) for row in control_samples)
        candidate_correct = sum(bool(row.get("correct")) for row in candidate_samples)
        if control_correct <= 0:
            passed = False
            status = "invalid_control"
            retained = None
        else:
            retained = candidate_correct / control_correct
            passed = retained >= minimum_retention
            status = "pass" if passed else "catastrophic_forgetting"
        gates.append(
            {
                "task": task,
                "status": status,
                "pass": passed,
                "n": control["n"],
                "control_correct": control_correct,
                "candidate_correct": candidate_correct,
                "retained_control_correct_fraction": retained,
                "minimum_retention": minimum_retention,
            }
        )

    passed = all(gate["pass"] for gate in gates)
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "gate_request_content_sha256": request["report_content_sha256"],
        "checkpoint_state_sha256": request["checkpoint_state_sha256"],
        "model": dict(candidate["model"]),
        "manifest_content_sha256": candidate["manifest"]["content_sha256"],
        "baseline_report_content_sha256": baseline["report_content_sha256"],
        "candidate_report_content_sha256": candidate["report_content_sha256"],
        "minimum_retention": minimum_retention,
        "gates": gates,
        "pass": passed,
        "decision": "allow_next_qat_stage" if passed else "stop_for_capability_failure",
    }
    authorization["report_content_sha256"] = canonical_digest(authorization)
    return authorization


def validate_continuation_authorization(
    authorization: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    verify_sealed_json(request)
    verify_sealed_json(authorization)
    if authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported QAT continuation authorization")
    if authorization.get("pass") is not True or authorization.get("decision") != "allow_next_qat_stage":
        raise ValueError("QAT capability gate did not pass")
    if authorization.get("gate_request_content_sha256") != request.get("report_content_sha256"):
        raise ValueError("authorization is for a different gate request")
    if authorization.get("checkpoint_state_sha256") != request.get("checkpoint_state_sha256"):
        raise ValueError("authorization is for a different checkpoint")
