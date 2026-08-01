#!/usr/bin/env python3
"""Fail-closed promotion gate for a BackdropKit ternary Swin2SR candidate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runtime-receipts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def indexed_samples(report: dict) -> dict[str, dict]:
    samples = {}
    for sample in report.get("samples", []):
        key = sample.get("sha256")
        if not key:
            raise ValueError("sample is missing sha256")
        if key in samples:
            raise ValueError(f"duplicate sample sha256: {key}")
        samples[key] = sample
    return samples


def paired_gate(
    reference: dict[str, dict],
    candidate: dict[str, dict],
    metric: str,
    config: dict,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    keys = sorted(reference)
    deltas = np.asarray([
        float(candidate[key][metric]) - float(reference[key][metric])
        for key in keys
    ])
    indices = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
    bootstrap_means = deltas[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    margin = float(config["noninferiority_margin"])
    p10_minimum = float(config["p10_delta_minimum"])
    p10 = float(np.quantile(deltas, 0.10))
    passed = bool(lower >= margin and p10 >= p10_minimum)
    return {
        "samples": len(keys),
        "candidate_minus_reference_mean": float(deltas.mean()),
        "candidate_minus_reference_median": float(np.median(deltas)),
        "candidate_minus_reference_p10": p10,
        "paired_bootstrap_95ci": [float(lower), float(upper)],
        "noninferiority_margin": margin,
        "p10_delta_minimum": p10_minimum,
        "improved": bool(lower > 0),
        "passed": passed,
    }


def stratified_gate(
    reference: dict[str, dict],
    candidate: dict[str, dict],
    config: dict,
) -> dict:
    key = config["key"]
    metric = config["metric"]
    minimum_samples = int(config["minimum_samples"])
    minimum = float(config["mean_delta_minimum"])
    groups = defaultdict(list)
    for sample_id, reference_sample in reference.items():
        value = reference_sample.get(key)
        if value is None:
            continue
        groups[str(value)].append(
            float(candidate[sample_id][metric]) - float(reference_sample[metric])
        )
    results = {}
    for name, values in sorted(groups.items()):
        deltas = np.asarray(values)
        mean = float(deltas.mean())
        results[name] = {
            "samples": len(values),
            "mean_delta": mean,
            "passed": bool(len(values) >= minimum_samples and mean >= minimum),
        }
    failed = [name for name, result in results.items() if not result["passed"]]
    return {
        "key": key,
        "metric": metric,
        "minimum_samples": minimum_samples,
        "mean_delta_minimum": minimum,
        "groups": len(results),
        "failed_groups": failed,
        "worst_groups": sorted(
            (
                {"name": name, **result}
                for name, result in results.items()
            ),
            key=lambda result: result["mean_delta"],
        )[:10],
        "passed": bool(results and not failed),
    }


def validate_reference(policy: dict, report: dict, sample_count: int) -> list[str]:
    expected = policy["reference"]
    failures = []
    checks = {
        "model": report.get("model"),
        "revision": report.get("model_revision"),
        "model_sha256": report.get("state_sha256"),
        "artifact_bytes": report.get("artifact_bytes"),
        "bundle_manifest_sha256": report.get("bundle_manifest_sha256"),
        "samples": sample_count,
    }
    for key, actual in checks.items():
        if actual != expected[key]:
            failures.append(f"{key}: expected {expected[key]!r}, got {actual!r}")
    return failures


def validate_candidate_contract(reference: dict, candidate: dict) -> list[str]:
    failures = []
    for key in (
        "bundle_manifest_sha256",
        "teacher_model",
        "teacher_model_revision",
        "crop_size",
        "device",
    ):
        if candidate.get(key) != reference.get(key):
            failures.append(
                f"{key}: reference {reference.get(key)!r}, "
                f"candidate {candidate.get(key)!r}"
            )
    for key in (
        "hard_state_sha256",
        "runtime_artifact_sha256",
        "runtime_artifact_bytes",
        "runtime_artifact_compression",
        "runtime_expanded_artifact_sha256",
        "runtime_expanded_artifact_bytes",
    ):
        if candidate.get(key) is None:
            failures.append(f"{key}: candidate value is missing")
    return failures


def runtime_gate(
    policy: dict,
    path: Path | None,
    candidate_sha256: str,
    candidate_bytes: int | None,
    candidate_compression: str | None,
    expanded_sha256: str | None,
    expanded_bytes: int | None,
    hard_state_sha256: str | None,
) -> dict:
    required = policy["required_runtime_surfaces"]
    limits = policy.get("runtime_limits", {})
    expected_shape = policy.get("runtime_input_shape")
    expected_harness = policy.get("runtime_harness_sha256")
    expected_compression = policy.get("runtime_artifact_compression")
    fixture_contract = policy.get("runtime_fixture_contract")
    if fixture_contract != {
        "fixture_sha256_required": True,
        "canonical_hard_state_sha256_required": True,
    }:
        raise ValueError("runtime fixture identity contract is not configured")
    if path is None:
        return {
            "required_surfaces": required,
            "missing_surfaces": required,
            "passed": False,
            "reason": "runtime receipts were not supplied",
        }
    receipts = json.loads(path.read_text())
    if receipts.get("model_sha256") != candidate_sha256:
        return {
            "required_surfaces": required,
            "missing_surfaces": required,
            "passed": False,
            "reason": "runtime receipts reference a different candidate artifact",
        }
    by_surface = {
        receipt.get("surface"): receipt
        for receipt in receipts.get("receipts", [])
        if receipt.get("surface")
    }
    fixture_hashes = {
        receipt.get("fixtureSha256")
        for surface, receipt in by_surface.items()
        if surface in required and receipt.get("fixtureSha256")
    }
    missing = [surface for surface in required if surface not in by_surface]
    violations = {}
    for surface in required:
        receipt = by_surface.get(surface)
        if receipt is None:
            continue
        reasons = []
        expected_browser, expected_backend = surface.rsplit("-", 1)
        limit = limits.get(surface)
        if limit is None:
            reasons.append("runtime limit is not configured")
        if receipt.get("browser") != expected_browser:
            reasons.append(
                f"browser {receipt.get('browser')!r} != {expected_browser!r}"
            )
        if receipt.get("backend") != expected_backend:
            reasons.append(
                f"backend {receipt.get('backend')!r} != {expected_backend!r}"
            )
        if receipt.get("modelSha256") != candidate_sha256:
            reasons.append("per-surface model hash mismatch")
        if receipt.get("modelBytes") != candidate_bytes:
            reasons.append(
                f"model bytes {receipt.get('modelBytes')!r} "
                f"!= {candidate_bytes!r}"
            )
        if receipt.get("modelCompression") != candidate_compression:
            reasons.append(
                f"model compression {receipt.get('modelCompression')!r} "
                f"!= {candidate_compression!r}"
            )
        if candidate_compression != expected_compression:
            reasons.append(
                f"candidate compression {candidate_compression!r} "
                f"!= policy {expected_compression!r}"
            )
        if receipt.get("expandedModelSha256") != expanded_sha256:
            reasons.append("expanded model hash mismatch")
        if receipt.get("expandedModelBytes") != expanded_bytes:
            reasons.append(
                f"expanded model bytes {receipt.get('expandedModelBytes')!r} "
                f"!= {expanded_bytes!r}"
            )
        if not receipt.get("fixtureSha256"):
            reasons.append("fixture hash is missing")
        if receipt.get("fixtureHardStateSha256") != hard_state_sha256:
            reasons.append("fixture hard-state hash mismatch")
        if not isinstance(expected_harness, dict):
            reasons.append("runtime harness hashes are not configured")
        elif receipt.get("harnessFiles") != expected_harness:
            reasons.append("runtime harness hash mismatch")
        if not receipt.get("passed"):
            reasons.append("numerical conformance failed")
        if expected_shape is not None and receipt.get("inputShape") != expected_shape:
            reasons.append(
                f"input shape {receipt.get('inputShape')!r} != {expected_shape!r}"
            )
        session_ms = receipt.get("sessionMs")
        warm_ms = receipt.get("warmInferenceMs")
        if limit is not None:
            if not isinstance(session_ms, (int, float)):
                reasons.append("session time is missing")
            elif session_ms > float(limit["max_session_ms"]):
                reasons.append(
                    f"session {session_ms}ms > {limit['max_session_ms']}ms"
                )
            if not isinstance(warm_ms, (int, float)):
                reasons.append("warm inference time is missing")
            elif warm_ms > float(limit["max_warm_inference_ms"]):
                reasons.append(
                    "warm inference "
                    f"{warm_ms}ms > {limit['max_warm_inference_ms']}ms"
                )
        if reasons:
            violations[surface] = reasons
    if len(fixture_hashes) != 1:
        violations["shared-fixture"] = [
            "required runtime surfaces do not share one fixture hash"
        ]
    failed = sorted(violations)
    return {
        "required_surfaces": required,
        "missing_surfaces": missing,
        "failed_surfaces": failed,
        "violations": violations,
        "passed": not missing and not failed,
    }


def evaluate(policy: dict, reference_report: dict, candidate_report: dict, runtime_path=None) -> dict:
    reference = indexed_samples(reference_report)
    candidate = indexed_samples(candidate_report)
    if reference.keys() != candidate.keys():
        raise ValueError(
            "reference and candidate must contain identical sample hashes; "
            f"reference={len(reference)}, candidate={len(candidate)}"
        )
    reference_failures = validate_reference(policy, reference_report, len(reference))
    candidate_failures = validate_candidate_contract(
        reference_report,
        candidate_report,
    )
    rng = np.random.default_rng(int(policy["seed"]))
    paired = {
        metric: paired_gate(
            reference,
            candidate,
            metric,
            config,
            int(policy["bootstrap_samples"]),
            rng,
        )
        for metric, config in policy["paired_quality_gates"].items()
    }
    stratified = [
        stratified_gate(reference, candidate, config)
        for config in policy["stratified_gates"]
    ]
    reference_bytes = int(reference_report["artifact_bytes"])
    candidate_bytes = candidate_report.get("runtime_artifact_bytes")
    artifact_config = policy.get("artifact_gate")
    if not isinstance(artifact_config, dict):
        raise ValueError("artifact_gate must be configured")
    maximum_fraction = artifact_config.get("maximum_fraction_of_reference")
    if (
        not isinstance(maximum_fraction, (int, float))
        or not 0 < float(maximum_fraction) < 1
    ):
        raise ValueError(
            "artifact_gate.maximum_fraction_of_reference must be between 0 and 1"
        )
    candidate_fraction = (
        candidate_bytes / reference_bytes
        if isinstance(candidate_bytes, int)
        else None
    )
    artifact_passed = (
        isinstance(candidate_bytes, int)
        and candidate_bytes > 0
        and candidate_fraction <= float(maximum_fraction)
    )
    artifact = {
        "reference_bytes": reference_bytes,
        "candidate_bytes": candidate_bytes,
        "candidate_fraction_of_reference": candidate_fraction,
        "maximum_fraction_of_reference": float(maximum_fraction),
        "bytes_saved": (
            reference_bytes - candidate_bytes
            if isinstance(candidate_bytes, int)
            else None
        ),
        "passed": artifact_passed,
    }
    runtime = runtime_gate(
        policy,
        runtime_path,
        candidate_report.get("runtime_artifact_sha256", ""),
        candidate_report.get("runtime_artifact_bytes"),
        candidate_report.get("runtime_artifact_compression"),
        candidate_report.get("runtime_expanded_artifact_sha256"),
        candidate_report.get("runtime_expanded_artifact_bytes"),
        candidate_report.get("hard_state_sha256"),
    )
    passed = bool(
        not reference_failures
        and not candidate_failures
        and all(result["passed"] for result in paired.values())
        and all(result["passed"] for result in stratified)
        and artifact["passed"]
        and runtime["passed"]
    )
    return {
        "format": "backdropkit-swin2sr-promotion-result-v1",
        "reference_failures": reference_failures,
        "candidate_failures": candidate_failures,
        "paired_quality": paired,
        "stratified_quality": stratified,
        "artifact": artifact,
        "runtime": runtime,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    policy = json.loads(args.policy.read_text())
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    result = evaluate(
        policy,
        reference,
        candidate,
        args.runtime_receipts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
