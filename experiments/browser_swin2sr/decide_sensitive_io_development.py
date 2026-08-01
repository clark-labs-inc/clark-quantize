#!/usr/bin/env python3
"""Select a sensitive-I/O recipe from full development-bundle evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from full_surface_provenance import file_sha256


METRICS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
)
TARGET_SURFACE = "swin2sr.first_convolution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-decision", type=Path, required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--q4-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--source-margin-db", type=float, default=0.10)
    parser.add_argument("--mean-margin-db", type=float, default=0.10)
    return parser.parse_args()


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("format") != "backdropkit-swin2sr-hard-started-stratified-v1":
        raise RuntimeError(f"unsupported development report: {path}")
    if not report.get("diagnostic_only"):
        raise RuntimeError(f"report is not diagnostic-only: {path}")
    if report.get("selection_source") != "development":
        raise RuntimeError(f"report is not development evidence: {path}")
    if not report.get("complete_bundle"):
        raise RuntimeError(f"report does not cover its full bundle: {path}")
    if report.get("model_promotion_authorized"):
        raise RuntimeError(f"report improperly authorizes promotion: {path}")
    return report


def indexed_samples(report: dict) -> dict[str, dict]:
    samples = {sample["sha256"]: sample for sample in report["samples"]}
    if None in samples or len(samples) != len(report["samples"]):
        raise RuntimeError("development report has missing or duplicate samples")
    return samples


def expected_candidate_arms(decision: dict) -> list[dict]:
    eligible = [
        arm
        for arm in decision["arms"]
        if arm.get("checkpoint_eligible")
    ]
    return sorted(
        eligible,
        key=lambda arm: (
            -float(arm["hard_ground_truth_db"]),
            float(arm["convolution_learning_rate_multiplier"]),
        ),
    )[:2]


def bootstrap_mean_ci(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    means = np.empty(samples, dtype=np.float64)
    chunk_size = 256
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        indices = rng.integers(
            0,
            len(values),
            size=(count, len(values)),
        )
        means[start:start + count] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return [float(lower), float(upper)]


def paired_analysis(
    candidate: dict,
    reference: dict,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    candidate_samples = indexed_samples(candidate)
    reference_samples = indexed_samples(reference)
    if candidate_samples.keys() != reference_samples.keys():
        raise RuntimeError("development reports do not share exact samples")
    deltas = {}
    source_deltas = defaultdict(list)
    for metric in METRICS:
        values = np.asarray([
            float(candidate_samples[key][metric])
            - float(reference_samples[key][metric])
            for key in sorted(candidate_samples)
        ], dtype=np.float64)
        deltas[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "paired_bootstrap_95ci": bootstrap_mean_ci(
                values,
                bootstrap_samples,
                rng,
            ),
            "candidate_win_fraction": float(np.mean(values > 0)),
        }
    for key, sample in candidate_samples.items():
        source = str(sample.get("source", "unknown"))
        source_deltas[source].append(
            float(sample["psnr_vs_ground_truth"])
            - float(reference_samples[key]["psnr_vs_ground_truth"])
        )
    sources = {
        source: {
            "samples": len(values),
            "mean_psnr_delta": float(np.mean(values)),
        }
        for source, values in sorted(source_deltas.items())
    }
    return {
        "samples": len(candidate_samples),
        "metrics": deltas,
        "sources": sources,
        "worst_source_psnr_delta": min(
            row["mean_psnr_delta"]
            for row in sources.values()
        ),
    }


def report_identity(report: dict) -> dict:
    return {
        "device": report["device"],
        "crop_size": int(report["crop_size"]),
        "bundle_sha256": report["bundle_sha256"],
        "bundle_manifest_sha256": report["bundle_manifest_sha256"],
        "q4_reference_sha256": report["q4_reference_sha256"],
        "gate_manifest_sha256": report["gate_manifest_sha256"],
        "surface_step": int(report["surface_step"]),
        "sample_hashes": sorted(indexed_samples(report)),
    }


def decide(
    decision_path: Path,
    gate_path: Path,
    control_path: Path,
    candidate_paths: list[Path],
    q4_path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
    source_margin_db: float,
    mean_margin_db: float,
) -> dict:
    decision = json.loads(decision_path.read_text())
    gate = json.loads(gate_path.read_text())
    if (
        decision.get("format")
        != "backdropkit-swin2sr-sensitive-io-checkpoint-decision-v1"
    ):
        raise RuntimeError("unsupported checkpoint decision")
    if decision.get("final_selection_authorized"):
        raise RuntimeError("checkpoint gate improperly finalized selection")
    if (
        gate.get("format")
        != "backdropkit-swin2sr-sensitive-io-development-gate-v1"
    ):
        raise RuntimeError("unsupported development gate")
    if gate.get("selection_source") != "development":
        raise RuntimeError("gate is not development-only")
    if gate["q4_reference_sha256"] != file_sha256(q4_path):
        raise RuntimeError("gate q4 reference differs")
    frozen_controls = {
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "source_margin_db": source_margin_db,
        "mean_margin_db": mean_margin_db,
    }
    for name, value in frozen_controls.items():
        if gate[name] != value:
            raise RuntimeError(f"selector differs from sealed {name}")
    expected = expected_candidate_arms(decision)
    expected_by_multiplier = {
        float(arm["convolution_learning_rate_multiplier"]): arm
        for arm in expected
    }
    if not expected_by_multiplier:
        raise RuntimeError("checkpoint gate produced no eligible candidates")
    q4 = json.loads(q4_path.read_text())
    control = load_report(control_path)
    identity = report_identity(control)
    gate_hash = file_sha256(gate_path)
    if control["gate_manifest_sha256"] != gate_hash:
        raise RuntimeError("control was not evaluated under the sealed gate")
    if control.get("surface_group_sizes") not in (
        {},
        {TARGET_SURFACE: 9},
    ):
        raise RuntimeError("development control is not group 9")
    if (
        control["state_sha256"]
        != decision["control"]["checkpoint_sha256"]
    ):
        raise RuntimeError("development control checkpoint differs from gate")
    if control["q4_reference_sha256"] != file_sha256(q4_path):
        raise RuntimeError("q4 reference hash differs from evaluation reports")
    if indexed_samples(control).keys() != indexed_samples(q4).keys():
        raise RuntimeError("control and q4 reference do not share exact samples")
    shared_parent = decision["shared_fork_parent_sha256"]
    control_parent = (
        control.get("fork_provenance") or {}
    ).get("parent_checkpoint_sha256")
    if control_parent != shared_parent:
        raise RuntimeError("control does not share the sealed fork parent")

    loaded_candidates = []
    seen_multipliers = set()
    rng = np.random.default_rng(seed)
    for path in candidate_paths:
        report = load_report(path)
        if report_identity(report) != identity:
            raise RuntimeError(f"candidate protocol differs from control: {path}")
        if report.get("surface_group_sizes") != {TARGET_SURFACE: 2}:
            raise RuntimeError(f"candidate is not group 2: {path}")
        parent = (
            report.get("fork_provenance") or {}
        ).get("parent_checkpoint_sha256")
        if parent != shared_parent:
            raise RuntimeError(f"candidate fork parent differs: {path}")
        multiplier = float(report["convolution_learning_rate_multiplier"])
        expected_arm = expected_by_multiplier.get(multiplier)
        if expected_arm is None:
            raise RuntimeError(f"candidate was not in the frozen top two: {path}")
        if multiplier in seen_multipliers:
            raise RuntimeError(f"duplicate candidate multiplier: {multiplier}")
        seen_multipliers.add(multiplier)
        if report["state_sha256"] != expected_arm["checkpoint_sha256"]:
            raise RuntimeError(f"candidate checkpoint differs from gate: {path}")
        versus_control = paired_analysis(
            report,
            control,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        versus_q4 = paired_analysis(
            report,
            q4,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        loaded_candidates.append({
            "_report": report,
            "report": str(path),
            "report_sha256": file_sha256(path),
            "checkpoint": report["state"],
            "checkpoint_sha256": report["state_sha256"],
            "convolution_learning_rate_multiplier": multiplier,
            "versus_control": versus_control,
            "versus_q4": versus_q4,
            "development_ground_truth_db": float(
                report["overall"]["psnr_vs_ground_truth"]["mean"]
            ),
        })
    if seen_multipliers != set(expected_by_multiplier):
        raise RuntimeError("development reports do not cover the frozen top two")

    best = max(
        loaded_candidates,
        key=lambda arm: arm["development_ground_truth_db"],
    )
    for arm in loaded_candidates:
        versus_best = paired_analysis(
            arm["_report"],
            best["_report"],
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        gates = {
            "overall_within_best_margin": (
                arm["development_ground_truth_db"]
                >= best["development_ground_truth_db"] - mean_margin_db
            ),
            "every_source_within_best_margin": (
                versus_best["worst_source_psnr_delta"]
                >= -source_margin_db
            ),
        }
        arm["versus_best_development_arm"] = versus_best
        arm["gates"] = gates
        arm["eligible"] = all(gates.values())
    eligible = [arm for arm in loaded_candidates if arm["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda arm: (
            arm["convolution_learning_rate_multiplier"],
            -arm["development_ground_truth_db"],
        ),
    )
    selected = ranked[0] if ranked else None
    return {
        "format": "backdropkit-swin2sr-sensitive-io-development-decision-v1",
        "selection_source": "development",
        "checkpoint_decision": str(decision_path),
        "checkpoint_decision_sha256": file_sha256(decision_path),
        "gate_manifest": str(gate_path),
        "gate_manifest_sha256": gate_hash,
        "control_report": str(control_path),
        "control_report_sha256": file_sha256(control_path),
        "q4_reference": str(q4_path),
        "q4_reference_sha256": file_sha256(q4_path),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "source_margin_db": source_margin_db,
        "mean_margin_db": mean_margin_db,
        "best_development_ground_truth_db": (
            best["development_ground_truth_db"]
        ),
        "best_development_multiplier": (
            best["convolution_learning_rate_multiplier"]
        ),
        "candidates": sorted(
            (
                {
                    key: value
                    for key, value in arm.items()
                    if key != "_report"
                }
                for arm in loaded_candidates
            ),
            key=lambda arm: arm["convolution_learning_rate_multiplier"],
        ),
        "selected_convolution_learning_rate_multiplier": (
            selected["convolution_learning_rate_multiplier"]
            if selected
            else None
        ),
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_checkpoint_sha256": (
            selected["checkpoint_sha256"] if selected else None
        ),
        "sensitive_recipe_selected": selected is not None,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "required_next_gate": (
            "Run the preregistered QAT8-versus-AdamW full-surface transfer "
            "study through all-hard development evaluation."
            if selected
            else "No sensitive-I/O recipe passed; revise development training."
        ),
    }


def main() -> None:
    args = parse_args()
    gate = json.loads(args.gate_manifest.read_text())
    repository = Path(__file__).resolve().parents[2]
    for relative, expected in gate.get("source_files", {}).items():
        if file_sha256(repository / relative) != expected:
            raise RuntimeError(f"development-gate source changed: {relative}")
    receipt = decide(
        args.checkpoint_decision,
        args.gate_manifest,
        args.control,
        args.candidates,
        args.q4_reference,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        source_margin_db=args.source_margin_db,
        mean_margin_db=args.mean_margin_db,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "sensitive_recipe_selected": receipt["sensitive_recipe_selected"],
        "selected_convolution_learning_rate_multiplier": (
            receipt["selected_convolution_learning_rate_multiplier"]
        ),
        "model_promotion_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
