#!/usr/bin/env python3
"""Screen and confirm a browser-native residual gain on disjoint dev folds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Swin2SRForImageSuperResolution

from decide_pixel_objective import paired_interval
from evaluate_stratified import (
    MODEL_ID,
    MODEL_REVISION,
    edges,
    file_sha256,
    load_bundle_records,
    load_hard_student,
    product_pixels,
    psnr,
    reconstruct,
    to_tensor,
)
from residual_gain import balanced_folds, residual_gain_output


FORMAT = "backdropkit-swin2sr-residual-gain-study-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-residual-gain-protocol-v1"
METRICS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def metric_row(
    record: dict,
    output: np.ndarray,
    truth: np.ndarray,
    teacher: np.ndarray,
) -> dict:
    return {
        "sha256": record["sha256"],
        "source": record["source"],
        "psnr_vs_ground_truth": psnr(output, truth),
        "edge_psnr_vs_ground_truth": psnr(
            (edges(output) + 1) / 2,
            (edges(truth) + 1) / 2,
        ),
        "psnr_vs_teacher": psnr(output, teacher),
    }


def indexed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        identity = row["sha256"]
        if identity in result:
            raise RuntimeError("duplicate residual-gain identity")
        result[identity] = row
    return result


def paired(candidate: list[dict], baseline: list[dict], rule: dict) -> dict:
    candidate_index = indexed(candidate)
    baseline_index = indexed(baseline)
    if candidate_index.keys() != baseline_index.keys():
        raise RuntimeError("residual-gain reports are not paired")
    ordered = sorted(baseline_index)
    evidence = {}
    for offset, metric in enumerate(METRICS):
        evidence[metric] = paired_interval(
            [
                candidate_index[identity][metric]
                - baseline_index[identity][metric]
                for identity in ordered
            ],
            seed=int(rule["bootstrap_seed"]) + offset,
            samples=int(rule["bootstrap_samples"]),
        )
    by_source = defaultdict(list)
    for identity in ordered:
        source = baseline_index[identity]["source"]
        by_source[source].append(
            candidate_index[identity]["psnr_vs_ground_truth"]
            - baseline_index[identity]["psnr_vs_ground_truth"]
        )
    return {
        "paired": evidence,
        "ground_truth_by_source": {
            source: float(np.mean(values))
            for source, values in sorted(by_source.items())
        },
    }


def gates(evidence: dict, rule: dict) -> dict[str, bool]:
    paired_metrics = evidence["paired"]
    return {
        "ground_truth_positive_lower_95": (
            paired_metrics["psnr_vs_ground_truth"]["lower_95"]
            > float(rule["ground_truth_lower_95_min_db"])
        ),
        "edge_mean_guard": (
            paired_metrics["edge_psnr_vs_ground_truth"]["mean"]
            >= float(rule["edge_mean_min_db"])
        ),
        "teacher_mean_guard": (
            paired_metrics["psnr_vs_teacher"]["mean"]
            >= float(rule["teacher_mean_min_db"])
        ),
        "every_source_guard": (
            min(evidence["ground_truth_by_source"].values())
            >= float(rule["every_source_ground_truth_mean_min_db"])
        ),
    }


def select_gain(screen: dict[str, dict], rule: dict) -> float:
    passing = [
        row for row in screen.values()
        if row["gain"] != 1.0 and all(gates(row["evidence"], rule).values())
    ]
    if not passing:
        return 1.0
    return float(max(
        passing,
        key=lambda row: (
            row["evidence"]["paired"]["psnr_vs_ground_truth"]["mean"],
            -abs(row["gain"] - 1.0),
        ),
    )["gain"])


def q4_evidence(candidate: list[dict], q4: dict[str, dict], rule: dict) -> dict:
    ordered = sorted(row["sha256"] for row in candidate)
    candidate_index = indexed(candidate)
    if any(identity not in q4 for identity in ordered):
        raise RuntimeError("q4 reference misses residual-gain identities")
    return {
        metric: paired_interval(
            [
                candidate_index[identity][metric] - q4[identity][metric]
                for identity in ordered
            ],
            seed=int(rule["bootstrap_seed"]) + 20 + offset,
            samples=int(rule["bootstrap_samples"]),
        )
        for offset, metric in enumerate(METRICS)
    }


def evaluate_fold(
    records: list[dict],
    student,
    teacher,
    gains: list[float],
    crop_size: int,
) -> tuple[list[dict], dict[float, list[dict]]]:
    baseline = []
    candidates = {gain: [] for gain in gains}
    for index, record in enumerate(records, 1):
        with Image.open(record["sealed_low_path"]) as source:
            low_image = source.convert("RGB").copy()
        with Image.open(record["sealed_high_path"]) as source:
            high_image = source.convert("RGB").copy()
        low = to_tensor(low_image, "cpu")
        truth = np.asarray(high_image, dtype=np.float32) / 255.0
        bicubic = np.asarray(
            low_image.resize((crop_size, crop_size), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        teacher_output, _ = reconstruct(teacher, low, "cpu")
        student_output, _ = reconstruct(student, low, "cpu")
        teacher_pixels = product_pixels(teacher_output)
        baseline_pixels = product_pixels(student_output)
        baseline.append(metric_row(
            record, baseline_pixels, truth, teacher_pixels
        ))
        for gain in gains:
            output = product_pixels(residual_gain_output(
                student_output, bicubic, gain
            ))
            candidates[gain].append(metric_row(
                record, output, truth, teacher_pixels
            ))
        if index % 32 == 0:
            print(json.dumps({"evaluated": index, "total": len(records)}), flush=True)
    return baseline, candidates


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"residual-gain result exists: {args.out}")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise RuntimeError("unsupported residual-gain protocol")
    if protocol.get("model_promotion_authorized"):
        raise RuntimeError("protocol improperly authorizes promotion")
    if not protocol.get("final_heldout_locked"):
        raise RuntimeError("protocol did not lock final heldout")
    for name, expected in protocol["source_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise RuntimeError(f"residual-gain source changed: {name}")
    for name in ("state", "bundle", "q4_reference"):
        if file_sha256(Path(protocol[name])) != protocol[f"{name}_sha256"]:
            raise RuntimeError(f"residual-gain input changed: {name}")
    bundle, records = load_bundle_records(Path(protocol["bundle"]))
    if bundle["manifest_sha256"] != protocol["bundle_manifest_sha256"]:
        raise RuntimeError("residual-gain bundle manifest changed")
    screen_records, confirmation_records = balanced_folds(
        records,
        screen_per_source=int(protocol["screen_per_source"]),
        confirmation_per_source=int(protocol["confirmation_per_source"]),
        seed=int(protocol["fold_seed"]),
    )
    torch.set_num_threads(int(protocol["torch_threads"]))
    student = load_hard_student(
        Path(protocol["state"]), "cpu", 60, "optimal_mse", "none", 4
    )
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to("cpu").eval()
    gains = [float(value) for value in protocol["gains"]]
    rule = protocol["decision_rule"]
    screen_baseline, screen_candidates = evaluate_fold(
        screen_records, student, teacher, gains, int(protocol["crop_size"])
    )
    screen = {}
    for gain in gains:
        evidence = paired(screen_candidates[gain], screen_baseline, rule)
        screen[str(gain)] = {
            "gain": gain,
            "evidence": evidence,
            "gates": gates(evidence, rule),
        }
    selected_gain = select_gain(screen, rule)
    confirmation_baseline, confirmation_candidates = evaluate_fold(
        confirmation_records,
        student,
        teacher,
        [selected_gain],
        int(protocol["crop_size"]),
    )
    confirmation_evidence = paired(
        confirmation_candidates[selected_gain], confirmation_baseline, rule
    )
    confirmation_gates = gates(confirmation_evidence, rule)
    q4 = indexed(json.loads(Path(protocol["q4_reference"]).read_text())["samples"])
    selected_q4 = q4_evidence(
        confirmation_candidates[selected_gain], q4, rule
    )
    admitted = selected_gain != 1.0 and all(confirmation_gates.values())
    report = {
        "format": FORMAT,
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "protocol": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "state_sha256": protocol["state_sha256"],
        "screen_sample_count": len(screen_records),
        "confirmation_sample_count": len(confirmation_records),
        "screen": screen,
        "selected_gain": selected_gain,
        "confirmation": {
            "evidence": confirmation_evidence,
            "gates": confirmation_gates,
            "candidate_minus_q4": selected_q4,
        },
        "residual_gain_admitted_to_full_development": admitted,
        "final_heldout_access_authorized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "selected_gain": selected_gain,
        "confirmation_gates": confirmation_gates,
        "candidate_minus_g2": confirmation_evidence["paired"],
        "candidate_minus_q4": selected_q4,
        "admitted": admitted,
    }, indent=2))


if __name__ == "__main__":
    main()
