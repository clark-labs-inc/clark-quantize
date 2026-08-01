#!/usr/bin/env python3
"""Screen and confirm compressed ternary scale codebooks on product imagery."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analyze_surface_restore_sensitivity import load_sample, state_metrics
from decide_pixel_objective import paired_interval
from evaluate_stratified import file_sha256, load_bundle_records, product_pixels, reconstruct
from materialize_surface_states import load_materialized_state
from scale_codebook import fp16_scale_dense_state, scale_quantized_dense_state
from seal_scale_codebook_study import FORMAT as PROTOCOL_FORMAT
from surface_model_state import pretrained


FORMAT = "backdropkit-swin2sr-scale-codebook-study-v1"
METRICS = (
    "psnr_vs_ground_truth",
    "edge_psnr_vs_ground_truth",
    "psnr_vs_teacher",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def indexed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        identity = str(row["sha256"])
        if identity in result:
            raise RuntimeError("duplicate scale-codebook sample identity")
        result[identity] = row
    return result


def evidence(candidate: list[dict], baseline: list[dict], rule: dict) -> dict:
    candidate_index = indexed(candidate)
    baseline_index = indexed(baseline)
    if candidate_index.keys() != baseline_index.keys():
        raise RuntimeError("scale-codebook reports are not paired")
    ordered = sorted(baseline_index)
    paired = {}
    for offset, metric in enumerate(METRICS):
        paired[metric] = paired_interval(
            [
                candidate_index[identity][metric] - baseline_index[identity][metric]
                for identity in ordered
            ],
            seed=int(rule["bootstrap_seed"]) + offset,
            samples=int(rule["bootstrap_samples"]),
        )
    by_source = defaultdict(list)
    for identity in ordered:
        source = str(baseline_index[identity]["source"])
        by_source[source].append(
            candidate_index[identity]["psnr_vs_ground_truth"]
            - baseline_index[identity]["psnr_vs_ground_truth"]
        )
    return {
        "paired": paired,
        "ground_truth_by_source": {
            source: float(np.mean(values))
            for source, values in sorted(by_source.items())
        },
    }


def gates(result: dict, storage: dict, baseline_storage: dict, rule: dict) -> dict[str, bool]:
    paired = result["paired"]
    reduction = 1 - storage["scale_storage_bytes"] / baseline_storage["scale_storage_bytes"]
    return {
        "ground_truth_noninferior": (
            paired["psnr_vs_ground_truth"]["lower_95"]
            >= float(rule["ground_truth_lower_95_min_db"])
        ),
        "edge_mean_guard": (
            paired["edge_psnr_vs_ground_truth"]["mean"]
            >= float(rule["edge_mean_min_db"])
        ),
        "teacher_noninferior": (
            paired["psnr_vs_teacher"]["lower_95"]
            >= float(rule["teacher_lower_95_min_db"])
        ),
        "every_source_guard": (
            min(result["ground_truth_by_source"].values())
            >= float(rule["every_source_ground_truth_mean_min_db"])
        ),
        "scale_storage_reduction": (
            reduction >= float(rule["minimum_scale_storage_reduction_fraction"])
        ),
    }


def metric_rows(samples: list[dict], metrics: dict) -> list[dict]:
    return [
        {
            "sha256": sample["sha256"],
            "source": sample["source"],
            **row,
        }
        for sample, row in zip(samples, metrics["rows"])
    ]


def evaluate_state(
    model,
    state: dict[str, torch.Tensor],
    samples: list[dict],
    teacher_outputs: list[np.ndarray],
    device: str,
) -> list[dict]:
    model.load_state_dict(state, strict=True)
    return metric_rows(samples, state_metrics(model, samples, teacher_outputs, device))


def q4_evidence(rows: list[dict], q4_rows: list[dict], rule: dict) -> dict:
    q4 = indexed(q4_rows)
    candidate = indexed(rows)
    ordered = sorted(candidate)
    if any(identity not in q4 for identity in ordered):
        raise RuntimeError("q4 reference misses scale-codebook identities")
    return {
        metric: paired_interval(
            [candidate[identity][metric] - q4[identity][metric] for identity in ordered],
            seed=int(rule["bootstrap_seed"]) + 20 + offset,
            samples=int(rule["bootstrap_samples"]),
        )
        for offset, metric in enumerate(METRICS)
    }


def select_bits(screen: dict[int, dict]) -> int:
    passing = [
        row for row in screen.values()
        if all(row["gates"].values())
    ]
    if not passing:
        return 16
    return int(min(
        passing,
        key=lambda row: (
            row["storage"]["packed_trunk_bits_per_quantized_weight"],
            row["bits"],
        ),
    )["bits"])


def verify_protocol(protocol: dict, protocol_path: Path, repository: Path) -> None:
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise RuntimeError("unsupported scale-codebook protocol")
    if protocol.get("model_promotion_authorized") or not protocol.get("final_heldout_locked"):
        raise RuntimeError("scale-codebook protocol violated promotion isolation")
    for name, expected in protocol["source_sha256"].items():
        if file_sha256(repository / name) != expected:
            raise RuntimeError(f"sealed scale-codebook source changed: {name}")
    for name in ("materialization", "bundle", "q4_reference", "excluded_prior_protocol"):
        if file_sha256(Path(protocol[name])) != protocol[f"{name}_sha256"]:
            raise RuntimeError(f"sealed scale-codebook input changed: {name}")
    if file_sha256(protocol_path) == "":
        raise RuntimeError("unreachable protocol identity")


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"scale-codebook result exists: {args.out}")
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    repository = Path(__file__).resolve().parents[2]
    protocol = json.loads(args.protocol.read_text())
    verify_protocol(protocol, args.protocol, repository)
    torch.set_num_threads(args.torch_threads)
    materialization, runtime_state = load_materialized_state(
        Path(protocol["materialization"]), "runtime_state"
    )
    _dense_manifest, dense_state = load_materialized_state(
        Path(protocol["materialization"]), "dense_state"
    )
    bundle, records = load_bundle_records(Path(protocol["bundle"]))
    if bundle["manifest_sha256"] != protocol["bundle_manifest_sha256"]:
        raise RuntimeError("scale-codebook bundle manifest changed")
    record_index = {str(row["sha256"]): row for row in records}
    screen_records = [record_index[row["sha256"]] for row in protocol["screen"]]
    confirmation_records = [record_index[row["sha256"]] for row in protocol["confirmation"]]
    screen_samples = [load_sample(row, args.device) for row in screen_records]
    confirmation_samples = [load_sample(row, args.device) for row in confirmation_records]
    teacher = pretrained(args.device).eval()
    student = pretrained(args.device).eval()
    started = time.perf_counter()
    with torch.inference_mode():
        screen_teacher = [
            product_pixels(reconstruct(teacher, sample["low"], args.device)[0])
            for sample in screen_samples
        ]
        baseline_state, baseline_storage = fp16_scale_dense_state(
            materialization, runtime_state, dense_state
        )
        baseline_rows = evaluate_state(
            student, baseline_state, screen_samples, screen_teacher, args.device
        )
        screen = {}
        for bits in protocol["candidate_scale_index_bits"]:
            state, storage = scale_quantized_dense_state(
                materialization, runtime_state, dense_state, int(bits)
            )
            rows = evaluate_state(student, state, screen_samples, screen_teacher, args.device)
            paired = evidence(rows, baseline_rows, protocol["decision_rule"])
            screen[int(bits)] = {
                "bits": int(bits),
                "storage": storage,
                "evidence": paired,
                "gates": gates(
                    paired, storage, baseline_storage, protocol["decision_rule"]
                ),
            }
            print(json.dumps({
                "screened_bits": bits,
                "gates": screen[int(bits)]["gates"],
            }), flush=True)
        selected_bits = select_bits(screen)
        confirmation = None
        selected_rows = []
        selected_storage = baseline_storage
        if selected_bits != 16:
            confirmation_teacher = [
                product_pixels(reconstruct(teacher, sample["low"], args.device)[0])
                for sample in confirmation_samples
            ]
            confirmation_baseline = evaluate_state(
                student,
                baseline_state,
                confirmation_samples,
                confirmation_teacher,
                args.device,
            )
            selected_state, selected_storage = scale_quantized_dense_state(
                materialization, runtime_state, dense_state, selected_bits
            )
            selected_rows = evaluate_state(
                student,
                selected_state,
                confirmation_samples,
                confirmation_teacher,
                args.device,
            )
            paired = evidence(
                selected_rows, confirmation_baseline, protocol["decision_rule"]
            )
            confirmation = {
                "evidence": paired,
                "gates": gates(
                    paired,
                    selected_storage,
                    baseline_storage,
                    protocol["decision_rule"],
                ),
            }
    q4 = json.loads(Path(protocol["q4_reference"]).read_text())
    admitted = bool(
        confirmation
        and all(confirmation["gates"].values())
    )
    if confirmation:
        confirmation["candidate_minus_q4"] = q4_evidence(
            selected_rows, q4["samples"], protocol["decision_rule"]
        )
    report = {
        "format": FORMAT,
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "final_heldout_access_authorized": False,
        "protocol": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "screen_sample_count": len(screen_samples),
        "confirmation_sample_count": len(confirmation_samples),
        "baseline_storage": baseline_storage,
        "screen": {str(key): value for key, value in sorted(screen.items())},
        "selected_scale_index_bits": selected_bits,
        "selected_storage": selected_storage,
        "confirmation": confirmation,
        "scale_codec_admitted_to_runtime_integration": admitted,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "selected_scale_index_bits": selected_bits,
        "selected_packed_trunk_bits": selected_storage[
            "packed_trunk_bits_per_quantized_weight"
        ],
        "confirmation_gates": confirmation["gates"] if confirmation else None,
        "admitted": admitted,
    }, indent=2))


if __name__ == "__main__":
    main()
