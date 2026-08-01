#!/usr/bin/env python3
"""Evaluate preregistered category-specific scale maps on a sealed screen."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

from analyze_surface_restore_sensitivity import load_sample, state_metrics
from evaluate_scale_codebooks import evidence, metric_rows, q4_evidence
from evaluate_stratified import file_sha256, load_bundle_records, product_pixels, reconstruct
from materialize_surface_states import load_materialized_state
from mixed_scale_codebook import mixed_scale_quantized_dense_state
from scale_codebook import fp16_scale_dense_state
from seal_mixed_scale_map_probe import FORMAT as PROTOCOL_FORMAT
from surface_model_state import pretrained


FORMAT = "backdropkit-swin2sr-mixed-scale-map-result-v1"


def evaluate_state(model, state, samples, teacher_outputs, device):
    model.load_state_dict(state, strict=True)
    return metric_rows(samples, state_metrics(model, samples, teacher_outputs, device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"mixed scale result exists: {args.out}")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise RuntimeError("unsupported mixed scale protocol")
    if protocol.get("model_promotion_authorized") or not protocol.get("final_heldout_locked"):
        raise RuntimeError("mixed scale protocol violates promotion isolation")
    repository = Path(__file__).resolve().parents[2]
    for name, expected in protocol["source_sha256"].items():
        if file_sha256(repository / name) != expected:
            raise RuntimeError(f"sealed mixed-scale source changed: {name}")
    for key in ("prior_protocol", "materialization", "bundle", "q4_reference"):
        if file_sha256(Path(protocol[key])) != protocol[f"{key}_sha256"]:
            raise RuntimeError(f"sealed mixed-scale input changed: {key}")
    torch.set_num_threads(args.torch_threads)
    materialization, runtime_state = load_materialized_state(
        Path(protocol["materialization"]), "runtime_state"
    )
    _manifest, dense_state = load_materialized_state(
        Path(protocol["materialization"]), "dense_state"
    )
    bundle, records = load_bundle_records(Path(protocol["bundle"]))
    if bundle["manifest_sha256"] != protocol["bundle_manifest_sha256"]:
        raise RuntimeError("mixed scale bundle identity changed")
    by_identity = {str(row["sha256"]): row for row in records}
    selected = [by_identity[str(row["sha256"])] for row in protocol["screen"]]
    samples = [load_sample(row, args.device) for row in selected]
    teacher = pretrained(args.device).eval()
    student = pretrained(args.device).eval()
    started = time.perf_counter()
    with torch.inference_mode():
        teacher_outputs = [
            product_pixels(reconstruct(teacher, sample["low"], args.device)[0])
            for sample in samples
        ]
        baseline_state, baseline_storage = fp16_scale_dense_state(
            materialization, runtime_state, dense_state
        )
        baseline_rows = evaluate_state(
            student, baseline_state, samples, teacher_outputs, args.device
        )
        candidates = {}
        rule = {
            "bootstrap_seed": 20260802,
            "bootstrap_samples": 5000,
        }
        for name, mapping in protocol["candidate_maps"].items():
            state, storage = mixed_scale_quantized_dense_state(
                materialization, runtime_state, dense_state, mapping
            )
            rows = evaluate_state(student, state, samples, teacher_outputs, args.device)
            paired = evidence(rows, baseline_rows, rule)
            candidates[name] = {
                "bits_by_category": mapping,
                "storage": storage,
                "candidate_minus_fp16": paired,
                "rows": rows,
            }
            print(json.dumps({
                "candidate": name,
                "trunk_bits": storage["packed_trunk_bits_per_quantized_weight"],
                "ground_truth_delta": paired["paired"]["psnr_vs_ground_truth"]["mean"],
            }), flush=True)
    q4 = json.loads(Path(protocol["q4_reference"]).read_text())
    for candidate in candidates.values():
        candidate["candidate_minus_q4"] = q4_evidence(
            candidate["rows"], q4["samples"], rule
        )
        del candidate["rows"]
    result = {
        "format": FORMAT,
        "selection_source": "development_mechanism_diagnostic",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "protocol": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "screen_sample_count": len(samples),
        "baseline_storage": baseline_storage,
        "candidates": candidates,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
