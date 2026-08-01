#!/usr/bin/env python3
"""Test small full-precision islands against hard-ternary Swin2SR output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Swin2SRForImageSuperResolution

from analyze_layer_sensitivity import load_student, select_balanced
from artifact_state import hard_state_dict
from evaluate_stratified import (
    deterministic_crop,
    edges,
    load_records,
    product_pixels,
    psnr,
    reconstruct,
    summarize,
    to_tensor,
)
from rotation_control import block_hadamard
from train_swin2sr_qat import MODEL_ID, MODEL_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--q4-metrics", type=Path)
    parser.add_argument("--samples-per-source", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def restoration_groups(ranking: list[dict]) -> dict[str, list[str]]:
    names = [row["name"] for row in ranking]
    values = {row["name"]: int(row["weight_values"]) for row in ranking}
    top_budget = []
    used = 0
    budget = sum(values.values()) * 0.05
    for name in names:
        if top_budget and used + values[name] > budget:
            continue
        top_budget.append(name)
        used += values[name]
    return {
        "hard_ternary": [],
        "top1_local_error": names[:1],
        "stage3_values": [
            name
            for name in names
            if "encoder.stages.3." in name and name.endswith(".attention.self.value")
        ],
        "stage3_qkv": [
            name
            for name in names
            if "encoder.stages.3." in name
            and any(name.endswith(f".attention.self.{kind}") for kind in ("query", "key", "value"))
        ],
        "top5pct_local_error": top_budget,
    }


def restored_state(student, quantizers, selected: set[str]) -> dict[str, torch.Tensor]:
    state = hard_state_dict(student, quantizers)
    for name, quantizer, module in quantizers:
        if name not in selected:
            continue
        master = module.parametrizations.weight.original.detach()
        effective = (
            block_hadamard(master, quantizer.rotation_block)
            if quantizer.rotation == "hadamard"
            else master
        )
        state[f"{name}.weight"] = effective.cpu()
        if module.bias is not None:
            folded = torch.mv(master, module.input_shift.detach()) + module.bias.detach()
            state[f"{name}.bias"] = folded.cpu()
    return state


def dense_model(state: dict[str, torch.Tensor], device: str):
    model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def metrics_for_output(
    record: dict,
    output: np.ndarray,
    truth: np.ndarray,
    bicubic: np.ndarray,
    teacher_output: np.ndarray,
    elapsed: float,
) -> dict:
    return {
        "sha256": record.get("sha256"),
        "source": record.get("source", "unknown"),
        "split": record.get("split", "unknown"),
        "language": record.get("language"),
        "script": (
            record["language"].rsplit("_", 1)[-1]
            if record.get("language") and "_" in record["language"]
            else None
        ),
        "psnr_vs_ground_truth": psnr(output, truth),
        "edge_psnr_vs_ground_truth": psnr(
            (edges(output) + 1) / 2,
            (edges(truth) + 1) / 2,
        ),
        "psnr_vs_teacher": psnr(output, teacher_output),
        "bicubic_psnr_vs_ground_truth": psnr(bicubic, truth),
        "teacher_psnr_vs_ground_truth": psnr(teacher_output, truth),
        "teacher_seconds": 0.0,
        "student_seconds": elapsed,
    }


def paired_deltas(candidate: list[dict], reference: dict[str, dict]) -> dict:
    by_source = defaultdict(list)
    overall = []
    teacher = []
    edge = []
    for sample in candidate:
        baseline = reference[sample["sha256"]]
        delta = sample["psnr_vs_ground_truth"] - baseline["psnr_vs_ground_truth"]
        overall.append(delta)
        teacher.append(sample["psnr_vs_teacher"] - baseline["psnr_vs_teacher"])
        edge.append(
            sample["edge_psnr_vs_ground_truth"]
            - baseline["edge_psnr_vs_ground_truth"]
        )
        by_source[sample["source"]].append(delta)
    return {
        "psnr_vs_ground_truth": float(np.mean(overall)),
        "edge_psnr_vs_ground_truth": float(np.mean(edge)),
        "psnr_vs_teacher": float(np.mean(teacher)),
        "by_source_psnr": {
            source: float(np.mean(values))
            for source, values in sorted(by_source.items())
        },
    }


def q4_index(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    return {
        row["sha256"]: row
        for row in json.loads(path.read_text())["samples"]
    }


def evaluate(args: argparse.Namespace) -> dict:
    sensitivity = json.loads(args.sensitivity.read_text())
    groups = restoration_groups(sensitivity["ranking"])
    student, quantizers, payload = load_student(args.checkpoint, "cpu")
    models = {
        label: dense_model(
            restored_state(student, quantizers, set(names)),
            args.device,
        )
        for label, names in groups.items()
    }
    teacher = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(args.device).eval()
    records = select_balanced(
        load_records(args.manifest),
        args.samples_per_source,
        args.seed,
    )
    samples = {label: [] for label in groups}
    for index, record in enumerate(records, 1):
        low_image, high_image = deterministic_crop(record, args.crop_size)
        low = to_tensor(low_image, args.device)
        truth = np.asarray(high_image, dtype=np.float32) / 255.0
        bicubic = np.asarray(
            low_image.resize(
                (args.crop_size, args.crop_size),
                Image.Resampling.BICUBIC,
            ),
            dtype=np.float32,
        ) / 255.0
        teacher_output, _ = reconstruct(teacher, low, args.device)
        teacher_output = product_pixels(teacher_output)
        for label, model in models.items():
            output, elapsed = reconstruct(model, low, args.device)
            samples[label].append(metrics_for_output(
                record,
                product_pixels(output),
                truth,
                bicubic,
                teacher_output,
                elapsed,
            ))
        if index % 10 == 0:
            print(json.dumps({"evaluated": index, "total": len(records)}), flush=True)
    hard_index = {
        row["sha256"]: row
        for row in samples["hard_ternary"]
    }
    q4 = q4_index(args.q4_metrics)
    results = {}
    module_values = {
        row["name"]: int(row["weight_values"])
        for row in sensitivity["ranking"]
    }
    for label, names in groups.items():
        rows = samples[label]
        results[label] = {
            "restored_modules": names,
            "restored_weight_values": sum(module_values[name] for name in names),
            "overall": summarize(rows),
            "paired_hard_ternary": paired_deltas(rows, hard_index),
            "paired_q4": paired_deltas(rows, q4) if q4 else None,
            "samples": rows,
        }
    return {
        "format": "backdropkit-swin2sr-restoration-islands-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "sensitivity": str(args.sensitivity),
        "samples": len(records),
        "groups": results,
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        label: {
            "restored_weight_values": row["restored_weight_values"],
            "paired_hard_ternary": row["paired_hard_ternary"],
            "paired_q4": row["paired_q4"],
        }
        for label, row in report["groups"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
