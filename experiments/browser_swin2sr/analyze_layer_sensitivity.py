#!/usr/bin/env python3
"""Measure source-balanced local error for each trained ternary projection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Swin2SRForImageSuperResolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate_stratified import deterministic_crop, load_records, to_tensor
from train_swin2sr_qat import (
    MODEL_ID,
    MODEL_REVISION,
    add_ternary_parametrizations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--samples-per-source", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def select_balanced(records: list[dict], per_source: int, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("source", "unknown")].append(record)
    selected = []
    for index, (source, source_records) in enumerate(sorted(grouped.items())):
        if len(source_records) < per_source:
            raise ValueError(
                f"source {source} has {len(source_records)} < {per_source} samples"
            )
        shuffled = source_records.copy()
        random.Random(seed + index).shuffle(shuffled)
        selected.extend(shuffled[:per_source])
    return selected


def summarize(accumulator: dict, module_values: dict[str, int]) -> list[dict]:
    rows = []
    for name, sources in accumulator.items():
        source_rows = {}
        total_error = 0.0
        total_signal = 0.0
        total_values = 0
        for source, values in sources.items():
            relative = values["squared_error"] / max(
                values["squared_signal"],
                1e-20,
            )
            source_rows[source] = {
                "values": values["values"],
                "mse": values["squared_error"] / values["values"],
                "relative_mse": relative,
            }
            total_error += values["squared_error"]
            total_signal += values["squared_signal"]
            total_values += values["values"]
        rows.append({
            "name": name,
            "weight_values": module_values[name],
            "sources": source_rows,
            "mse": total_error / total_values,
            "relative_mse": total_error / max(total_signal, 1e-20),
            "worst_source_relative_mse": max(
                row["relative_mse"]
                for row in source_rows.values()
            ),
        })
    return sorted(
        rows,
        key=lambda row: row["worst_source_relative_mse"],
        reverse=True,
    )


def load_student(path: Path, device: str):
    payload = torch.load(path, map_location=device)
    quantization = payload["quantization"]
    student = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(device).eval()
    quantizers = add_ternary_parametrizations(
        student,
        quantization["mode"],
        int(quantization["group_size"]),
        quantization["method"],
        quantization["rotation"],
        int(quantization["rotation_block"]),
    )
    student.load_state_dict(payload["model"], strict=True)
    for _name, quantizer, _module in quantizers:
        quantizer.set_progress(1.0)
    return student, quantizers, payload


def analyze(args) -> dict:
    student, quantizers, payload = load_student(args.checkpoint, args.device)
    records = select_balanced(
        load_records(args.manifest),
        args.samples_per_source,
        args.seed,
    )
    accumulator = defaultdict(
        lambda: defaultdict(
            lambda: {
                "squared_error": 0.0,
                "squared_signal": 0.0,
                "values": 0,
            }
        )
    )
    module_values = {
        name: module.parametrizations.weight.original.numel()
        for name, _quantizer, module in quantizers
    }
    current_source = [None]
    checked = set()
    handles = []
    for name, _quantizer, module in quantizers:
        def capture(target, inputs, output, layer_name=name):
            values = inputs[0]
            master = target.parametrizations.weight.original
            with torch.no_grad():
                hard_output = F.linear(values, target.weight, target.bias)
                if layer_name not in checked:
                    maximum = (hard_output - output).abs().max().item()
                    if maximum > 1e-5:
                        raise RuntimeError(
                            f"hook input mismatch for {layer_name}: {maximum}"
                        )
                    checked.add(layer_name)
                master_output = F.linear(values, master, target.bias)
                difference = output - master_output
                row = accumulator[layer_name][current_source[0]]
                row["squared_error"] += difference.square().sum().item()
                row["squared_signal"] += master_output.square().sum().item()
                row["values"] += difference.numel()

        handles.append(module.register_forward_hook(capture))
    try:
        for index, record in enumerate(records, 1):
            low, _high = deterministic_crop(record, args.crop_size)
            current_source[0] = record.get("source", "unknown")
            with torch.inference_mode():
                student(to_tensor(low, args.device))
            if index % 10 == 0:
                print(json.dumps({"evaluated": index, "total": len(records)}))
    finally:
        for handle in handles:
            handle.remove()
    ranking = summarize(accumulator, module_values)
    return {
        "format": "backdropkit-swin2sr-layer-sensitivity-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "quantization": payload["quantization"],
        "samples": len(records),
        "samples_per_source": args.samples_per_source,
        "sources": sorted({
            record.get("source", "unknown")
            for record in records
        }),
        "ranking": [
            {"rank": index + 1, **row}
            for index, row in enumerate(ranking)
        ],
    }


def main() -> None:
    args = parse_args()
    report = analyze(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "samples": report["samples"],
        "top10": report["ranking"][:10],
    }, indent=2))


if __name__ == "__main__":
    main()
