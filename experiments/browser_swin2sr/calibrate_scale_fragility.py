#!/usr/bin/env python3
"""Measure isolated Swin2SR scale fragility and allocate exact browser bytes."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data_pipeline import degrade, load_rgb, to_tensor, ui_crop
from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from mixfrag_scale_allocation import (
    PrecisionChoice,
    aligned_scale_storage_bytes,
    allocate_precision,
)
from recovery_data import PRODUCT_SOURCE_WEIGHTS, load_records
from scale_codebook import group_nonzero_counts, quantize_scale_codebook
from training_data_contract import dataset_fingerprints


FORMAT = "backdropkit-swin2sr-output-fragility-calibration-v1"
DEFAULT_BITS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bit-width", type=int, action="append")
    parser.add_argument("--budget-bytes", type=int, action="append", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def weighted_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total < len(weights) or any(value <= 0 for value in weights.values()):
        raise ValueError("calibration total and source weights are inconsistent")
    denominator = sum(weights.values())
    exact = {name: total * value / denominator for name, value in weights.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda name: (exact[name] - quotas[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def calibration_records(roots: list[Path], count: int, seed: int):
    records = load_records(roots)
    by_source = defaultdict(list)
    for record in records:
        by_source[record.source].append(record)
    weights = (
        PRODUCT_SOURCE_WEIGHTS
        if set(by_source) == set(PRODUCT_SOURCE_WEIGHTS)
        else {source: 1.0 for source in by_source}
    )
    quotas = weighted_quotas(count, weights)
    selected = []
    for source in sorted(by_source):
        quota = quotas[source]
        source_records = by_source[source]
        languages = defaultdict(list)
        for record in source_records:
            if record.language:
                languages[record.language].append(record)
        rng = random.Random(seed + sum(map(ord, source)))
        if languages:
            names = sorted(languages)
            rng.shuffle(names)
            for index in range(quota):
                language = names[index % len(names)]
                choices = sorted(languages[language], key=lambda item: str(item.path))
                selected.append(choices[(index // len(names)) % len(choices)])
        else:
            choices = sorted(source_records, key=lambda item: str(item.path))
            rng.shuffle(choices)
            selected.extend(choices[index % len(choices)] for index in range(quota))
    if len(selected) != count:
        raise RuntimeError("calibration selection has the wrong size")
    random.Random(seed).shuffle(selected)
    return selected, dict(sorted(quotas.items()))


def scale_candidate(weight: torch.Tensor, group_size: int, bit_width: int | None):
    flat = weight.detach().float().cpu().numpy().reshape(-1)
    pad = (-flat.size) % group_size
    padded = np.pad(flat, (0, pad)) if pad else flat
    groups = padded.reshape(-1, group_size)
    scales = np.max(np.abs(groups), axis=1).astype(np.float32)
    safe = np.maximum(scales, np.finfo(np.float32).eps)
    codes = np.rint(groups / safe[:, None]).clip(-1, 1).astype(np.int8)
    codes[scales == 0] = 0
    if bit_width is None:
        decoded = scales.astype(np.float16).astype(np.float32)
        label = "fp16"
        centers = len(scales)
        storage = aligned_scale_storage_bytes(len(scales))
    else:
        codebook = quantize_scale_codebook(
            scales,
            group_nonzero_counts(codes.reshape(-1), group_size),
            bit_width,
        )
        decoded = codebook.decoded
        label = f"q{bit_width}"
        centers = len(codebook.centers)
        storage = aligned_scale_storage_bytes(
            len(scales),
            bit_width=bit_width,
            center_count=centers,
        )
        if storage != codebook.index_bytes + codebook.codebook_bytes + (
            -(codebook.index_bytes + codebook.codebook_bytes)
        ) % 4:
            raise RuntimeError("scale-codebook storage accounting diverged")
    candidate = torch.from_numpy(
        (codes.astype(np.float32) * decoded[:, None]).reshape(-1)[: flat.size]
        .reshape(weight.shape)
    ).to(device=weight.device, dtype=weight.dtype)
    return candidate, {
        "label": label,
        "storage_bytes": storage,
        "groups": len(scales),
        "centers": centers,
    }


def edge_components(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.diff(images, dim=-2), torch.diff(images, dim=-1)


def output_fragility(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    delta = candidate - reference
    output_error = float(torch.sum(delta.square()).item())
    output_energy = float(torch.sum(reference.square()).item())
    reference_edges = edge_components(reference)
    candidate_edges = edge_components(candidate)
    edge_error = sum(
        float(torch.sum((actual - expected).square()).item())
        for actual, expected in zip(candidate_edges, reference_edges)
    )
    edge_energy = sum(float(torch.sum(value.square()).item()) for value in reference_edges)
    normalized_output = output_error / max(output_energy, 1e-30)
    normalized_edge = edge_error / max(edge_energy, 1e-30)
    return {
        "normalized_output_mse": normalized_output,
        "normalized_edge_mse": normalized_edge,
        "fragility": normalized_output + 0.25 * normalized_edge,
    }


@torch.inference_mode()
def reconstruct(model, low: torch.Tensor, batch_size: int) -> torch.Tensor:
    return torch.cat([
        model(batch).reconstruction.clamp(0, 1)
        for batch in low.split(batch_size)
    ])


def prepared_inputs(records, crop_size: int, seed: int, device: str):
    tensors = []
    receipts = []
    for index, record in enumerate(records):
        path_hash = file_sha256(record.path)
        rng = random.Random(seed + index * 1_000_003 + int(path_hash[:8], 16))
        high = ui_crop(load_rgb(record.path), crop_size, rng)
        tensors.append(to_tensor(degrade(high, rng), device))
        receipts.append({
            "source": record.source,
            "language": record.language,
            "path": str(record.path),
            "sha256": path_hash,
        })
    return torch.cat(tensors), receipts


def load_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["surface"]] = row
    return rows


def calibration_manifest(args, parent_payload, records, quotas, bits) -> dict:
    return {
        "format": FORMAT,
        "parent": str(args.parent.resolve()),
        "parent_sha256": file_sha256(args.parent),
        "parent_step": int(parent_payload["step"]),
        "training_data": dataset_fingerprints(args.data),
        "calibration_samples": args.samples,
        "source_quotas": quotas,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "crop_size": int(parent_payload["surface_training"]["crop_size"]),
        "bit_widths": bits,
        "budget_bytes": sorted(set(args.budget_bytes)),
        "device": args.device,
        "fragility": "normalized_output_mse + 0.25 * normalized_edge_mse",
        "records": records,
        "selection_source": "training_calibration_only",
        "development_accessed": False,
        "final_heldout_accessed": False,
    }


def write_or_verify_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("scale-fragility calibration configuration changed")
        return
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def allocations(rows: list[dict], budgets: list[int]) -> dict:
    choice_map = {
        row["surface"]: [
            PrecisionChoice(
                choice["label"],
                int(choice["storage_bytes"]),
                float(choice["fragility"]),
            )
            for choice in row["choices"]
        ]
        for row in rows
    }
    results = {}
    for budget in sorted(set(budgets)):
        allocation = allocate_precision(choice_map, budget)
        results[str(budget)] = {
            "budget_bytes": budget,
            "storage_bytes": allocation.storage_bytes,
            "summed_isolated_fragility": allocation.fragility,
            "choices": {
                name: choice.label
                for name, choice in sorted(allocation.choices.items())
            },
            "choice_counts": {
                label: sum(choice.label == label for choice in allocation.choices.values())
                for label in sorted({choice.label for choice in allocation.choices.values()})
            },
        }
    return results


def main() -> None:
    args = parse_args()
    bits = sorted(set(args.bit_width or DEFAULT_BITS))
    if args.samples < 5 or args.batch_size < 1:
        raise ValueError("calibration sample and batch sizes are invalid")
    if any(bit < 1 or bit > 8 for bit in bits):
        raise ValueError("calibration bit widths must be between one and eight")
    args.out.mkdir(parents=True, exist_ok=True)
    student, surfaces, parent_payload = load_checkpoint_student(args.parent, args.device)
    for surface in surfaces:
        surface.quantizer.amount.fill_(1.0)
    selected, quotas = calibration_records(args.data, args.samples, args.seed)
    low, record_receipts = prepared_inputs(
        selected,
        int(parent_payload["surface_training"]["crop_size"]),
        args.seed,
        args.device,
    )
    manifest = calibration_manifest(
        args,
        parent_payload,
        record_receipts,
        quotas,
        bits,
    )
    manifest_path = args.out / "calibration-manifest.json"
    write_or_verify_manifest(manifest_path, manifest)
    if args.inspect_only:
        storage_totals = defaultdict(int)
        for surface in surfaces:
            hard = surface.module.weight.detach()
            for bit_width in (*bits, None):
                _candidate, receipt = scale_candidate(
                    hard,
                    int(surface.quantizer.group_size),
                    bit_width,
                )
                storage_totals[receipt["label"]] += receipt["storage_bytes"]
        print(json.dumps({
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "surfaces": len(surfaces),
            "records": len(record_receipts),
            "source_quotas": quotas,
            "storage_totals": dict(sorted(storage_totals.items())),
            "budgets": sorted(set(args.budget_bytes)),
        }, indent=2))
        return
    reference = reconstruct(student, low, args.batch_size)
    rows_path = args.out / "fragility.jsonl"
    completed = load_completed(rows_path)
    with rows_path.open("a") as log:
        for index, surface in enumerate(surfaces, start=1):
            if surface.name in completed:
                continue
            module = surface.module
            master = module.parametrizations.weight.original
            original = master.detach().clone()
            amount = float(surface.quantizer.amount.item())
            hard = module.weight.detach().clone()
            choices = []
            try:
                for bit_width in (*bits, None):
                    candidate_weight, storage = scale_candidate(
                        hard,
                        int(surface.quantizer.group_size),
                        bit_width,
                    )
                    with torch.no_grad():
                        surface.quantizer.amount.fill_(0.0)
                        master.copy_(candidate_weight)
                    candidate = reconstruct(student, low, args.batch_size)
                    choices.append({**storage, **output_fragility(reference, candidate)})
            finally:
                with torch.no_grad():
                    master.copy_(original)
                    surface.quantizer.amount.fill_(amount)
            row = {
                "surface": surface.name,
                "category": surface.category,
                "values": master.numel(),
                "group_size": int(surface.quantizer.group_size),
                "choices": choices,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            completed[surface.name] = row
            print(json.dumps({
                "event": "surface_complete",
                "surface": surface.name,
                "completed": index,
                "total": len(surfaces),
            }), flush=True)
    rows = [completed[surface.name] for surface in surfaces]
    report = {
        **manifest,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "surface_count": len(rows),
        "surfaces": rows,
        "allocations": allocations(rows, args.budget_bytes),
    }
    report_path = args.out / "fragility-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "allocations": report["allocations"],
    }, indent=2))


if __name__ == "__main__":
    main()
