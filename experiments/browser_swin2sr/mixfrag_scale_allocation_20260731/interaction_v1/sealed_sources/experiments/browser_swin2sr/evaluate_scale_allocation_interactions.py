#!/usr/bin/env python3
"""Evaluate combined MixFrag scale maps on two training-only shards."""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from calibrate_scale_fragility import (
    output_fragility,
    prepared_inputs,
    reconstruct,
    scale_candidate,
    weighted_quotas,
)
from evaluate_hard_started_checkpoints import load_checkpoint_student
from full_surface_provenance import file_sha256
from recovery_data import PRODUCT_SOURCE_WEIGHTS, Record, load_records
from scale_allocation_interactions import (
    advancement_decision,
    precision_map_storage,
    uniform_map,
)


FORMAT = "backdropkit-swin2sr-scale-allocation-interaction-result-v1"
PROTOCOL_FORMAT = "backdropkit-swin2sr-scale-allocation-interaction-study-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def verify_protocol(protocol: dict) -> None:
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unsupported scale-allocation interaction protocol")
    if (
        protocol.get("model_promotion_authorized")
        or protocol.get("development_access_authorized")
        or protocol.get("final_heldout_access_authorized")
        or not protocol.get("final_heldout_locked")
    ):
        raise RuntimeError("interaction protocol violates evaluation isolation")
    source_root = Path(protocol["sealed_source_root"])
    for relative, expected in protocol["source_sha256"].items():
        if file_sha256(source_root / relative) != expected:
            raise RuntimeError(f"sealed interaction source changed: {relative}")
    for key in (
        "allocation_report",
        "allocation_manifest",
        "allocation_study_manifest",
        "parent_checkpoint",
    ):
        if file_sha256(Path(protocol[key])) != protocol[f"{key}_sha256"]:
            raise RuntimeError(f"sealed interaction input changed: {key}")
    for dataset in protocol["training_data"]:
        for key in ("summary", "manifest"):
            if file_sha256(Path(dataset[key])) != dataset[f"{key}_sha256"]:
                raise RuntimeError(f"training dataset {key} changed")


def report_records(report: dict) -> list[Record]:
    records = []
    for receipt in report["records"]:
        path = Path(receipt["path"])
        if file_sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"allocation record changed: {path}")
        records.append(Record(path, receipt["source"], receipt.get("language")))
    return records


def disjoint_records(
    roots: list[Path],
    count: int,
    seed: int,
    excluded: set[Path],
) -> tuple[list[Record], dict[str, int]]:
    by_source = defaultdict(list)
    for record in load_records(roots):
        if record.path.resolve() not in excluded:
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
            if len(choices) < quota:
                raise ValueError(f"confirmation source is too small: {source}")
            selected.extend(choices[:quota])
    random.Random(seed).shuffle(selected)
    if len(selected) != count:
        raise RuntimeError("confirmation selection has the wrong size")
    if any(record.path.resolve() in excluded for record in selected):
        raise RuntimeError("confirmation selection overlaps allocation records")
    return selected, dict(sorted(quotas.items()))


def precision_maps(protocol: dict, report: dict) -> tuple[dict, dict]:
    maps = {}
    candidates = {}
    for name, specification in protocol["candidates"].items():
        allocation = report["allocations"][str(specification["budget_bytes"])]
        mapping = allocation["choices"]
        storage = precision_map_storage(report, mapping)
        maps[name] = mapping
        candidates[name] = {
            **specification,
            "summed_isolated_fragility": allocation["summed_isolated_fragility"],
            "storage": storage,
        }
    for label in protocol["uniform_controls"]:
        maps[f"uniform_{label}"] = uniform_map(report, label)
    return maps, candidates


def candidate_bank(surfaces, report: dict, maps: dict) -> dict:
    rows = {row["surface"]: row for row in report["surfaces"]}
    surface_names = {surface.name for surface in surfaces}
    if surface_names != set(rows):
        raise RuntimeError("allocation report and checkpoint surfaces differ")
    labels = {
        surface: {mapping[surface] for mapping in maps.values()}
        for surface in surface_names
    }
    bank = {}
    for index, surface in enumerate(surfaces, start=1):
        if abs(float(surface.quantizer.amount.item()) - 1.0) > 1e-7:
            raise RuntimeError("interaction parent is not fully hard")
        hard = surface.module.weight.detach().clone()
        expected = {choice["label"]: choice for choice in rows[surface.name]["choices"]}
        bank[surface.name] = {}
        for label in sorted(labels[surface.name]):
            bit_width = None if label == "fp16" else int(label[1:])
            weight, receipt = scale_candidate(
                hard, int(surface.quantizer.group_size), bit_width
            )
            expected_receipt = expected[label]
            for key in ("label", "storage_bytes", "groups", "centers"):
                if receipt[key] != expected_receipt[key]:
                    raise RuntimeError(
                        f"materialized choice differs for {surface.name}/{label}/{key}"
                    )
            bank[surface.name][label] = weight
        if index % 32 == 0 or index == len(surfaces):
            print(json.dumps({
                "event": "candidate_bank_progress",
                "completed": index,
                "total": len(surfaces),
            }), flush=True)
    return bank


@torch.inference_mode()
def evaluate_mapping(model, surfaces, bank, mapping, low, reference, batch_size):
    originals = []
    try:
        for surface in surfaces:
            master = surface.module.parametrizations.weight.original
            originals.append((surface, master.detach().clone(), float(surface.quantizer.amount)))
            master.copy_(bank[surface.name][mapping[surface.name]])
            surface.quantizer.amount.fill_(0.0)
        candidate = reconstruct(model, low, batch_size)
        return output_fragility(reference, candidate)
    finally:
        for surface, original, amount in originals:
            surface.module.parametrizations.weight.original.copy_(original)
            surface.quantizer.amount.fill_(amount)


def evaluate_shard(
    model, surfaces, bank, maps, records, seed, batch_size, crop_size, device
):
    low, receipts = prepared_inputs(records, crop_size, seed, device)
    with torch.inference_mode():
        reference = reconstruct(model, low, batch_size)
    states = {}
    for name, mapping in maps.items():
        metrics = evaluate_mapping(
            model, surfaces, bank, mapping, low, reference, batch_size
        )
        states[name] = metrics
        print(json.dumps({"event": "state_complete", "state": name, **metrics}), flush=True)
    del low, reference
    if device == "mps":
        torch.mps.empty_cache()
    return {"records": receipts, "states": states}


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"interaction result exists: {args.out}")
    protocol = json.loads(args.protocol.read_text())
    verify_protocol(protocol)
    report = json.loads(Path(protocol["allocation_report"]).read_text())
    if (
        report.get("selection_source") != "training_calibration_only"
        or report.get("development_accessed")
        or report.get("final_heldout_accessed")
    ):
        raise RuntimeError("allocation report is not training-only")
    maps, candidates = precision_maps(protocol, report)
    model, surfaces, parent = load_checkpoint_student(
        Path(protocol["parent_checkpoint"]), args.device
    )
    if int(parent["step"]) != int(report["parent_step"]):
        raise RuntimeError("allocation parent step changed")
    bank = candidate_bank(surfaces, report, maps)
    discovery = report_records(report)
    roots = [Path(dataset["root"]) for dataset in protocol["training_data"]]
    confirmation, quotas = disjoint_records(
        roots,
        int(protocol["confirmation_samples"]),
        int(protocol["confirmation_seed"]),
        {record.path.resolve() for record in discovery},
    )
    started = time.perf_counter()
    shards = {
        "allocation_calibration": evaluate_shard(
            model, surfaces, bank, maps, discovery, int(report["seed"]),
            int(protocol["batch_size"]), int(protocol["crop_size"]), args.device,
        ),
        "disjoint_confirmation": evaluate_shard(
            model, surfaces, bank, maps, confirmation,
            int(protocol["confirmation_seed"]), int(protocol["batch_size"]),
            int(protocol["crop_size"]), args.device,
        ),
    }
    shards["allocation_calibration"]["source_quotas"] = report["source_quotas"]
    shards["disjoint_confirmation"]["source_quotas"] = quotas
    for name, candidate in candidates.items():
        measured = shards["allocation_calibration"]["states"][name]["fragility"]
        candidate["measured_to_summed_isolated_ratio"] = (
            measured / max(candidate["summed_isolated_fragility"], 1e-30)
        )
    decision = advancement_decision(
        candidates,
        shards,
        minimum_relative_gain=float(protocol["minimum_relative_gain"]),
    )
    result = {
        "format": FORMAT,
        "status": "combined_interactions_measured",
        "selection_source": "training_only",
        "model_promotion_authorized": False,
        "development_accessed": False,
        "final_heldout_locked": True,
        "final_heldout_accessed": False,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": file_sha256(args.protocol),
        "parent_checkpoint_sha256": protocol["parent_checkpoint_sha256"],
        "surface_count": len(surfaces),
        "candidates": candidates,
        "shards": shards,
        "decision": decision,
        "storage_erratum": protocol["storage_erratum"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "result": str(args.out),
        "result_sha256": file_sha256(args.out),
        "selected_candidate": decision["selected_candidate"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
