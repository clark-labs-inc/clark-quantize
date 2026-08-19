#!/usr/bin/env python3
"""Overlay selected trained CQ3 MLP layers into an unpromoted GGUF template.

The input is deliberately a ``.partial`` file.  It already has CQ3_0 slots
for exactly the preregistered tensors.  We authenticate the full v44 training
denominator, select the requested layers, overwrite only those slots, verify
the payload and non-target digests, and rename the file atomically.  This
avoids keeping two 16+ GiB copies on the disk-constrained CPU host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from materialize_qwen38_27b_all_layer_trained_cq3_v46 import (
    _slots_from_gguf,
    atomic_json,
    collect_training_payloads,
    sha256_excluding,
    sha256_file,
    validate_template_scope,
    validate_tensor_slots,
)


def parse_layers(value: str) -> tuple[int, ...]:
    """Parse one canonical, duplicate-free layer list."""
    try:
        layers = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not layers or layers != tuple(sorted(set(layers))):
        raise argparse.ArgumentTypeError("layers must be nonempty, unique, and sorted")
    if layers[0] < 0 or layers[-1] >= 64:
        raise argparse.ArgumentTypeError("layers must be in [0, 63]")
    return layers


def selected_payloads(payloads: list[Any], layers: tuple[int, ...]) -> list[Any]:
    selected = [payload for payload in payloads if payload.layer in set(layers)]
    expected = len(layers) * 3
    if len(selected) != expected:
        raise ValueError(
            f"selected CQ3 denominator changed: observed={len(selected)} expected={expected}"
        )
    if {payload.layer for payload in selected} != set(layers):
        raise ValueError("selected CQ3 layer denominator changed")
    return selected


def overlay_in_place(
    path: Path, ranges: list[tuple[int, int, Any]]
) -> None:
    if path.suffix != ".partial":
        raise ValueError("in-place overlay is restricted to an unpromoted .partial file")
    with path.open("r+b", buffering=0) as target:
        for start, stop, payload in ranges:
            target.seek(start)
            with payload.path.open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    target.write(block)
            if target.tell() != stop:
                raise ValueError(f"CQ3 overlay ended at wrong offset: {payload.tensor}")
        target.flush()
        os.fsync(target.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-partial", required=True, type=Path)
    parser.add_argument("--template-scope-receipt", required=True, type=Path)
    parser.add_argument("--training-receipt", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=parse_layers)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {args.output}")
    if not args.template_partial.is_file():
        raise FileNotFoundError(args.template_partial)
    if args.template_partial.resolve().parent != args.output.resolve().parent:
        raise ValueError("partial and promoted candidate must share a filesystem")

    all_payloads = collect_training_payloads(args.training_receipt, args.artifact_root)
    payloads = selected_payloads(all_payloads, args.layers)
    scope = validate_template_scope(
        args.template_partial, args.template_scope_receipt, payloads
    )

    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise RuntimeError(
            "set PYTHONPATH to the pinned llama.cpp gguf-py directory"
        ) from error

    names = {payload.tensor for payload in payloads}
    reader = GGUFReader(str(args.template_partial), mode="r")
    ranges = validate_tensor_slots(_slots_from_gguf(reader, names), payloads)
    del reader
    byte_ranges = [(start, stop) for start, stop, _ in ranges]
    template_sha256 = sha256_file(args.template_partial)
    template_bytes = args.template_partial.stat().st_size
    template_non_target_sha256 = sha256_excluding(
        args.template_partial, byte_ranges
    )

    overlay_in_place(args.template_partial, ranges)
    output_non_target_sha256 = sha256_excluding(args.template_partial, byte_ranges)
    output_reader = GGUFReader(str(args.template_partial), mode="r")
    output_slots = _slots_from_gguf(output_reader, names)
    validate_tensor_slots(output_slots, payloads)
    output_by_name = {tensor.name: tensor for tensor in output_reader.tensors}
    target_results: dict[str, dict[str, Any]] = {}
    for start, stop, payload in ranges:
        observed = hashlib.sha256(
            memoryview(output_by_name[payload.tensor].data).cast("B")
        ).hexdigest()
        target_results[payload.tensor] = {
            "layer": payload.layer,
            "shape": list(payload.shape),
            "offset": start,
            "bytes": stop - start,
            "expected_sha256": payload.sha256,
            "observed_sha256": observed,
            "match": observed == payload.sha256,
        }
    del output_reader

    candidate_sha256 = sha256_file(args.template_partial)
    checks = {
        "full_v44_training_gate_authenticated": len(all_payloads) == 192,
        "template_scope_valid": scope.get("valid") is True,
        "exact_selected_layer_count": len(args.layers) * 3 == len(target_results),
        "same_complete_file_bytes": args.template_partial.stat().st_size
        == template_bytes,
        "all_non_target_file_bytes_identical": template_non_target_sha256
        == output_non_target_sha256,
        "all_selected_trained_payloads_exact": all(
            result["match"] for result in target_results.values()
        ),
    }
    receipt = {
        "schema": "qwen38_27b_sensitivity_selected_cq3_materialization_v67",
        "model": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "selected_layers": list(args.layers),
        "selected_tensor_count": len(target_results),
        "training_receipt": {
            "path": str(args.training_receipt),
            "sha256": sha256_file(args.training_receipt),
        },
        "template": {
            "pre_overlay_path": str(args.template_partial),
            "bytes": template_bytes,
            "sha256": template_sha256,
            "scope_receipt": str(args.template_scope_receipt),
            "scope_receipt_sha256": sha256_file(args.template_scope_receipt),
        },
        "candidate": {
            "path": str(args.output),
            "bytes": args.template_partial.stat().st_size,
            "sha256": candidate_sha256,
        },
        "non_target_file_bytes_sha256": {
            "template": template_non_target_sha256,
            "candidate": output_non_target_sha256,
        },
        "targets": target_results,
        "checks": checks,
        "valid": all(checks.values()),
        "claim_boundary": (
            "exact sensitivity-selected trained-CQ3 GGUF assembly only; "
            "whole-model capability, throughput, energy, novelty, and "
            "breakthrough remain unmeasured"
        ),
    }
    if not receipt["valid"]:
        atomic_json(receipt, args.out)
        raise SystemExit(2)

    # Write the authenticated receipt first, then publish the already-fsynced
    # model in one same-filesystem rename.  A crash can expose neither a
    # partially written final model nor an unauthenticated payload.
    atomic_json(receipt, args.out)
    os.replace(args.template_partial, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
