#!/usr/bin/env python3
"""Overlay a frozen tensor-NMSE selection of trained CQ3 payloads.

This experiment deliberately breaks the layer-triplet coupling used by v67
and v68.  The complete v44 training receipt is authenticated first, then each
MLP tensor is admitted independently iff its calibration-only selected NMSE is
at or below a preregistered threshold.  Only exact trained CQ3 bytes are copied
into the corresponding structural slots; every other GGUF byte must remain
identical to the Q4-derived template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


ROLES = ("ffn_down", "ffn_gate", "ffn_up")


def parse_threshold(value: str) -> float:
    """Parse a finite, nonnegative calibration-NMSE threshold."""
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold must be a float") from error
    if not math.isfinite(threshold) or threshold < 0:
        raise argparse.ArgumentTypeError("threshold must be finite and nonnegative")
    return threshold


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_nmse_ranking(training_receipt: Path) -> list[dict[str, Any]]:
    """Return the frozen 192-tensor calibration ranking from the v44 receipt."""
    receipt = json.loads(training_receipt.read_text())
    layers = receipt.get("layers", [])
    if len(layers) != 64 or [item.get("layer") for item in layers] != list(range(64)):
        raise ValueError("v44 tensor ranking layer denominator changed")

    ranking: list[dict[str, Any]] = []
    seen: set[str] = set()
    for layer in layers:
        layer_index = int(layer["layer"])
        tensors = layer.get("cq3_training", [])
        if len(tensors) != 3:
            raise ValueError(f"v44 layer {layer_index} tensor denominator changed")
        for item in tensors:
            tensor = item.get("gguf_tensor")
            expected_prefix = f"blk.{layer_index}."
            if not isinstance(tensor, str) or not tensor.startswith(expected_prefix):
                raise ValueError(f"v44 tensor identity changed in layer {layer_index}")
            parts = tensor.split(".")
            if len(parts) != 4 or parts[2] not in ROLES or parts[3] != "weight":
                raise ValueError(f"unexpected v44 MLP tensor: {tensor}")
            if tensor in seen:
                raise ValueError(f"duplicate v44 MLP tensor: {tensor}")
            seen.add(tensor)
            metrics = item.get("aggregate_calibration_metrics", {})
            score = metrics.get("selected_calibration_nmse")
            if (
                not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or float(score) < 0
                or metrics.get("selection_data") != "calibration_only"
                or metrics.get("all_finite") is not True
            ):
                raise ValueError(f"invalid v44 calibration score: {tensor}")
            ranking.append(
                {
                    "tensor": tensor,
                    "layer": layer_index,
                    "role": parts[2],
                    "score": float(score),
                }
            )
    if len(ranking) != 192 or len(seen) != 192:
        raise ValueError("v44 complete tensor ranking denominator changed")
    return sorted(ranking, key=lambda item: (item["score"], item["tensor"]))


def select_tensor_names(
    ranking: list[dict[str, Any]], threshold: float
) -> tuple[str, ...]:
    """Select tensor names independently at an inclusive frozen threshold."""
    return tuple(
        sorted(item["tensor"] for item in ranking if item["score"] <= threshold)
    )


def selected_payloads(payloads: list[Any], names: tuple[str, ...]) -> list[Any]:
    selected_names = set(names)
    selected = [payload for payload in payloads if payload.tensor in selected_names]
    if len(selected) != len(names) or {payload.tensor for payload in selected} != selected_names:
        raise ValueError("selected CQ3 tensor denominator changed")
    return selected


def overlay_in_place(path: Path, ranges: list[tuple[int, int, Any]]) -> None:
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
    parser.add_argument("--threshold", required=True, type=parse_threshold)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--expected-ranking-sha256", required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.expected_count <= 0 or args.expected_count > 192:
        raise ValueError("expected count must be in [1, 192]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {args.output}")
    if not args.template_partial.is_file():
        raise FileNotFoundError(args.template_partial)
    if args.template_partial.resolve().parent != args.output.resolve().parent:
        raise ValueError("partial and promoted candidate must share a filesystem")

    # collect_training_payloads authenticates all 192 trained payload files and
    # the promoted v44 receipt before any bytes can be published.
    all_payloads = collect_training_payloads(args.training_receipt, args.artifact_root)
    ranking = tensor_nmse_ranking(args.training_receipt)
    ranking_sha256 = canonical_sha256(ranking)
    selected_names = select_tensor_names(ranking, args.threshold)
    selection_sha256 = canonical_sha256(list(selected_names))
    if ranking_sha256 != args.expected_ranking_sha256:
        raise ValueError("frozen tensor-NMSE ranking digest changed")
    if selection_sha256 != args.expected_selection_sha256:
        raise ValueError("frozen tensor-NMSE selection digest changed")
    if len(selected_names) != args.expected_count:
        raise ValueError(
            f"selected tensor count changed: {len(selected_names)} != {args.expected_count}"
        )
    payloads = selected_payloads(all_payloads, selected_names)
    scope = validate_template_scope(
        args.template_partial, args.template_scope_receipt, payloads
    )

    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise RuntimeError(
            "set PYTHONPATH to the pinned llama.cpp gguf-py directory"
        ) from error

    names = set(selected_names)
    reader = GGUFReader(str(args.template_partial), mode="r")
    ranges = validate_tensor_slots(_slots_from_gguf(reader, names), payloads)
    del reader
    byte_ranges = [(start, stop) for start, stop, _ in ranges]
    template_sha256 = sha256_file(args.template_partial)
    template_bytes = args.template_partial.stat().st_size
    template_non_target_sha256 = sha256_excluding(args.template_partial, byte_ranges)

    overlay_in_place(args.template_partial, ranges)
    output_non_target_sha256 = sha256_excluding(args.template_partial, byte_ranges)
    output_reader = GGUFReader(str(args.template_partial), mode="r")
    output_slots = _slots_from_gguf(output_reader, names)
    validate_tensor_slots(output_slots, payloads)
    output_by_name = {tensor.name: tensor for tensor in output_reader.tensors}
    target_results: dict[str, dict[str, Any]] = {}
    score_by_name = {item["tensor"]: item["score"] for item in ranking}
    for start, stop, payload in ranges:
        observed = hashlib.sha256(
            memoryview(output_by_name[payload.tensor].data).cast("B")
        ).hexdigest()
        target_results[payload.tensor] = {
            "layer": payload.layer,
            "role": payload.tensor.split(".")[2],
            "selected_calibration_nmse": score_by_name[payload.tensor],
            "shape": list(payload.shape),
            "offset": start,
            "bytes": stop - start,
            "expected_sha256": payload.sha256,
            "observed_sha256": observed,
            "match": observed == payload.sha256,
        }
    del output_reader

    candidate_sha256 = sha256_file(args.template_partial)
    selected_by_role = {
        role: sorted(
            item["layer"]
            for item in ranking
            if item["tensor"] in names and item["role"] == role
        )
        for role in ROLES
    }
    checks = {
        "full_v44_training_gate_authenticated": len(all_payloads) == 192,
        "calibration_only_ranking_authenticated": len(ranking) == 192,
        "ranking_digest_matches_preregistration": ranking_sha256
        == args.expected_ranking_sha256,
        "selection_digest_matches_preregistration": selection_sha256
        == args.expected_selection_sha256,
        "template_scope_valid": scope.get("valid") is True,
        "exact_selected_tensor_count": len(target_results) == args.expected_count,
        "same_complete_file_bytes": args.template_partial.stat().st_size
        == template_bytes,
        "all_non_target_file_bytes_identical": template_non_target_sha256
        == output_non_target_sha256,
        "all_selected_trained_payloads_exact": all(
            result["match"] for result in target_results.values()
        ),
    }
    receipt = {
        "schema": "qwen38_27b_tensor_nmse_selected_cq3_materialization_v69",
        "model": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "selection": {
            "rule": "selected_calibration_nmse <= inclusive_threshold",
            "threshold": args.threshold,
            "ranking_sha256": ranking_sha256,
            "selection_sha256": selection_sha256,
            "selected_tensor_count": len(selected_names),
            "selected_tensors": list(selected_names),
            "selected_layers_by_role": selected_by_role,
        },
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
            "exact calibration-selected trained-CQ3 GGUF assembly only; "
            "whole-model capability, throughput, energy, novelty, and "
            "breakthrough remain unmeasured"
        ),
    }
    if not receipt["valid"]:
        atomic_json(receipt, args.out)
        raise SystemExit(2)

    atomic_json(receipt, args.out)
    os.replace(args.template_partial, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
