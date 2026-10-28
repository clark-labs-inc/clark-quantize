#!/usr/bin/env python3
"""Verify a selective GGUF transcode without trusting container offsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO


SELECTED_SUFFIXES = (
    ".attn_qkv.weight",
    ".attn_gate.weight",
    ".ssm_out.weight",
    ".ssm_alpha.weight",
    ".ssm_beta.weight",
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
)
RECURRENT_SUFFIXES = (
    ".attn_qkv.weight",
    ".attn_gate.weight",
    ".ssm_out.weight",
    ".ssm_alpha.weight",
    ".ssm_beta.weight",
)
RECURRENT_AND_ATTENTION_Q_SUFFIXES = RECURRENT_SUFFIXES + (".attn_q.weight",)


def is_selected(name: str, profile: str) -> bool:
    if profile == "all_q8_family":
        suffixes = SELECTED_SUFFIXES
    elif profile == "recurrent_only":
        suffixes = RECURRENT_SUFFIXES
    else:
        suffixes = RECURRENT_AND_ATTENTION_Q_SUFFIXES
    return name.startswith("blk.") and name.endswith(suffixes)


def update_payload_hash(
    digest: "hashlib._Hash",
    handle: BinaryIO,
    *,
    name: str,
    offset: int,
    size: int,
) -> None:
    encoded = name.encode()
    digest.update(len(encoded).to_bytes(4, "little"))
    digest.update(encoded)
    digest.update(size.to_bytes(8, "little"))
    handle.seek(offset)
    remaining = size
    while remaining:
        block = handle.read(min(16 * 1024 * 1024, remaining))
        if not block:
            raise EOFError(f"unexpected EOF in {name}")
        digest.update(block)
        remaining -= len(block)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-selected", type=int, default=288)
    parser.add_argument(
        "--selection-profile",
        choices=(
            "all_q8_family",
            "recurrent_only",
            "recurrent_and_attention_q",
        ),
        default="all_q8_family",
    )
    parser.add_argument(
        "--expected-selected-weight-count", type=int, default=6_734_479_360
    )
    parser.add_argument("--expected-remaining-q8", type=int, default=0)
    parser.add_argument("--expected-target-type", default="Q4_K")
    parser.add_argument("--expected-bytes-saved", type=int, default=3_367_239_680)
    parser.add_argument(
        "--expected-selected-payload-bytes-saved",
        type=int,
        help=(
            "Expected bytes removed from selected tensor payloads. Defaults to "
            "--expected-bytes-saved; pass it separately when deterministic GGUF "
            "provenance metadata changes the container header size."
        ),
    )
    args = parser.parse_args()

    from gguf import GGUFReader

    control_reader = GGUFReader(str(args.control), mode="r")
    candidate_reader = GGUFReader(str(args.candidate), mode="r")
    control = {tensor.name: tensor for tensor in control_reader.tensors}
    candidate = {tensor.name: tensor for tensor in candidate_reader.tensors}
    if set(control) != set(candidate):
        raise ValueError("control and candidate tensor names differ")

    selected_names = sorted(
        name for name in control if is_selected(name, args.selection_profile)
    )
    selected_weight_count = 0
    selected_payload_saved = 0
    structural_mismatches: list[str] = []
    control_types: dict[str, int] = {}
    candidate_types: dict[str, int] = {}
    control_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()

    with args.control.open("rb") as control_file, args.candidate.open("rb") as candidate_file:
        for name in sorted(control):
            before = control[name]
            after = candidate[name]
            before_type = before.tensor_type.name
            after_type = after.tensor_type.name
            control_types[before_type] = control_types.get(before_type, 0) + 1
            candidate_types[after_type] = candidate_types.get(after_type, 0) + 1
            if tuple(before.shape) != tuple(after.shape):
                structural_mismatches.append(f"{name}:shape")
                continue
            if name in selected_names:
                if before_type != "Q8_0" or after_type != args.expected_target_type:
                    structural_mismatches.append(
                        f"{name}:type:{before_type}->{after_type}"
                    )
                selected_weight_count += int(before.n_elements)
                selected_payload_saved += int(before.n_bytes) - int(after.n_bytes)
                continue
            if before_type != after_type or int(before.n_bytes) != int(after.n_bytes):
                structural_mismatches.append(f"{name}:nonselected-structure")
                continue
            update_payload_hash(
                control_digest,
                control_file,
                name=name,
                offset=int(before.data_offset),
                size=int(before.n_bytes),
            )
            update_payload_hash(
                candidate_digest,
                candidate_file,
                name=name,
                offset=int(after.data_offset),
                size=int(after.n_bytes),
            )

    del control_reader, candidate_reader
    control_nonselected_sha256 = control_digest.hexdigest()
    candidate_nonselected_sha256 = candidate_digest.hexdigest()
    actual_bytes_saved = args.control.stat().st_size - args.candidate.stat().st_size
    expected_selected_payload_bytes_saved = (
        args.expected_selected_payload_bytes_saved
        if args.expected_selected_payload_bytes_saved is not None
        else args.expected_bytes_saved
    )
    checks = {
        "selected_tensor_count_exact": len(selected_names) == args.expected_selected,
        "selected_types_exact": not structural_mismatches,
        "selected_weight_count_exact": (
            selected_weight_count == args.expected_selected_weight_count
        ),
        "q8_count_exact": (
            candidate_types.get("Q8_0", 0) == args.expected_remaining_q8
        ),
        "target_count_increased_by_selected": (
            candidate_types.get(args.expected_target_type, 0)
            - control_types.get(args.expected_target_type, 0)
            == args.expected_selected
        ),
        "nonselected_payloads_byte_identical": (
            control_nonselected_sha256 == candidate_nonselected_sha256
        ),
        "artifact_bytes_saved_exact": actual_bytes_saved == args.expected_bytes_saved,
        "selected_payload_bytes_saved_exact": (
            selected_payload_saved == expected_selected_payload_bytes_saved
        ),
    }
    payload = {
        "schema": "qwen38_27b_selective_gguf_verification_v1",
        "terminal": True,
        "valid": all(checks.values()),
        "control": {"path": str(args.control), "bytes": args.control.stat().st_size},
        "candidate": {
            "path": str(args.candidate),
            "bytes": args.candidate.stat().st_size,
        },
        "selected_tensor_count": len(selected_names),
        "selected_weight_count": selected_weight_count,
        "selection_profile": args.selection_profile,
        "expected_target_type": args.expected_target_type,
        "selected_payload_bytes_saved": selected_payload_saved,
        "expected_selected_payload_bytes_saved": expected_selected_payload_bytes_saved,
        "artifact_bytes_saved": actual_bytes_saved,
        "control_type_counts": control_types,
        "candidate_type_counts": candidate_types,
        "control_nonselected_payloads_sha256": control_nonselected_sha256,
        "candidate_nonselected_payloads_sha256": candidate_nonselected_sha256,
        "structural_mismatches": structural_mismatches,
        "checks": checks,
        "claim_boundary": "Container and payload integrity only; no model correctness or quality claim.",
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "terminal", "valid": payload["valid"]}))
    return 0 if payload["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
