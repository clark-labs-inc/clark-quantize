"""Validation and decision helpers for per-surface scale allocations."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter


PRECISION_LABELS = {"q1", "q2", "q4", "q8", "fp16"}


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def choice_storage(choice: dict) -> dict[str, int]:
    """Reconcile calibration bytes with the browser export's aligned span."""

    label = str(choice["label"])
    if label not in PRECISION_LABELS:
        raise ValueError(f"unsupported precision label: {label}")
    groups = int(choice["groups"])
    centers = int(choice["centers"])
    if min(groups, centers) < 1:
        raise ValueError("scale choice requires positive groups and centers")
    if label == "fp16":
        center_bytes = 0
        index_bytes = groups * 2
    else:
        requested_bits = int(label[1:])
        if centers > min(1 << requested_bits, groups):
            raise ValueError("scale center count exceeds its precision label")
        stored_bits = 0 if centers == 1 else math.ceil(math.log2(centers))
        center_bytes = centers * 2
        index_bytes = (groups * stored_bits + 7) // 8
    payload_bytes = center_bytes + index_bytes
    calibration_bytes = payload_bytes + (-payload_bytes) % 4
    physical_span_bytes = (
        center_bytes
        + (-center_bytes) % 4
        + index_bytes
        + (-index_bytes) % 4
    )
    if calibration_bytes != int(choice["storage_bytes"]):
        raise ValueError("calibration scale-storage receipt is inconsistent")
    return {
        "payload_bytes": payload_bytes,
        "calibration_storage_bytes": calibration_bytes,
        "physical_aligned_span_bytes": physical_span_bytes,
    }


def precision_map_storage(report: dict, precision_map: dict[str, str]) -> dict:
    rows = {row["surface"]: row for row in report["surfaces"]}
    if set(precision_map) != set(rows):
        missing = sorted(set(rows) - set(precision_map))
        extra = sorted(set(precision_map) - set(rows))
        raise ValueError(
            f"precision map does not cover every surface: missing={missing[:3]} "
            f"extra={extra[:3]}"
        )
    totals = Counter()
    receipts = []
    labels = Counter()
    for surface in sorted(rows):
        label = precision_map[surface]
        choices = {
            choice["label"]: choice for choice in rows[surface]["choices"]
        }
        if label not in choices:
            raise ValueError(f"surface {surface} has no {label} choice")
        storage = choice_storage(choices[label])
        totals.update(storage)
        labels[label] += 1
        receipts.append({"surface": surface, "label": label, **storage})
    return {
        **dict(totals),
        "surface_count": len(receipts),
        "choice_counts": dict(sorted(labels.items())),
        "surface_receipts_sha256": canonical_sha256(receipts),
    }


def uniform_map(report: dict, label: str) -> dict[str, str]:
    if label not in PRECISION_LABELS:
        raise ValueError(f"unsupported uniform precision: {label}")
    return {row["surface"]: label for row in report["surfaces"]}


def advancement_decision(
    candidates: dict,
    shards: dict,
    *,
    minimum_relative_gain: float,
) -> dict:
    """Select the smallest candidate beating its matched uniform control."""

    if not 0 <= minimum_relative_gain < 1:
        raise ValueError("relative-gain threshold must be in [0, 1)")
    passing = []
    decisions = {}
    for name, candidate in sorted(
        candidates.items(), key=lambda item: int(item[1]["budget_bytes"])
    ):
        control = candidate["uniform_control"]
        shard_gates = {}
        for shard_name, shard in shards.items():
            actual = float(shard["states"][name]["fragility"])
            baseline = float(shard["states"][control]["fragility"])
            relative_gain = 1 - actual / max(baseline, 1e-30)
            shard_gates[shard_name] = {
                "candidate_fragility": actual,
                "control_fragility": baseline,
                "relative_gain": relative_gain,
                "passes": relative_gain >= minimum_relative_gain,
            }
        storage = candidate["storage"]
        storage_passes = (
            int(storage["calibration_storage_bytes"])
            == int(candidate["budget_bytes"])
            and int(storage["physical_aligned_span_bytes"])
            <= int(candidate["budget_bytes"])
        )
        passes = storage_passes and all(
            gate["passes"] for gate in shard_gates.values()
        )
        decisions[name] = {
            "uniform_control": control,
            "minimum_relative_gain": minimum_relative_gain,
            "storage_passes": storage_passes,
            "shards": shard_gates,
            "passes": passes,
        }
        if passes:
            passing.append((int(candidate["budget_bytes"]), name))
    return {
        "candidate_decisions": decisions,
        "selected_candidate": min(passing)[1] if passing else None,
        "selection_rule": (
            "smallest exact-byte candidate with at least the preregistered "
            "relative fragility gain over its matched uniform control on both "
            "allocation-calibration and disjoint training-only shards"
        ),
    }
