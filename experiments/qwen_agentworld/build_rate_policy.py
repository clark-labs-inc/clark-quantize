"""Build a Clark Air mixed-rate policy for Qwen AgentWorld.

This produces the first explicit binary/ternary/FP8 storage policy for the
Qwen AgentWorld MoE checkpoint. It uses safetensors metadata plus optional
router-utilization calibration. It does not write model weights.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.rate import RateCandidate, choose_rates


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
TERNARY_BPW = 64.0 / 40.0


def lowbit_rate_names(scale_bits: int) -> tuple[str, str]:
    if scale_bits == 16:
        return "ternary40_fp16", "binary_fp16"
    if scale_bits == 8:
        return "ternary40_log8", "binary_log8"
    return f"ternary40_scale{scale_bits}", f"binary_scale{scale_bits}"


def tensor_layer(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def load_router_utilization(path: Path | None) -> dict[int, list[float]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    layers = data.get("router", {}).get("layers", {})
    result = {}
    for layer, info in layers.items():
        if layer.isdigit():
            result[int(layer)] = [float(x) for x in info.get("route_fraction", [])]
    return result


def bits(params: float, bpw: float) -> float:
    return float(params) * float(bpw)


def scaled_damage(params: float, sensitivity: float, penalty: float) -> float:
    # Keep the proxy in a numerically tame range. It is only used for ordering
    # until a real Fisher diagonal is available.
    return float(params) / 1e9 * float(sensitivity) * float(penalty)


def add_candidate(
    rows: list[RateCandidate],
    *,
    group: str,
    rate: str,
    params: float,
    bpw: float,
    damage: float,
    meta: dict[str, Any],
) -> None:
    rows.append(RateCandidate(group=group, rate=rate, bits=bits(params, bpw), damage=damage, meta=meta))


def load_damage_measurements(
    paths: list[Path] | None,
    *,
    metric: str,
    scale: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    measurements: dict[tuple[str, str], dict[str, Any]] = {}
    if not paths:
        return measurements
    for path in paths:
        data = json.loads(path.read_text())
        for row in data.get("rows", []):
            if row.get("missing"):
                continue
            group = row.get("group")
            rate = row.get("rate")
            if not group or not rate or metric not in row:
                continue
            value = float(row[metric]) * scale
            measurements[(str(group), str(rate))] = {
                "value": value,
                "raw_value": float(row[metric]),
                "metric": metric,
                "scale": scale,
                "source": str(path),
                "relative_rmse": row.get("relative_rmse"),
                "top1_agreement": row.get("top1_agreement"),
            }
    return measurements


def apply_damage_measurements(
    candidates: list[RateCandidate],
    measurements: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[RateCandidate], int]:
    if not measurements:
        return candidates, 0
    updated = []
    applied = 0
    for candidate in candidates:
        measurement = measurements.get((candidate.group, candidate.rate))
        if measurement is None:
            updated.append(candidate)
            continue
        meta = dict(candidate.meta or {})
        meta["proxy_damage"] = candidate.damage
        meta["measured_damage"] = measurement["value"]
        meta["measured_damage_raw"] = measurement["raw_value"]
        meta["measured_damage_metric"] = measurement["metric"]
        meta["measured_damage_scale"] = measurement["scale"]
        meta["damage_measurement_source"] = measurement["source"]
        if measurement.get("relative_rmse") is not None:
            meta["measured_relative_rmse"] = measurement["relative_rmse"]
        if measurement.get("top1_agreement") is not None:
            meta["measured_top1_agreement"] = measurement["top1_agreement"]
        updated.append(
            RateCandidate(
                group=candidate.group,
                rate=candidate.rate,
                bits=candidate.bits,
                damage=measurement["value"],
                meta=meta,
            )
        )
        applied += 1
    return updated, applied


def enforce_measured_damage_floors(candidates: list[RateCandidate]) -> tuple[list[RateCandidate], int]:
    """Prevent unmeasured lower-bit options from undercutting measured options.

    If ternary for a group has measured damage but binary does not, binary
    should not keep a tiny proxy damage and become preferred only because it
    was not measured. Measured candidates remain authoritative.
    """

    by_group: dict[str, list[RateCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.group].append(candidate)

    updated = []
    raised = 0
    for candidate in candidates:
        meta = dict(candidate.meta or {})
        if "measured_damage" in meta:
            updated.append(candidate)
            continue

        floor_sources = []
        for other in by_group[candidate.group]:
            other_meta = other.meta or {}
            if "measured_damage" not in other_meta:
                continue
            if other.bits > candidate.bits and other.damage > candidate.damage:
                floor_sources.append(other)

        if not floor_sources:
            updated.append(candidate)
            continue

        floor = max(floor_sources, key=lambda row: row.damage)
        meta["proxy_damage"] = candidate.damage
        meta["damage_floor_from_measured_rate"] = floor.rate
        meta["damage_floor_from_measured_value"] = floor.damage
        updated.append(
            RateCandidate(
                group=candidate.group,
                rate=candidate.rate,
                bits=candidate.bits,
                damage=floor.damage,
                meta=meta,
            )
        )
        raised += 1
    return updated, raised


def candidates_for_tensor(
    name: str,
    info: dict[str, Any],
    *,
    group_size: int,
    scale_bits: int,
) -> list[RateCandidate]:
    bucket = info["bucket"]
    params = float(info["numel"])
    shape = info.get("shape", [])
    rows: list[RateCandidate] = []
    meta = {"tensor": name, "bucket": bucket, "params": int(params)}
    ternary_rate, binary_rate = lowbit_rate_names(scale_bits)

    if bucket in {"router", "norm", "linear_attention_state", "other"}:
        is_matrix_like = len(shape) >= 2 and params >= 4096
        if bucket != "other" or not is_matrix_like:
            add_candidate(rows, group=name, rate="bf16", params=params, bpw=16.0, damage=0.0, meta=meta)
            return rows

        ternary_bpw = TERNARY_BPW + scale_bits / group_size
        binary_bpw = 1.0 + scale_bits / group_size
        add_candidate(rows, group=name, rate="bf16", params=params, bpw=16.0, damage=0.0, meta=meta)
        add_candidate(rows, group=name, rate=ternary_rate, params=params, bpw=ternary_bpw, damage=scaled_damage(params, 1.35, 0.30), meta=meta)
        add_candidate(rows, group=name, rate=binary_rate, params=params, bpw=binary_bpw, damage=scaled_damage(params, 1.35, 1.2), meta=meta)
        return rows

    if bucket in {"embedding", "lm_head"}:
        ternary_bpw = TERNARY_BPW + scale_bits / group_size
        binary_bpw = 1.0 + scale_bits / group_size
        add_candidate(rows, group=name, rate="bf16", params=params, bpw=16.0, damage=0.0, meta=meta)
        add_candidate(rows, group=name, rate="fp8", params=params, bpw=8.0, damage=scaled_damage(params, 3.0, 0.08), meta=meta)
        add_candidate(rows, group=name, rate=ternary_rate, params=params, bpw=ternary_bpw, damage=scaled_damage(params, 3.0, 0.30), meta=meta)
        add_candidate(rows, group=name, rate=binary_rate, params=params, bpw=binary_bpw, damage=scaled_damage(params, 3.0, 1.4), meta=meta)
        return rows

    sensitivity = {
        "full_attention": 1.6,
        "linear_attention_projection": 1.25,
        "shared_expert": 1.15,
        "mlp": 1.2,
    }.get(bucket, 1.0)
    ternary_bpw = TERNARY_BPW + scale_bits / group_size
    binary_bpw = 1.0 + scale_bits / group_size
    add_candidate(rows, group=name, rate="bf16", params=params, bpw=16.0, damage=0.0, meta=meta)
    add_candidate(rows, group=name, rate=ternary_rate, params=params, bpw=ternary_bpw, damage=scaled_damage(params, sensitivity, 0.25), meta=meta)
    add_candidate(rows, group=name, rate=binary_rate, params=params, bpw=binary_bpw, damage=scaled_damage(params, sensitivity, 1.0), meta=meta)
    return rows


def candidates_for_moe_expert_slices(
    name: str,
    info: dict[str, Any],
    *,
    router_util: dict[int, list[float]],
    group_size: int,
    scale_bits: int,
) -> list[RateCandidate]:
    shape = info.get("shape", [])
    if not shape or int(shape[0]) <= 1:
        return candidates_for_tensor(name, info, group_size=group_size, scale_bits=scale_bits)

    layer = tensor_layer(name)
    expert_count = int(shape[0])
    params_per_expert = float(info["numel"]) / expert_count
    util = router_util.get(layer or -1)
    if not util or len(util) != expert_count:
        util = [1.0 / expert_count] * expert_count

    rows: list[RateCandidate] = []
    ternary_bpw = TERNARY_BPW + scale_bits / group_size
    binary_bpw = 1.0 + scale_bits / group_size
    ternary_rate, binary_rate = lowbit_rate_names(scale_bits)
    for expert_idx, route_fraction in enumerate(util):
        group = f"{name}::expert_{expert_idx:03d}"
        sensitivity = 0.05 + expert_count * float(route_fraction)
        meta = {
            "tensor": name,
            "bucket": "moe_expert",
            "layer": layer,
            "expert": expert_idx,
            "params": int(params_per_expert),
            "route_fraction": route_fraction,
        }
        add_candidate(rows, group=group, rate="bf16", params=params_per_expert, bpw=16.0, damage=0.0, meta=meta)
        add_candidate(
            rows,
            group=group,
            rate=ternary_rate,
            params=params_per_expert,
            bpw=ternary_bpw,
            damage=scaled_damage(params_per_expert, sensitivity, 0.22),
            meta=meta,
        )
        add_candidate(
            rows,
            group=group,
            rate=binary_rate,
            params=params_per_expert,
            bpw=binary_bpw,
            damage=scaled_damage(params_per_expert, sensitivity, 1.0),
            meta=meta,
        )
    return rows


def summarize(policy_rows: list[dict[str, Any]], reference_bits: float, target_bits: float) -> dict[str, Any]:
    by_rate: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "bits": 0.0, "params": 0.0})
    by_bucket: dict[str, dict[str, float]] = defaultdict(lambda: {"groups": 0, "bits": 0.0, "params": 0.0})
    for row in policy_rows:
        meta = row.get("meta") or {}
        params = float(meta.get("params", 0))
        bucket = str(meta.get("bucket", "unknown"))
        for table, key in ((by_rate, row["rate"]), (by_bucket, bucket)):
            table[key]["groups"] += 1
            table[key]["bits"] += float(row["bits"])
            table[key]["params"] += params

    chosen_bits = sum(float(row["bits"]) for row in policy_rows)
    return {
        "chosen_bits": chosen_bits,
        "chosen_bytes": chosen_bits / 8,
        "chosen_gib": chosen_bits / 8 / 1024**3,
        "target_bits": target_bits,
        "target_bytes": target_bits / 8,
        "reference_bits": reference_bits,
        "compression": reference_bits / chosen_bits if chosen_bits else math.inf,
        "by_rate": {k: dict(v) for k, v in sorted(by_rate.items())},
        "by_bucket": {k: dict(v) for k, v in sorted(by_bucket.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="/workspace/agentworld_runs/model_inventory.json", type=Path)
    parser.add_argument("--router-calibration", default=None, type=Path)
    parser.add_argument("--out", default="/workspace/agentworld_runs/policy_qwen_agentworld_8x.json", type=Path)
    parser.add_argument("--target-bytes", type=float, default=None)
    parser.add_argument("--group-size", type=int, default=112)
    parser.add_argument("--scale-bits", type=int, default=8)
    parser.add_argument("--damage-measurements", nargs="*", type=Path)
    parser.add_argument("--damage-metric", default="kl_mean")
    parser.add_argument("--damage-scale", type=float, default=1.0)
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text())
    router_util = load_router_utilization(args.router_calibration)
    target_bytes = float(args.target_bytes or inventory["target_8x_bytes"])
    reference_bits = float(inventory["total_size_bytes"]) * 8.0
    target_bits = target_bytes * 8.0

    candidates: list[RateCandidate] = []
    for name, info in sorted(inventory["tensor_summary"].items()):
        if info["bucket"] == "moe_expert":
            candidates.extend(
                candidates_for_moe_expert_slices(
                    name,
                    info,
                    router_util=router_util,
                    group_size=args.group_size,
                    scale_bits=args.scale_bits,
                )
            )
        else:
            candidates.extend(candidates_for_tensor(name, info, group_size=args.group_size, scale_bits=args.scale_bits))

    measurements = load_damage_measurements(
        args.damage_measurements,
        metric=args.damage_metric,
        scale=args.damage_scale,
    )
    candidates, measured_overrides = apply_damage_measurements(candidates, measurements)
    candidates, measured_damage_floors = enforce_measured_damage_floors(candidates)

    policy = choose_rates(candidates, target_bits=target_bits, max_exact_saving_states=10_000)
    rows = policy.as_rows()
    result = {
        "inventory": str(args.inventory),
        "router_calibration": str(args.router_calibration) if args.router_calibration else None,
        "damage_measurements": [str(path) for path in args.damage_measurements or []],
        "damage_metric": args.damage_metric,
        "damage_scale": args.damage_scale,
        "measured_overrides": measured_overrides,
        "measured_damage_floors": measured_damage_floors,
        "group_size": args.group_size,
        "scale_bits": args.scale_bits,
        "feasible": policy.feasible,
        "bit_price": policy.bit_price,
        "summary": summarize(rows, reference_bits=reference_bits, target_bits=target_bits),
        "choices": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("feasible", "bit_price", "summary")}, indent=2))


if __name__ == "__main__":
    main()
