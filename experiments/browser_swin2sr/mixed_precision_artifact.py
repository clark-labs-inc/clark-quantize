"""Validation and exact code recovery for mixed browser precision islands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PrecisionContract:
    fp16_weights: frozenset[str]
    int7_weights: frozenset[str]
    int7_records: dict[str, dict]
    islands: list[dict]


def retained_precision_islands(materialization: dict) -> list[dict]:
    islands = materialization.get("retained_precision_islands")
    if islands is None:
        legacy = materialization.get("retained_precision_island")
        islands = [] if legacy is None else [legacy]
    names = [island.get("name") for island in islands]
    if len(names) != len(set(names)):
        raise ValueError("retained-precision island names are not unique")
    return [dict(island) for island in islands]


def validate_precision_contract(
    materialization: dict,
    state: dict[str, torch.Tensor],
    expected_weights: set[str],
) -> PrecisionContract:
    islands = retained_precision_islands(materialization)
    fp16_weights: set[str] = set()
    int7_weights: set[str] = set()
    int7_records: dict[str, dict] = {}
    claimed: set[str] = set()
    for island in islands:
        weights = set(island.get("weights", []))
        if not weights or not weights <= expected_weights:
            raise ValueError("retained-precision island is outside the target scope")
        if claimed & weights:
            raise ValueError("retained-precision islands overlap")
        claimed |= weights
        actual_values = sum(state[name].numel() for name in weights)
        if actual_values != int(island["weight_values"]):
            raise ValueError("retained-precision island value count changed")
        precision = island.get("precision")
        if precision == "fp16":
            if island.get("storage_dtype") != "float16":
                raise ValueError("FP16 island storage dtype changed")
            if any(
                not torch.equal(
                    state[name].float(),
                    state[name].to(torch.float16).to(torch.float32),
                )
                for name in weights
            ):
                raise ValueError("retained FP16 island is not an exact roundtrip")
            fp16_weights |= weights
            continue
        if precision == "int7":
            if (
                island.get("storage_dtype") != "packed_int7"
                or int(island.get("bit_width", 0)) != 7
                or island.get("granularity") != "per_output"
            ):
                raise ValueError("unsupported int7 island contract")
            int7_weights |= weights
            for name in weights:
                int7_records[name] = island
            continue
        if precision != "ternary":
            raise ValueError(f"unsupported retained precision: {precision}")
    return PrecisionContract(
        fp16_weights=frozenset(fp16_weights),
        int7_weights=frozenset(int7_weights),
        int7_records=int7_records,
        islands=islands,
    )


def recover_symmetric_per_output(
    tensor: torch.Tensor,
    bits: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if bits < 2 or bits > 8:
        raise ValueError("symmetric integer width must be between 2 and 8")
    value = tensor.detach().cpu().float().numpy()
    if value.ndim != 4 or tuple(value.shape[2:]) != (3, 3):
        raise ValueError("per-output mixed weight must be a 3x3 convolution")
    rows = value.reshape(value.shape[0], -1)
    limit = (1 << (bits - 1)) - 1
    maximum = np.max(np.abs(rows), axis=1)
    scales = (maximum / limit).astype(np.float16).astype(np.float32)
    safe = np.where(scales > 0, scales, np.ones_like(scales))
    codes = np.rint(rows / safe[:, None]).clip(-limit, limit).astype(np.int16)
    recovered = (codes.astype(np.float32) * scales[:, None]).reshape(value.shape)
    maximum_error = float(np.max(np.abs(recovered - value)))
    if maximum_error != 0.0:
        raise ValueError(
            f"mixed integer state is not exactly recoverable: {maximum_error}"
        )
    return codes.reshape(value.shape), scales, maximum_error
