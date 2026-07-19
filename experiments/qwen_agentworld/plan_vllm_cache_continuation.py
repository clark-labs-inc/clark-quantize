"""Plan source offsets for continuing an interrupted vLLM text cache build."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # Metadata JSON inputs do not require torch.
    torch = None  # type: ignore[assignment]


FINEWEB_RE = re.compile(r"^fineweb2:(?P<config>.+):(?P<index>\d+)$")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
    else:
        if torch is None:
            raise RuntimeError("loading .pt cache files requires torch; pass the .json metadata or activate a torch env")
        data = torch.load(path, map_location="cpu")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"rows is not a list in {path}")
    return rows


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "")


def max_seen_indices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_aya = -1
    max_openr1 = -1
    fineweb_by_config: dict[str, int] = defaultdict(lambda: -1)
    other_ids: list[str] = []
    for row in rows:
        ident = row_id(row)
        if ident.startswith("aya:"):
            max_aya = max(max_aya, int(ident.rsplit(":", 1)[1]))
        elif ident.startswith("openr1_math:"):
            max_openr1 = max(max_openr1, int(ident.rsplit(":", 1)[1]))
        else:
            match = FINEWEB_RE.match(ident)
            if match:
                config = match.group("config")
                fineweb_by_config[config] = max(fineweb_by_config[config], int(match.group("index")))
            elif ident:
                other_ids.append(ident)
    return {
        "aya_next_offset": max_aya + 1,
        "openr1_next_offset": max_openr1 + 1,
        "fineweb_next_offsets_by_config": dict(sorted((key, value + 1) for key, value in fineweb_by_config.items())),
        "other_ids_sample": other_ids[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_or_meta", type=Path)
    parser.add_argument("--aya-target", type=int, default=2000)
    parser.add_argument("--openr1-target", type=int, default=4000)
    parser.add_argument("--fineweb-per-config-target", type=int, default=450)
    args = parser.parse_args()

    rows = load_rows(args.cache_or_meta)
    offsets = max_seen_indices(rows)
    aya_remaining = max(args.aya_target - offsets["aya_next_offset"], 0)
    openr1_remaining = max(args.openr1_target - offsets["openr1_next_offset"], 0)
    fineweb_offsets = offsets["fineweb_next_offsets_by_config"]
    fineweb_next_offset = max(fineweb_offsets.values(), default=0)
    fineweb_remaining_per_config = max(args.fineweb_per_config_target - fineweb_next_offset, 0)
    result = {
        "input": str(args.cache_or_meta),
        "rows": len(rows),
        **offsets,
        "recommended_env": {
            "AYA_OFFSET": offsets["aya_next_offset"],
            "AYA_SAMPLES": aya_remaining,
            "OPENR1_OFFSET": offsets["openr1_next_offset"],
            "OPENR1_SAMPLES": openr1_remaining,
            "FINEWEB_OFFSET": fineweb_next_offset,
            "FINEWEB_PER_CONFIG": fineweb_remaining_per_config,
            "MAX_ROWS": aya_remaining + openr1_remaining + 9 * fineweb_remaining_per_config,
        },
        "shell": (
            f"AYA_OFFSET={offsets['aya_next_offset']} AYA_SAMPLES={aya_remaining} "
            f"OPENR1_OFFSET={offsets['openr1_next_offset']} OPENR1_SAMPLES={openr1_remaining} "
            f"FINEWEB_OFFSET={fineweb_next_offset} FINEWEB_PER_CONFIG={fineweb_remaining_per_config} "
            f"MAX_ROWS={aya_remaining + openr1_remaining + 9 * fineweb_remaining_per_config}"
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
