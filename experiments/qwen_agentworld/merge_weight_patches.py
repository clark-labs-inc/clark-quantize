"""Merge safetensors weight patches in order, with later patches overriding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def canonical_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model.") :]
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tensors: dict[str, torch.Tensor] = {}
    sources: dict[str, str] = {}
    for patch in args.patch:
        with safe_open(patch, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                out_key = canonical_key(key)
                tensors[out_key] = handle.get_tensor(key).contiguous()
                sources[out_key] = str(patch)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    report = {
        "out": str(args.out),
        "patches": [str(path) for path in args.patch],
        "tensors": len(tensors),
        "sources": sources,
    }
    report_path = args.out.with_suffix(".merge_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "report": str(report_path), "tensors": len(tensors)}, indent=2))


if __name__ == "__main__":
    main()
