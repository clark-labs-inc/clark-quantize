#!/usr/bin/env python3
"""Materialize a chained scale-collapse checkpoint for evaluation/export."""

import argparse
import json
from pathlib import Path

from staged_scale_codebook_state import materialize_staged_scale_codebook_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-device", default="cpu")
    args = parser.parse_args()
    receipt = materialize_staged_scale_codebook_checkpoint(
        args.checkpoint,
        args.out,
        args.source_device,
    )
    print(json.dumps({
        "manifest": receipt["manifest"],
        "source_device": receipt["source_device"],
        "scale_step": receipt["scale_step"],
        "surfaces": len(receipt["surfaces"]),
        "dense_state": receipt["dense_state"]["path"],
        "runtime_state": receipt["runtime_state"]["path"],
    }, indent=2))


if __name__ == "__main__":
    main()
