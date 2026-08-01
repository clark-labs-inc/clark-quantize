#!/usr/bin/env python3
"""Materialize a scale-codebook checkpoint into dense and packed bases."""

import argparse
import json
from pathlib import Path

from scale_codebook_state import materialize_scale_codebook_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-device", default="cpu")
    args = parser.parse_args()
    receipt = materialize_scale_codebook_checkpoint(
        args.checkpoint, args.out, args.source_device
    )
    print(json.dumps({
        "manifest": receipt["manifest"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "scale_step": receipt["scale_step"],
        "runtime_state": receipt["runtime_state"]["path"],
        "dense_state": receipt["dense_state"]["path"],
    }, indent=2))


if __name__ == "__main__":
    main()
