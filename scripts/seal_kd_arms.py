#!/usr/bin/env python3
"""Hash and atomically seal the six Qwen ternary-KD token arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from clark_quantize.kd_data_manifest import seal_kd_arms
from scripts.train_qwen35_ternary_kd import ARMS, PROBE_TOKENS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = args.out.resolve() if args.out else root / "manifest.json"
    result = seal_kd_arms(
        root,
        arms=args.arms,
        model=args.model,
        revision=args.revision,
        probe_tokens=PROBE_TOKENS,
        output=output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
