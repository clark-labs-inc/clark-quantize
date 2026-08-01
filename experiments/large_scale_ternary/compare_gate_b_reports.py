"""Produce a paired, sealed comparison of two Gate B evaluation arms."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_b_protocol import compare_gate_b_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    comparison = compare_gate_b_reports(
        left,
        right,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, args.out)
    print(f"wrote {args.out} ({comparison['content_sha256']})")


if __name__ == "__main__":
    main()
