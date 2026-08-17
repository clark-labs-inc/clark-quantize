#!/usr/bin/env python3
"""Seal a checkpoint-bound QAT continuation decision from Gate B reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clark_quantize.qwen_quality_gate import (
    atomic_write_json,
    authorize_gate_b_continuation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-request", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--minimum-retention", type=float, default=0.60)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    authorization = authorize_gate_b_continuation(
        json.loads(args.gate_request.read_text()),
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        minimum_retention=args.minimum_retention,
    )
    atomic_write_json(args.out, authorization)
    print(json.dumps(authorization, indent=2), flush=True)
    raise SystemExit(0 if authorization["pass"] else 2)


if __name__ == "__main__":
    main()
