"""Skeleton entrypoint for dense LLM Clark Air QAT.

This module intentionally performs no model/provider selection by default.
The first experiment must be launched with an explicit teacher model and data
source so no paid or heavyweight model calls happen accidentally.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Clark Air dense LLM QAT")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            "configured dense LLM QAT: "
            f"teacher={args.teacher_model} data={args.data} "
            "loss=teacher_kl+hidden_geometry+codebook"
        )
        return
    raise SystemExit("training implementation is intentionally gated; run with --dry-run or wire a confirmed model runner")


if __name__ == "__main__":
    main()
