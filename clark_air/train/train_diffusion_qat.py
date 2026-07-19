"""Skeleton entrypoint for diffusion transformer Clark Air QAT."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Clark Air diffusion transformer QAT")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--prompt-pool", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            "configured diffusion QAT: "
            f"teacher={args.teacher_model} prompts={args.prompt_pool} "
            "loss=symmetric_velocity+hidden_geometry+codebook"
        )
        return
    raise SystemExit("training implementation is intentionally gated; use existing Sana trainer or wire a confirmed runner")


if __name__ == "__main__":
    main()
