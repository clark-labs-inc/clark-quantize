#!/usr/bin/env python3
"""Run the bounded Qwen 27B collapse-localization matrix sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAMILY_ARMS: tuple[tuple[str, str | None], ...] = (
    ("bf16", None),
    ("full_patch", ""),
    ("matrices_only", r"(?:norm|layernorm)\.weight$"),
    ("protect_linear_attention", r"\.linear_attn\."),
    ("protect_full_attention", r"\.self_attn\."),
    ("protect_all_attention", r"\.(?:linear_attn|self_attn)\."),
    ("mlp_only", r"\.(?:linear_attn|self_attn)\.|(?:norm|layernorm)\.weight$"),
    (
        "linear_attention_only",
        r"\.self_attn\.|\.mlp\.|(?:input_layernorm|post_attention_layernorm)\.weight$",
    ),
    (
        "full_attention_only",
        r"\.linear_attn\.|\.mlp\.|(?:input_layernorm|post_attention_layernorm)\.weight$",
    ),
)

TARGET_ARMS: tuple[tuple[str, str | None], ...] = (
    ("protect_linear_ab_with_norms", r"\.linear_attn\.in_proj_[ab]\."),
    (
        "canonical_400_matrices",
        r"\.linear_attn\.in_proj_[ab]\.|(?:norm|layernorm)\.weight$",
    ),
    (
        "linear_ab_only",
        r"^(?!.*\.linear_attn\.in_proj_[ab]\.weight$).*",
    ),
    (
        "linear_core_only",
        r"^(?!.*\.linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)\.weight$).*",
    ),
    (
        "attention_matrices_only",
        r"^(?!.*(?:\.linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj))\.weight$).*",
    ),
)

PROJECTION_ARMS: tuple[tuple[str, str | None], ...] = (
    ("linear_qkv_only", r"^(?!.*\.linear_attn\.in_proj_qkv\.weight$).*"),
    ("linear_z_only", r"^(?!.*\.linear_attn\.in_proj_z\.weight$).*"),
    ("linear_out_only", r"^(?!.*\.linear_attn\.out_proj\.weight$).*"),
    ("mlp_gate_only", r"^(?!.*\.mlp\.gate_proj\.weight$).*"),
    ("mlp_up_only", r"^(?!.*\.mlp\.up_proj\.weight$).*"),
    ("mlp_down_only", r"^(?!.*\.mlp\.down_proj\.weight$).*"),
)

COMPOSITION_ARMS: tuple[tuple[str, str | None], ...] = (
    (
        "safe_read_stack",
        r"^(?!.*(?:\.linear_attn\.(?:in_proj_a|in_proj_b|out_proj)|\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|\.mlp\.(?:gate_proj|up_proj))\.weight$).*",
    ),
    (
        "safe_read_plus_qkv_stack",
        r"^(?!.*(?:\.linear_attn\.(?:in_proj_a|in_proj_b|in_proj_qkv|out_proj)|\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|\.mlp\.(?:gate_proj|up_proj))\.weight$).*",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-rows-per-task", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--matrix",
        choices=("family", "target", "projection", "composition"),
        default="family",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    script = Path(__file__).with_name("forensic_checkpoint_ablation.py")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    run_rows: list[dict[str, Any]] = []
    if args.matrix == "family":
        arms = FAMILY_ARMS
    elif args.matrix == "target":
        arms = TARGET_ARMS
    elif args.matrix == "projection":
        arms = PROJECTION_ARMS
    else:
        arms = COMPOSITION_ARMS
    for arm, exclude_regex in arms:
        out = args.out_dir / f"{arm}.json"
        log = args.out_dir / f"{arm}.log"
        if out.exists() and not args.force:
            report = json.loads(out.read_text())
            if report.get("schema") == "qwen38_hard_patch_forensic_ablation_v1":
                print(json.dumps({"event": "arm_skip_complete", "arm": arm, "out": str(out)}), flush=True)
                run_rows.append({"arm": arm, "status": "reused", "report": str(out), "sha256": sha256_file(out)})
                continue
        command = [
            sys.executable,
            str(script),
            "--model-dir",
            str(args.model_dir),
            "--cache",
            str(args.cache),
            "--arm",
            arm,
            "--out",
            str(out),
            "--max-rows-per-task",
            str(args.max_rows_per_task),
            "--max-tokens",
            str(args.max_tokens),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if arm != "bf16":
            command.extend(["--patch", str(args.patch)])
        if exclude_regex:
            command.extend(["--exclude-regex", exclude_regex])
        print(json.dumps({"event": "arm_start", "arm": arm, "command": command}), flush=True)
        arm_started = datetime.now(timezone.utc)
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
        arm_finished = datetime.now(timezone.utc)
        row = {
            "arm": arm,
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "report": str(out),
            "log": str(log),
            "started_at": arm_started.isoformat(),
            "finished_at": arm_finished.isoformat(),
            "elapsed_seconds": (arm_finished - arm_started).total_seconds(),
        }
        if out.exists():
            row["sha256"] = sha256_file(out)
        run_rows.append(row)
        print(json.dumps({"event": "arm_end", **row}), flush=True)
        if completed.returncode != 0:
            atomic_json(
                args.out_dir / "matrix_state.json",
                {
                    "schema": "qwen38_hard_patch_forensic_matrix_v1",
                    "status": "failed",
                    "started_at": started.isoformat(),
                    "failed_arm": arm,
                    "arms": run_rows,
                },
            )
            raise SystemExit(completed.returncode)

    reports = [json.loads(Path(row["report"]).read_text()) for row in run_rows]
    ranking = sorted(
        (
            {
                "arm": report["arm"],
                **report["summary"],
                "last_hidden_relative_mse": report["hidden_summary_by_layer"][
                    max(report["hidden_summary_by_layer"], key=int)
                ]["relative_mse"],
                "generation_texts": [row["text"] for row in report["generations"]],
                "applied_count": (report.get("patch_application") or {}).get("applied_count", 0),
                "applied_parameters": (report.get("patch_application") or {}).get("applied_parameters", 0),
            }
            for report in reports
        ),
        key=lambda row: (float(row["teacher_topk_kl_full_mean"]), -float(row["top1_agreement"])),
    )
    finished = datetime.now(timezone.utc)
    aggregate = {
        "schema": "qwen38_hard_patch_forensic_matrix_v1",
        "status": "complete",
        "model_dir": str(args.model_dir),
        "cache": str(args.cache),
        "patch": str(args.patch),
        "patch_sha256": sha256_file(args.patch),
        "matrix": args.matrix,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "arms": run_rows,
        "ranking": ranking,
    }
    atomic_json(args.out_dir / "matrix_state.json", aggregate)
    print(json.dumps({"event": "matrix_complete", "ranking": ranking}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
