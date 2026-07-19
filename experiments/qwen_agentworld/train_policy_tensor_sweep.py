"""Run bounded tensor-QAT over low-bit tensors selected by a policy.

The first Qwen3.5 AgentWorld pack proved the storage format, but behavior
collapsed because most low-bit groups were still PTQ tensors. This script
trains a ranked slice of those policy tensors against a cached teacher-behavior
bank and emits individual patches plus an optional merged patch for repacking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LOWBIT_PREFIXES = ("ternary", "binary")


def load_patch_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def key_aliases(tensor_name: str) -> set[str]:
    aliases = {tensor_name}
    language_prefix = "model.language_model."
    model_prefix = "model."
    if tensor_name.startswith(language_prefix):
        aliases.add(model_prefix + tensor_name[len(language_prefix) :])
    elif tensor_name.startswith(model_prefix):
        aliases.add(language_prefix + tensor_name[len(model_prefix) :])
    return aliases


def is_patched(tensor_name: str, patch_keys: set[str]) -> bool:
    return bool(key_aliases(tensor_name) & patch_keys)


def select_tensors(
    policy: dict[str, Any],
    *,
    already_patched: set[str],
    include_tensors: set[str] | None,
    exclude_tensors: set[str],
    rates: set[str] | None,
    buckets: set[str] | None,
    max_tensors: int,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for choice in policy.get("choices", []):
        rate = str(choice.get("rate", ""))
        if not rate.startswith(LOWBIT_PREFIXES):
            continue
        if rates is not None and rate not in rates:
            continue
        meta = choice.get("meta") or {}
        tensor = meta.get("tensor")
        if not tensor:
            continue
        tensor = str(tensor)
        if include_tensors is not None and tensor not in include_tensors:
            continue
        if tensor in exclude_tensors:
            continue
        if buckets is not None and str(meta.get("bucket")) not in buckets:
            continue
        if is_patched(tensor, already_patched):
            continue
        params = int(meta.get("params") or 0)
        current = rows.get(tensor)
        if current is None or params > int(current.get("params") or 0):
            rows[tensor] = {
                "tensor": tensor,
                "rate": rate,
                "bucket": meta.get("bucket"),
                "params": params,
                "source_bytes": int(meta.get("source_bytes") or (params * 2)),
            }
    selected = sorted(rows.values(), key=lambda row: int(row["source_bytes"]), reverse=True)
    return selected[:max_tensors] if max_tensors > 0 else selected


def merge_patches(paths: list[Path], out_path: Path) -> dict[str, Any]:
    tensors: dict[str, torch.Tensor] = {}
    sources: dict[str, str] = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
                sources[key] = str(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out_path)
    return {
        "out": str(out_path),
        "tensor_count": len(tensors),
        "sources": sources,
        "bytes": out_path.stat().st_size,
    }


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--base-patch", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-tensors", type=int, default=8)
    parser.add_argument("--rates")
    parser.add_argument("--buckets")
    parser.add_argument("--include-tensor", action="append", default=[])
    parser.add_argument("--exclude-tensor", action="append", default=[])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-train-tokens", type=int, default=9000)
    parser.add_argument("--max-train-rows", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--codebook-weight", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--special-token-id", action="append", type=int, default=[])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--merge-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text())
    patch_keys = load_patch_keys(args.base_patch)
    selected = select_tensors(
        policy,
        already_patched=patch_keys,
        include_tensors=set(args.include_tensor) if args.include_tensor else None,
        exclude_tensors=set(args.exclude_tensor),
        rates=parse_csv(args.rates),
        buckets=parse_csv(args.buckets),
        max_tensors=args.max_tensors,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = args.out_dir / "patches"
    report_dir = args.out_dir / "reports"
    patch_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        out = args.out_dir / "dry_run_selection.json"
        out.write_text(json.dumps({"selected": selected}, indent=2) + "\n")
        print(json.dumps({"out": str(out), "selected": selected}, indent=2))
        return

    run_rows = []
    completed_patches: list[Path] = []
    context_patches = [*args.base_patch]
    script = Path(__file__).with_name("train_cache_tensor_qat.py")
    for index, row in enumerate(selected, start=1):
        mode = "binary" if str(row["rate"]).startswith("binary") else "ternary"
        patch_path = patch_dir / f"{index:03d}_{row['tensor'].replace('.', '__')}.safetensors"
        report_path = report_dir / f"{index:03d}_{row['tensor'].replace('.', '__')}.json"
        cmd = [
            sys.executable,
            str(script),
            "--model-dir",
            str(args.model_dir),
            "--cache",
            str(args.cache),
            "--tensor",
            row["tensor"],
            "--out-patch",
            str(patch_path),
            "--out-report",
            str(report_path),
            "--mode",
            mode,
            "--group-size",
            str(args.group_size),
            "--max-train-tokens",
            str(args.max_train_tokens),
            "--max-train-rows",
            str(args.max_train_rows),
            "--steps",
            str(args.steps),
            "--lr",
            str(args.lr),
            "--codebook-weight",
            str(args.codebook_weight),
            "--temperature",
            str(args.temperature),
            "--seed",
            str(args.seed + index),
        ]
        for special_id in args.special_token_id:
            cmd.extend(["--special-token-id", str(special_id)])
        for context_patch in context_patches:
            cmd.extend(["--context-weight-patch", str(context_patch)])

        started = {
            **row,
            "index": index,
            "mode": mode,
            "patch": str(patch_path),
            "report": str(report_path),
            "command": cmd,
        }
        print(json.dumps({"event": "start", **started}), flush=True)
        proc = subprocess.run(cmd)
        finished = {**started, "returncode": proc.returncode}
        if proc.returncode == 0:
            completed_patches.append(patch_path)
            context_patches.append(patch_path)
        run_rows.append(finished)
        (args.out_dir / "sweep_status.json").write_text(
            json.dumps(
                {
                    "selected": selected,
                    "runs": run_rows,
                    "completed_patches": [str(path) for path in completed_patches],
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"event": "done", **finished}), flush=True)
        if proc.returncode != 0 and args.stop_on_failure:
            break

    merge_result = None
    if args.merge_out is not None:
        merge_result = merge_patches([*args.base_patch, *completed_patches], args.merge_out)

    result = {
        "policy": str(args.policy),
        "base_patches": [str(path) for path in args.base_patch],
        "selected": selected,
        "runs": run_rows,
        "completed_patches": [str(path) for path in completed_patches],
        "merge": merge_result,
    }
    out = args.out_dir / "sweep_result.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(out), "completed": len(completed_patches), "merge": merge_result}, indent=2))


if __name__ == "__main__":
    main()
