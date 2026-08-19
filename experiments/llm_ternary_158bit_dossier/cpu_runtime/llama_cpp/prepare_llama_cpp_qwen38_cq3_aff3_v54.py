#!/usr/bin/env python3
"""Compose the trained AFF3 graph seam with the accepted CQ3 AVX2 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Optional


HERE = Path(__file__).resolve().parent
CPU_RUNTIME = HERE.parent
AFF3_PREPARE = HERE / "prepare_llama_cpp_qwen38_aff3_v50.py"
LLAMA_REVISION = "4df29be4f4c3673f428170fda944a5b19f743bb8"
MODEL = "Qwen/Qwen3.8-27B"
ACCEPTED_RECEIPT = CPU_RUNTIME / (
    "receipts/qwen38_27b_cq3_avx2_x8_repack_v15/terminal-decision.json"
)
PATCHES = (
    (
        CPU_RUNTIME / "llama_cpp_4df29be_cq3.patch",
        "b5c584a338018ff7e509bc160c195693102fb6e265b6cbe4935d6e54d7cd67c1",
    ),
    (
        CPU_RUNTIME / "llama_cpp_4df29be_cq3_x86_pair_v1.patch",
        "684568733cd3cc3464fa55ba0def22ed05e33684352c1f8ee5215c95afc832ad",
    ),
    (
        CPU_RUNTIME / "kernels/llama_cpp_cq3_avx2_x8_repack_v15.patch",
        "0c0fb98d3b74686a1e42c6e4da1b6eb0b46519a3775eb2d5de21ba38d66160fc",
    ),
    (
        CPU_RUNTIME
        / "kernels/llama_cpp_cq3_avx2_x8_repack_integration_v15.patch",
        "ae86805c0bc159f8d20f55fac785e234710ca9bf80af7b9e1cb4d3e62592f41f",
    ),
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_accepted_receipt(receipt: dict) -> None:
    if receipt.get("schema") != "qwen38_27b_cq3_avx2_x8_repack_result_v15":
        raise ValueError("unexpected CQ3 x8 evidence schema")
    if receipt.get("model") != MODEL:
        raise ValueError("CQ3 x8 evidence belongs to a different model")
    if not receipt.get("terminal") or not receipt.get("infrastructure_valid"):
        raise ValueError("CQ3 x8 evidence is not terminal and infrastructure-valid")
    if not receipt.get("native_kernel_passed"):
        raise ValueError("CQ3 x8 primitive did not pass its native gates")
    provenance = receipt.get("provenance", {})
    if provenance.get("native_patch_sha256") != PATCHES[2][1]:
        raise ValueError("accepted CQ3 x8 native patch hash changed")
    if provenance.get("integration_patch_sha256") != PATCHES[3][1]:
        raise ValueError("accepted CQ3 x8 integration patch hash changed")


def run(*args: str, cwd: Optional[Path] = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True)


def authenticated_working_tree(source: Path) -> dict[str, str]:
    files: set[Path] = set()
    status = run("git", "status", "--porcelain", cwd=source)
    for line in status.splitlines():
        relative = line[3:]
        candidate = source / relative
        if candidate.is_dir():
            files.update(path for path in candidate.rglob("*") if path.is_file())
        elif candidate.is_file():
            files.add(candidate)
    return {
        str(path.relative_to(source)): sha256_path(path)
        for path in sorted(files)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-source", type=Path, required=True)
    args = parser.parse_args()
    source = args.llama_source.resolve()

    revision = run("git", "rev-parse", "HEAD", cwd=source).strip()
    status_before = run("git", "status", "--porcelain", cwd=source)
    if revision != LLAMA_REVISION or status_before:
        raise RuntimeError("llama.cpp source revision or cleanliness changed")

    receipt_bytes = ACCEPTED_RECEIPT.read_bytes()
    receipt = json.loads(receipt_bytes)
    validate_accepted_receipt(receipt)
    for patch, expected in PATCHES:
        if sha256_path(patch) != expected:
            raise RuntimeError(f"frozen patch changed: {patch.name}")

    aff3_output = subprocess.check_output(
        [
            sys.executable,
            str(AFF3_PREPARE),
            "--llama-source",
            str(source),
        ],
        text=True,
    )
    aff3_receipt = json.loads(aff3_output)
    if not aff3_receipt.get("complete") or not aff3_receipt.get("terminal"):
        raise RuntimeError("AFF3 v50 transform did not complete")

    applied = []
    for patch, expected in PATCHES:
        subprocess.check_call(["git", "apply", "--check", str(patch)], cwd=source)
        subprocess.check_call(["git", "apply", str(patch)], cwd=source)
        applied.append({"path": str(patch.relative_to(CPU_RUNTIME)), "sha256": expected})
    subprocess.check_call(["git", "diff", "--check"], cwd=source)

    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary"], cwd=source
    )
    result = {
        "schema": "qwen38_27b_cq3_aff3_llamacpp_transform_v54",
        "model": MODEL,
        "llama_revision": revision,
        "accepted_cq3_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "accepted_native_speedup": receipt["native_measurement"][
            "control_over_candidate_speedup"
        ],
        "aff3_transform": aff3_receipt,
        "patches": applied,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "working_tree_files_sha256": authenticated_working_tree(source),
        "high_precision_weight_materialization": False,
        "complete": True,
        "terminal": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
