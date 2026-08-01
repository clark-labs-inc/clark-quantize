#!/usr/bin/env python3
"""Reconstruct byte-exact sealed sources from a manifest after live edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from full_surface_provenance import file_sha256


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def remove_once(text: str, block: str) -> str:
    if text.count(block) != 1:
        raise ValueError(f"expected one historical patch block, found {text.count(block)}")
    return text.replace(block, "", 1)


def replace_once(text: str, current: str, historical: str) -> str:
    if text.count(current) != 1:
        raise ValueError(f"expected one historical replacement, found {text.count(current)}")
    return text.replace(current, historical, 1)


def reconstruct(relative: str, text: str) -> str | None:
    if relative.endswith("/surface_quantization.py"):
        start = text.index("\ndef apply_surface_group_sizes")
        end = text.index("\ndef set_surface_progress")
        text = text[:start] + text[end:]
        text = remove_once(
            text,
            "    convolution_learning_rate_multiplier: float = 0.25,\n",
        )
        text = remove_once(
            text,
            "    if convolution_learning_rate_multiplier <= 0:\n"
            "        raise ValueError("
            "\"convolution learning-rate multiplier must be positive\")\n",
        )
        return replace_once(
            text,
            '            "lr": learning_rate * '
            "convolution_learning_rate_multiplier,\n",
            '            "lr": learning_rate * 0.25,\n',
        )
    if relative.endswith("/surface_model_state.py"):
        text = remove_once(text, "    apply_surface_group_sizes,\n")
        text = remove_once(
            text,
            '    surface_group_sizes = dict(\n'
            '        quantization.get("surface_group_sizes", {}) or {}\n'
            '    )\n',
        )
        text = remove_once(
            text,
            "    args.surface_group_sizes = surface_group_sizes\n",
        )
        text = remove_once(
            text,
            '    previous_training = payload.get("surface_training", {})\n'
            "    previous_conv_lr_multiplier = float(\n"
            "        previous_training.get(\n"
            '            "convolution_learning_rate_multiplier",\n'
            "            0.25,\n"
            "        )\n"
            "    )\n"
            "    requested_conv_lr_multiplier = getattr(\n"
            "        args,\n"
            '        "convolution_learning_rate_multiplier",\n'
            "        None,\n"
            "    )\n"
            "    args.convolution_learning_rate_multiplier = (\n"
            "        previous_conv_lr_multiplier\n"
            "        if resuming and requested_conv_lr_multiplier is None\n"
            "        else 0.25\n"
            "        if requested_conv_lr_multiplier is None\n"
            "        else requested_conv_lr_multiplier\n"
            "    )\n",
        )
        while "        apply_surface_group_sizes(surfaces, surface_group_sizes)\n" in text:
            text = text.replace(
                "        apply_surface_group_sizes(surfaces, surface_group_sizes)\n",
                "",
                1,
            )
        return text
    if relative.endswith("/surface_run_state.py"):
        text = remove_once(
            text,
            '    previous_surface_group_sizes = dict(\n'
            '        previous.get("surface_group_sizes", {}) or {}\n'
            "    )\n"
            "    requested_surface_group_sizes = dict(\n"
            '        getattr(args, "surface_group_sizes", {}) or {}\n'
            "    )\n"
            "    if previous_surface_group_sizes != requested_surface_group_sizes:\n"
            "        raise ValueError(\n"
            '            "cannot resume with changed surface_group_sizes: "\n'
            '            f"{previous_surface_group_sizes!r} -> "\n'
            '            f"{requested_surface_group_sizes!r}"\n'
            "        )\n"
            "    previous_conv_lr_multiplier = float(\n"
            '        previous.get("convolution_learning_rate_multiplier", 0.25)\n'
            "    )\n"
            "    requested_conv_lr_multiplier = float(\n"
            '        getattr(args, "convolution_learning_rate_multiplier", 0.25)\n'
            "    )\n"
            "    if previous_conv_lr_multiplier != requested_conv_lr_multiplier:\n"
            "        raise ValueError(\n"
            '            "cannot resume with changed convolution_learning_rate_multiplier: "\n'
            '            f"{previous_conv_lr_multiplier!r} -> "\n'
            '            f"{requested_conv_lr_multiplier!r}"\n'
            "        )\n",
        )
        text = remove_once(
            text,
            "    surface_group_sizes = dict(sorted(\n"
            '        (config.get("surface_group_sizes", {}) or {}).items()\n'
            "    ))\n",
        )
        return remove_once(
            text,
            '            "surface_group_sizes": surface_group_sizes,\n',
        )
    if relative.endswith("/train_swin2sr_full_surface.py"):
        text = remove_once(
            text,
            "    parser.add_argument(\n"
            '        "--convolution-learning-rate-multiplier",\n'
            "        type=float,\n"
            "    )\n",
        )
        text = remove_once(
            text,
            "        convolution_learning_rate_multiplier=(\n"
            "            args.convolution_learning_rate_multiplier\n"
            "        ),\n",
        )
        block = (
            '        "surface_group_sizes": dict(\n'
            '            sorted((getattr(args, "surface_group_sizes", {}) or {}).items())\n'
            "        ),\n"
            '        "convolution_learning_rate_multiplier": (\n'
            "            args.convolution_learning_rate_multiplier\n"
            "        ),\n"
        )
        if text.count(block) != 2:
            raise ValueError(
                f"expected two trainer metadata blocks, found {text.count(block)}"
            )
        return text.replace(block, "", 2)
    return None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    repository = Path(__file__).resolve().parents[2]
    search_roots = (
        repository / "clark_air",
        repository / "clark_quantize",
        repository / "experiments/browser_swin2sr",
    )
    expected = manifest["source_files"]
    exact_sources: dict[str, Path] = {}
    wanted = set(expected.values())
    for root in search_roots:
        for candidate in root.rglob("*.py"):
            digest = file_sha256(candidate)
            if digest in wanted and digest not in exact_sources:
                exact_sources[digest] = candidate

    prepared: dict[str, tuple[bytes, str]] = {}
    for relative, digest in expected.items():
        destination = args.out / relative
        if destination.exists() and file_sha256(destination) == digest:
            prepared[relative] = (destination.read_bytes(), "already-sealed")
            continue
        source = exact_sources.get(digest)
        if source is not None:
            prepared[relative] = (source.read_bytes(), f"exact-copy:{source}")
            continue
        reconstructed = reconstruct(
            relative,
            (repository / relative).read_text(),
        )
        if reconstructed is None:
            raise RuntimeError(f"no exact source or reconstruction for {relative}")
        prepared[relative] = (reconstructed.encode(), "inverse-known-patch")

    for relative, (data, method) in prepared.items():
        digest = sha256(data)
        if digest != expected[relative]:
            raise RuntimeError(
                f"reconstructed hash mismatch: {relative}: {digest}"
            )
        destination = args.out / relative
        if destination.exists():
            if destination.read_bytes() != data:
                raise RuntimeError(f"refusing to overwrite sealed source: {relative}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            output.write(data)

    receipt = {
        "format": "backdropkit-sealed-source-reconstruction-v1",
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "sources": {
            relative: {
                "sha256": expected[relative],
                "method": prepared[relative][1],
            }
            for relative in sorted(expected)
        },
    }
    receipt_path = args.out / "reconstruction-receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"receipt already exists: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "receipt": str(receipt_path),
        "sources": len(prepared),
        "hashes_verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()
