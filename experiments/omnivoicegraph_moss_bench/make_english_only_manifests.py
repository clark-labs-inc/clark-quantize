#!/usr/bin/env python3
"""Materialize deterministic English-only subsets from MOSS-style manifests.

This is intentionally a conservative script: a row is eligible only when every
alphabetic reference character belongs to the Unicode Latin script and there is
at least one alphabetic character.  This excludes Chinese, Japanese, Korean,
Cyrillic, Thai, and mixed-script rows without pretending that a script check is
a full language-identification model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


def script_name(ch: str) -> str:
    name = unicodedata.name(ch, "")
    if "LATIN" in name:
        return "Latin"
    if any(token in name for token in ("CJK", "IDEOGRAPH", "HIRAGANA", "KATAKANA")):
        return "CJK"
    if "HANGUL" in name:
        return "Hangul"
    if "CYRILLIC" in name:
        return "Cyrillic"
    if "THAI" in name:
        return "Thai"
    return name.split()[0] if name else "Other"


def reference_scripts(text: str) -> set[str]:
    return {script_name(ch) for ch in text if ch.isalpha()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: list[dict] = []
    excluded = Counter()
    for row in rows:
        scripts = reference_scripts(str(row.get("reference", "")))
        if scripts == {"Latin"}:
            selected.append(row)
        elif not scripts:
            excluded["no_alphabetic_reference"] += 1
        else:
            excluded["non_latin_script:" + "+".join(sorted(scripts))] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    receipt = {
        "schema": "omnivoicegraph.english_only_manifest_receipt.v1",
        "source_manifest": str(args.input),
        "source_manifest_sha256": sha256(args.input),
        "output_manifest": str(args.output),
        "output_manifest_sha256": sha256(args.output),
        "rows_in": len(rows),
        "rows_out": len(selected),
        "filter": {
            "policy": "all alphabetic reference characters must be Unicode Latin and at least one alphabetic character is required",
            "note": "This is a conservative script gate, not a full language-identification claim.",
        },
        "excluded": dict(sorted(excluded.items())),
    }
    receipt_path = args.output.with_name(args.output.stem + "_receipt.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows_in": len(rows), "rows_out": len(selected), "excluded": dict(excluded)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
