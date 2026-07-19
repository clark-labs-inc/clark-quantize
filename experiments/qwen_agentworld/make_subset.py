"""Create a deterministic domain-balanced AgentWorldBench subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--per-domain", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    if args.per_domain <= 0:
        raise ValueError("--per-domain must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": str(args.data_dir),
        "per_domain": args.per_domain,
        "offset": args.offset,
        "files": [],
    }
    for src in sorted(args.data_dir.glob("*_test.jsonl")):
        with src.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        selected = rows[args.offset : args.offset + args.per_domain]
        dst = args.out_dir / src.name
        with dst.open("w") as f:
            for row in selected:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["files"].append({"file": src.name, "selected": len(selected), "available": len(rows)})

    (args.out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
