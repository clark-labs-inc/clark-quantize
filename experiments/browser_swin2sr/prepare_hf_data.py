#!/usr/bin/env python3
"""Build a pinned, licensed Swin2SR training corpus from Hugging Face.

The materialized images are a bounded local cache. ``manifest.jsonl`` records
the exact repository revision, upstream shard, license, record ID, dimensions,
and content hash for every accepted sample.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


SOURCES = {
    "aria_ui": {
        "repo": "Aria-UI/Aria-UI_Data",
        "revision": "459c849ec0d5fd1822439f431941ebc46967533e",
        "license": "Apache-2.0",
        "file": "desktop/screenshots_fix_batch2.zip",
        "limit": 3000,
    },
    "rico": {
        "repo": "bevaya/RICO-Screen2Words",
        "revision": "fcd10c2cda2538ed71b1028984675fa29854283b",
        "license": "CC-BY-4.0",
        "file_prefix": "data/train-",
        "shards": 8,
        "limit": 15_500,
    },
    "websight": {
        "repo": "HuggingFaceM4/WebSight",
        "revision": "b11f8172f89c992b56ac702319e02c428cca4a4e",
        "license": "CC-BY-4.0",
        "file_prefix": "data/train-",
        "shards": 9,
        "limit": 84_000,
        "version": "v0.1; synthetic HTML/CSS without third-party images",
    },
    "pd12m": {
        "repo": "Spawning/PD12M",
        "revision": "867988b01138799b89d3ffdd5b4f7e1455951f32",
        "license": "CDLA-Permissive-2.0; source images public domain or CC0",
        "file": "metadata/pd12m.000.parquet",
        "limit": 2000,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".cache/clark-quantize/swin2sr-hf-v1",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--minimum-side", type=int, default=256)
    parser.add_argument("--pd-workers", type=int, default=12)
    parser.add_argument(
        "--skip-pd12m",
        action="store_true",
        help="Materialize only the UI sources; useful for a fast data-pipeline smoke.",
    )
    return parser.parse_args()


def image_metadata(data: bytes, minimum_side: int) -> tuple[str, int, int] | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "PNG").lower()
    except (OSError, ValueError):
        return None
    if min(width, height) < minimum_side:
        return None
    extension = {"jpeg": "jpg", "png": "png", "webp": "webp"}.get(image_format)
    if extension is None:
        return None
    return extension, width, height


class CorpusWriter:
    def __init__(self, root: Path, minimum_side: int) -> None:
        self.root = root
        self.minimum_side = minimum_side
        self.seen: set[str] = set()
        self.records: list[dict] = []
        partial = root / "manifest.partial.jsonl"
        complete = root / "manifest.jsonl"
        existing = partial if partial.exists() else complete
        if existing.exists():
            for line in existing.read_text().splitlines():
                record = json.loads(line)
                self.records.append(record)
                self.seen.add(record["sha256"])
        self.partial = partial.open("a")

    def add(
        self,
        source: str,
        data: bytes,
        upstream_file: str,
        record_id: str,
        extra: dict | None = None,
    ) -> bool:
        metadata = image_metadata(data, self.minimum_side)
        if metadata is None:
            return False
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.seen:
            return False
        extension, width, height = metadata
        source_dir = self.root / source
        source_dir.mkdir(parents=True, exist_ok=True)
        destination = source_dir / f"{digest[:20]}.{extension}"
        destination.write_bytes(data)
        definition = SOURCES[source]
        record = {
            "source": source,
            "repo": definition["repo"],
            "revision": definition["revision"],
            "license": definition["license"],
            "upstream_file": upstream_file,
            "record_id": record_id,
            "sha256": digest,
            "width": width,
            "height": height,
            "local_path": str(destination.relative_to(self.root)),
        }
        if extra:
            record.update(extra)
        self.records.append(record)
        self.seen.add(digest)
        self.partial.write(json.dumps(record, sort_keys=True) + "\n")
        self.partial.flush()
        return True

    def finish(self) -> None:
        self.partial.close()
        ordered = sorted(self.records, key=lambda item: (item["source"], item["record_id"]))
        with (self.root / "manifest.jsonl").open("w") as handle:
            for record in ordered:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        counts: dict[str, int] = {}
        for record in ordered:
            counts[record["source"]] = counts.get(record["source"], 0) + 1
        (self.root / "summary.json").write_text(
            json.dumps(
                {
                    "images": len(ordered),
                    "counts": counts,
                    "minimum_side": self.minimum_side,
                    "sources": SOURCES,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (self.root / "manifest.partial.jsonl").unlink(missing_ok=True)


def source_files(source: str) -> list[str]:
    definition = SOURCES[source]
    if "file" in definition:
        return [definition["file"]]
    files = HfApi().list_repo_files(
        definition["repo"],
        repo_type="dataset",
        revision=definition["revision"],
    )
    matches = sorted(name for name in files if name.startswith(definition["file_prefix"]))
    return matches[: definition["shards"]]


def fetch_source(source: str, filename: str) -> Path:
    definition = SOURCES[source]
    return Path(
        hf_hub_download(
            repo_id=definition["repo"],
            repo_type="dataset",
            filename=filename,
            revision=definition["revision"],
        )
    )


def add_aria(writer: CorpusWriter, rng: random.Random) -> None:
    accepted = sum(record["source"] == "aria_ui" for record in writer.records)
    if accepted >= SOURCES["aria_ui"]["limit"]:
        return
    filename = source_files("aria_ui")[0]
    path = fetch_source("aria_ui", filename)
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".png")]
        rng.shuffle(names)
        for name in names:
            if writer.add("aria_ui", archive.read(name), filename, name):
                accepted += 1
            if accepted >= SOURCES["aria_ui"]["limit"]:
                break


def add_parquet_images(writer: CorpusWriter, source: str, id_field: str | None) -> None:
    accepted = sum(record["source"] == source for record in writer.records)
    if accepted >= SOURCES[source]["limit"]:
        return
    for filename in source_files(source):
        path = fetch_source(source, filename)
        columns = ["image"] + ([id_field] if id_field else [])
        table = pq.read_table(path, columns=columns)
        for row_index, record in enumerate(table.to_pylist()):
            image = record["image"]
            data = image["bytes"] if image else None
            record_id = str(record[id_field]) if id_field else f"{filename}:{row_index}"
            if data and writer.add(source, data, filename, record_id):
                accepted += 1
            if accepted >= SOURCES[source]["limit"]:
                return


def fetch_pd_image(record: dict) -> tuple[dict, bytes | None]:
    request = urllib.request.Request(
        record["url"],
        headers={"User-Agent": "ClarkQuantizeData/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if int(response.headers.get("Content-Length", 0) or 0) > 20_000_000:
                return record, None
            return record, response.read(20_000_001)
    except (OSError, ValueError):
        return record, None


def add_pd12m(writer: CorpusWriter, rng: random.Random, workers: int) -> None:
    filename = source_files("pd12m")[0]
    path = fetch_source("pd12m", filename)
    records = [
        record
        for record in pq.read_table(path).to_pylist()
        if min(record["width"], record["height"]) >= writer.minimum_side
        and max(record["width"], record["height"]) <= 8192
        and record["mime_type"] in {"image/jpeg", "image/png", "image/webp"}
    ]
    rng.shuffle(records)
    accepted = sum(record["source"] == "pd12m" for record in writer.records)
    if accepted >= SOURCES["pd12m"]["limit"]:
        return
    cursor = 0
    while accepted < SOURCES["pd12m"]["limit"] and cursor < len(records):
        batch = records[cursor : cursor + workers * 4]
        cursor += len(batch)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_pd_image, record) for record in batch]
            for future in as_completed(futures):
                record, data = future.result()
                if data and len(data) <= 20_000_000 and writer.add(
                    "pd12m",
                    data,
                    filename,
                    record["id"],
                    {
                        "upstream_hash": record["hash"],
                        "upstream_license": record["license"],
                        "upstream_source": record["source"],
                    },
                ):
                    accepted += 1
                if accepted >= SOURCES["pd12m"]["limit"]:
                    break


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    writer = CorpusWriter(args.out, args.minimum_side)
    add_aria(writer, rng)
    add_parquet_images(writer, "rico", "screenId")
    add_parquet_images(writer, "websight", None)
    if not args.skip_pd12m:
        add_pd12m(writer, rng, args.pd_workers)
    writer.finish()
    print((args.out / "summary.json").read_text())


if __name__ == "__main__":
    main()
