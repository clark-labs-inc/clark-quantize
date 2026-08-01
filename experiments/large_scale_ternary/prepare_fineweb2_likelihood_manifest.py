"""Freeze an equal-per-language FineWeb2 likelihood gate with exact row hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_b_protocol import digest_json, row_digest, seal_manifest

DATASET = "HuggingFaceFW/fineweb-2"
PINNED_REVISION = "66d1a3b989bad0c42b88179a7fb60c47ce87af74"
LIKELIHOOD_CONTRACT = {
    "version": "fineweb2-causal-likelihood-v1",
    "max_tokens": 512,
    "minimum_tokens": 32,
    "label_shift": 1,
    "reduction": "token_micro_mean",
}


def discover_test_universe(revision: str) -> list[dict[str, str]]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(
        DATASET, revision=revision, files_metadata=False
    )
    if info.sha != revision:
        raise ValueError(f"resolved revision {info.sha} != requested {revision}")
    records = []
    for sibling in info.siblings:
        path = sibling.rfilename
        parts = path.split("/")
        if (
            len(parts) == 4
            and parts[0] == "data"
            and parts[2] == "test"
            and path.endswith(".parquet")
            and not parts[1].endswith("_removed")
        ):
            records.append({"config": parts[1], "path": path})
    records.sort(key=lambda row: (row["config"], row["path"]))
    if not records:
        raise ValueError("no FineWeb2 test shards discovered")
    configs = [row["config"] for row in records]
    if len(configs) != len(set(configs)):
        raise ValueError("expected exactly one test shard per language config")
    return records


def sample_config(
    config: str,
    *,
    revision: str,
    samples_per_language: int,
    seed: int,
) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET,
        name=config,
        split="test",
        revision=revision,
        streaming=True,
    )
    random_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{config}".encode()).digest()[:4], "big"
    )
    rng = random.Random(random_seed)
    reservoir = []
    for index, row in enumerate(dataset):
        sample = {
            "sample_id": f"fineweb2:{config}:{index}",
            "config": config,
            "stratum": config,
            "index": index,
            "row_sha256": row_digest(row),
        }
        if len(reservoir) < samples_per_language:
            reservoir.append(sample)
        else:
            replacement = rng.randint(0, index)
            if replacement < samples_per_language:
                reservoir[replacement] = sample
    if len(reservoir) != samples_per_language:
        raise ValueError(
            f"{config} yielded {len(reservoir)} of "
            f"{samples_per_language} samples"
        )
    reservoir.sort(key=lambda row: row["index"])
    return {
        "config": config,
        "source_contract_sha256": digest_json(
            {
                "dataset": DATASET,
                "revision": revision,
                "config": config,
                "split": "test",
            }
        ),
        "samples": reservoir,
    }


def read_progress(path: Path) -> dict[str, dict]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        row = json.loads(line)
        completed[row["config"]] = row
    return completed


def append_progress(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--samples-per-language", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--limit-languages", type=int)
    parser.add_argument("--universe-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    universe = discover_test_universe(args.revision)
    if args.limit_languages:
        universe = universe[: args.limit_languages]
    universe_payload = {
        "schema_version": 1,
        "dataset": DATASET,
        "revision": args.revision,
        "test_language_configs": len(universe),
        "shards": universe,
    }
    universe_payload["content_sha256"] = digest_json(universe_payload)
    if args.universe_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(universe_payload, indent=2) + "\n")
        print(
            f"wrote {args.out}: {len(universe)} language-script test configs"
        )
        return

    progress_path = args.out.with_suffix(args.out.suffix + ".progress.jsonl")
    completed = read_progress(progress_path)
    for position, entry in enumerate(universe, start=1):
        config = entry["config"]
        if config in completed:
            continue
        record = sample_config(
            config,
            revision=args.revision,
            samples_per_language=args.samples_per_language,
            seed=args.seed,
        )
        append_progress(progress_path, record)
        completed[config] = record
        print(f"{position}/{len(universe)} {config}", flush=True)

    ordered = [completed[entry["config"]] for entry in universe]
    samples = [
        sample for config in ordered for sample in config["samples"]
    ]
    task = {
        "task": "fineweb2_likelihood",
        "dataset": DATASET,
        "config": "equal_language_test_splits",
        "split": "test",
        "revision": args.revision,
        "fingerprint": digest_json(
            {
                row["config"]: row["source_contract_sha256"]
                for row in ordered
            }
        ),
        "universe_sha256": universe_payload["content_sha256"],
        "dataset_size": None,
        "shots": 0,
        "sampling": {
            "method": "equal_per_language_seeded_reservoir",
            "seed": args.seed,
            "samples_per_language": args.samples_per_language,
            "language_configs": len(ordered),
        },
        "likelihood_contract": LIKELIHOOD_CONTRACT,
        "config_fingerprints": {
            row["config"]: row["source_contract_sha256"] for row in ordered
        },
        "samples": samples,
    }
    manifest = seal_manifest(
        [task],
        args.seed,
        prompt_contract_hash=digest_json(LIKELIHOOD_CONTRACT),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(
        f"wrote {args.out}: {len(samples)} documents across "
        f"{len(ordered)} language-script configs"
    )


if __name__ == "__main__":
    main()
