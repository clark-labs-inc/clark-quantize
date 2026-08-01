"""Deterministic source- and language-balanced recovery sampling."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from data_pipeline import discover_images
from recovery_corpus_identity import recovery_corpus_identity
from recovery_resume_policy import sampler_resume_mode
from run_state import restore_run_state


SAMPLER_FORMAT = "backdropkit-product-source-language-v3"
PRODUCT_SOURCE_WEIGHTS = {
    "rico": 0.35,
    "aria_ui": 0.15,
    "websight": 0.20,
    "flores200_balanced_synthetic_ui": 0.20,
    "pd12m": 0.10,
}


@dataclass(frozen=True)
class Record:
    path: Path
    source: str
    language: str | None


def _quotas(total: int, labels: list[str]) -> dict[str, int]:
    if total < len(labels):
        raise ValueError("balanced epoch is smaller than its strata")
    base, remainder = divmod(total, len(labels))
    return {
        label: base + (index < remainder)
        for index, label in enumerate(sorted(labels))
    }


def _weighted_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    if not weights or any(weight <= 0 for weight in weights.values()):
        raise ValueError("source weights must be positive")
    denominator = sum(weights.values())
    exact = {
        label: total * weight / denominator
        for label, weight in weights.items()
    }
    quotas = {label: int(value) for label, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda label: (exact[label] - quotas[label], label),
        reverse=True,
    )
    for label in order[:remainder]:
        quotas[label] += 1
    return quotas


def _cycled_sample(paths: list[Path], count: int, seed: int) -> list[Path]:
    if not paths:
        raise ValueError("cannot sample an empty recovery stratum")
    sampled = []
    cycle = 0
    while len(sampled) < count:
        ordered = paths.copy()
        random.Random(seed + cycle * 1_000_003).shuffle(ordered)
        sampled.extend(ordered[: count - len(sampled)])
        cycle += 1
    return sampled


def load_records(roots: list[Path]) -> list[Record]:
    records = []
    for root in roots:
        manifest = root / "manifest.jsonl"
        if manifest.exists():
            with manifest.open() as rows:
                for line in rows:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    path = root / row["local_path"]
                    if not path.is_file():
                        raise FileNotFoundError(f"manifest image is missing: {path}")
                    records.append(Record(
                        path,
                        str(row.get("source") or root.name),
                        row.get("language"),
                    ))
        else:
            records.extend(
                Record(path, root.name, None)
                for path in discover_images([root])
            )
    if not records:
        raise ValueError("recovery corpus contains no images")
    return records


class BalancedSourceSampler:
    def __init__(
        self,
        records: list[Record],
        seed: int,
        corpus_identity: dict,
    ) -> None:
        self.seed = seed
        self.total = len(records)
        self.corpus_identity = corpus_identity
        self.sources = defaultdict(list)
        for record in records:
            self.sources[record.source].append(record)
        if set(self.sources) == set(PRODUCT_SOURCE_WEIGHTS):
            self.source_weights = PRODUCT_SOURCE_WEIGHTS
            self.source_quotas = _weighted_quotas(
                self.total,
                PRODUCT_SOURCE_WEIGHTS,
            )
        else:
            self.source_weights = {
                source: 1 / len(self.sources)
                for source in self.sources
            }
            self.source_quotas = _quotas(self.total, list(self.sources))
        self.language_counts = {
            source: len({
                record.language
                for record in source_records
                if record.language
            })
            for source, source_records in self.sources.items()
        }

    @classmethod
    def from_roots(cls, roots: list[Path], seed: int):
        return cls(
            load_records(roots),
            seed,
            recovery_corpus_identity(roots),
        )

    def __len__(self) -> int:
        return self.total

    @property
    def metadata(self) -> dict:
        return {
            "format": SAMPLER_FORMAT,
            "epoch_images": self.total,
            "source_weights": dict(sorted(self.source_weights.items())),
            "source_quotas": dict(sorted(self.source_quotas.items())),
            "languages_per_source": dict(sorted(self.language_counts.items())),
            "corpus_identity": self.corpus_identity,
        }

    def _source_paths(self, source: str, epoch: int) -> list[Path]:
        records = self.sources[source]
        quota = self.source_quotas[source]
        languages = defaultdict(list)
        unlabelled = []
        for record in records:
            if record.language:
                languages[record.language].append(record.path)
            else:
                unlabelled.append(record.path)
        source_seed = self.seed + epoch * 10_000_019 + sum(map(ord, source))
        if not languages:
            return _cycled_sample(unlabelled, quota, source_seed)
        if unlabelled:
            raise ValueError(f"source mixes labelled and unlabelled images: {source}")
        language_quotas = _quotas(quota, list(languages))
        sampled = []
        for language in sorted(languages):
            language_seed = source_seed + sum(map(ord, language)) * 1009
            sampled.extend(_cycled_sample(
                languages[language],
                language_quotas[language],
                language_seed,
            ))
        return sampled

    def paths(self, epoch: int) -> list[Path]:
        sampled = []
        for source in sorted(self.sources):
            sampled.extend(self._source_paths(source, epoch))
        if len(sampled) != self.total:
            raise RuntimeError("balanced recovery epoch has the wrong length")
        random.Random(self.seed + epoch).shuffle(sampled)
        return sampled

    def restore(
        self,
        checkpoint: dict,
        augmentation_rng: random.Random,
        examples_seen: int,
        reset_data_stream: bool = False,
    ) -> tuple[int, int, bool]:
        sampler = (
            checkpoint.get("recovery", {}).get("sampler")
            or checkpoint.get("surface_training", {}).get("sampler")
        )
        mode = sampler_resume_mode(
            sampler,
            self.metadata,
            reset_data_stream,
        )
        if mode != "exact":
            return 0, 0, False
        epoch, cursor, _elapsed, exact = restore_run_state(
            checkpoint,
            augmentation_rng,
            examples_seen,
            self.total,
        )
        return epoch, cursor, exact
