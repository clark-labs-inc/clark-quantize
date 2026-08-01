"""Pure helpers for a reproducible, fail-closed large-model evaluation gate."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
VALID_ARMS = {"bf16", "qat_ternary", "ptq_ternary", "int4"}
PROMPT_CONTRACT_VERSION = "gate-b-prompts-v1"
GSM8K_SHOTS = [
    (
        "Natalia sold clips to 48 friends in April, and then she sold half as "
        "many clips in May. How many clips did she sell altogether in April and May?",
        "In April she sold 48. In May she sold 48/2 = 24. Altogether 48+24 = 72.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?",
        "Per minute she earns 12/60 = $0.2. For 50 minutes she earned 0.2*50 = $10.\n#### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only "
        "half of the money she needs. Her parents decided to give her $15 for "
        "that purpose, and her grandparents twice as much as her parents. How "
        "much more money does Betty need?",
        "Betty has 100/2 = $50. Grandparents give 2*15 = $30. Now she has "
        "50+15+30 = $95. She needs 100-95 = $5 more.\n#### 5",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How "
        "many pages does he write a year?",
        "Each time he writes 3*2 = 6 pages. Twice a week is 6*2 = 12 pages. "
        "In a year 12*52 = 624 pages.\n#### 624",
    ),
]


def prompt_contract_sha256() -> str:
    return digest_json(
        {
            "version": PROMPT_CONTRACT_VERSION,
            "gsm8k_shots": GSM8K_SHOTS,
            "mmlu": {
                "letters": ["A", "B", "C", "D"],
                "header": "The following are multiple choice questions (with answers) about {subject}.",
                "answer_suffix": "Answer:",
            },
        }
    )


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def row_digest(row: Mapping[str, Any]) -> str:
    return digest_json(dict(row))


def _seed_for(seed: int, label: str) -> int:
    material = f"{seed}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def stratified_indices(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
    stratum_key: str,
) -> list[int]:
    """Return deterministic, near-equal samples per stratum.

    The allocator repeatedly gives one slot to every non-empty stratum in
    sorted order, then shuffles within each stratum using an independent seed.
    This prevents a large subject from dominating a small MMLU gate.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(rows):
        raise ValueError(f"limit {limit} exceeds dataset size {len(rows)}")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if stratum_key not in row:
            raise KeyError(f"row {index} is missing stratum key {stratum_key!r}")
        groups[str(row[stratum_key])].append(index)
    if not groups:
        raise ValueError("cannot sample an empty dataset")
    for label, indices in groups.items():
        random.Random(_seed_for(seed, label)).shuffle(indices)

    selected: list[int] = []
    offsets = {label: 0 for label in groups}
    labels = sorted(groups)
    while len(selected) < limit:
        added = False
        for label in labels:
            offset = offsets[label]
            if offset >= len(groups[label]):
                continue
            selected.append(groups[label][offset])
            offsets[label] = offset + 1
            added = True
            if len(selected) == limit:
                break
        if not added:
            raise RuntimeError("stratified allocator exhausted unexpectedly")
    return selected


def random_indices(size: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit > size:
        raise ValueError(f"limit must be in [1, {size}], got {limit}")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices[:limit]


def make_task_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    dataset: str,
    config: str,
    split: str,
    revision: str,
    fingerprint: str,
    limit: int,
    seed: int,
    shots: int,
    stratum_key: str | None = None,
) -> dict[str, Any]:
    indices = (
        stratified_indices(rows, limit, seed, stratum_key)
        if stratum_key
        else random_indices(len(rows), limit, seed)
    )
    samples = [
        {
            "sample_id": f"{task}:{index}",
            "index": index,
            "row_sha256": row_digest(rows[index]),
            **(
                {"stratum": str(rows[index][stratum_key])}
                if stratum_key is not None
                else {}
            ),
        }
        for index in indices
    ]
    return {
        "task": task,
        "dataset": dataset,
        "config": config,
        "split": split,
        "revision": revision,
        "fingerprint": fingerprint,
        "shots": shots,
        "sampling": {
            "method": "stratified_equal" if stratum_key else "seeded_random",
            "seed": seed,
            "stratum_key": stratum_key,
        },
        "dataset_size": len(rows),
        "samples": samples,
    }


def make_stratified_support_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    config: str,
    split: str,
    revision: str,
    fingerprint: str,
    shots_per_stratum: int,
    stratum_key: str,
) -> dict[str, Any]:
    if shots_per_stratum < 0:
        raise ValueError("shots_per_stratum must be non-negative")
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        label = str(row[stratum_key])
        if len(by_stratum[label]) < shots_per_stratum:
            by_stratum[label].append(
                {"index": index, "row_sha256": row_digest(row)}
            )
    undersized = [
        label
        for label, samples in by_stratum.items()
        if len(samples) != shots_per_stratum
    ]
    if undersized:
        raise ValueError(f"not enough support rows for strata {undersized[:8]}")
    return {
        "dataset": dataset,
        "config": config,
        "split": split,
        "revision": revision,
        "fingerprint": fingerprint,
        "stratum_key": stratum_key,
        "shots_per_stratum": shots_per_stratum,
        "by_stratum": dict(sorted(by_stratum.items())),
    }


def select_and_verify_support_rows(
    rows: Sequence[Mapping[str, Any]], support_manifest: Mapping[str, Any]
) -> dict[str, list[Mapping[str, Any]]]:
    selected: dict[str, list[Mapping[str, Any]]] = {}
    for label, samples in support_manifest["by_stratum"].items():
        selected[label] = []
        for sample in samples:
            index = int(sample["index"])
            if index < 0 or index >= len(rows):
                raise ValueError(f"support index {index} is outside dataset")
            row = rows[index]
            if str(row[support_manifest["stratum_key"]]) != label:
                raise ValueError(f"support stratum drift at index {index}")
            actual = row_digest(row)
            if actual != sample["row_sha256"]:
                raise ValueError(f"support dataset drift at index {index}")
            selected[label].append(row)
    return selected


def seal_manifest(
    tasks: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    prompt_contract_hash: str | None = None,
) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "tasks": list(tasks),
    }
    if prompt_contract_hash is not None:
        body["prompt_contract_sha256"] = prompt_contract_hash
    body["content_sha256"] = digest_json(
        {
            key: value
            for key, value in body.items()
            if key not in {"created_at", "content_sha256"}
        }
    )
    return body


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Gate B manifest schema")
    expected = manifest.get("content_sha256")
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "content_sha256"}
    }
    actual = digest_json(body)
    if expected != actual:
        raise ValueError(f"manifest content hash mismatch: {expected} != {actual}")
    seen: set[str] = set()
    for task in manifest.get("tasks", []):
        if not task.get("revision"):
            raise ValueError(f"{task.get('task', 'task')} has no dataset revision")
        for sample in task.get("samples", []):
            sample_id = sample["sample_id"]
            if sample_id in seen:
                raise ValueError(f"duplicate sample id {sample_id}")
            seen.add(sample_id)


def select_and_verify_rows(
    rows: Sequence[Mapping[str, Any]], task_manifest: Mapping[str, Any]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    selected = []
    for sample in task_manifest["samples"]:
        index = int(sample["index"])
        if index < 0 or index >= len(rows):
            raise ValueError(f"sample index {index} is outside dataset")
        row = rows[index]
        actual = row_digest(row)
        if actual != sample["row_sha256"]:
            raise ValueError(
                f"dataset drift for {sample['sample_id']}: "
                f"{actual} != {sample['row_sha256']}"
            )
        selected.append((row, sample))
    return selected


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("invalid binomial counts")
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def paired_accuracy_delta(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260729,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    left_by_id = {row["sample_id"]: bool(row["correct"]) for row in left}
    right_by_id = {row["sample_id"]: bool(row["correct"]) for row in right}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("paired reports do not contain identical sample ids")
    ids = sorted(left_by_id)
    deltas = [int(left_by_id[key]) - int(right_by_id[key]) for key in ids]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    boot = []
    for _ in range(bootstrap_samples):
        boot.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    boot.sort()
    lo = boot[int(0.025 * bootstrap_samples)]
    hi = boot[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]
    return {
        "n": len(ids),
        "accuracy_delta": observed,
        "bootstrap_95_ci": [lo, hi],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def compare_gate_b_reports(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    seed: int = 20260729,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    for field in ("id", "revision"):
        if left["model"][field] != right["model"][field]:
            raise ValueError(f"model {field} differs between reports")
    if (
        left["manifest"]["content_sha256"]
        != right["manifest"]["content_sha256"]
    ):
        raise ValueError("reports do not use the same sealed sample manifest")
    left_tasks = {result["task"]: result for result in left["results"]}
    right_tasks = {result["task"]: result for result in right["results"]}
    if left_tasks.keys() != right_tasks.keys():
        raise ValueError("reports do not contain identical task sets")
    task_comparisons = []
    for task in sorted(left_tasks):
        left_result, right_result = left_tasks[task], right_tasks[task]
        paired = paired_accuracy_delta(
            left_result["samples"],
            right_result["samples"],
            seed=_seed_for(seed, task),
            bootstrap_samples=bootstrap_samples,
        )
        paired.update(
            {
                "task": task,
                "left_accuracy": left_result["accuracy"],
                "right_accuracy": right_result["accuracy"],
                "mean_latency_ratio": (
                    left_result["latency_ms"]["mean"]
                    / right_result["latency_ms"]["mean"]
                ),
            }
        )
        task_comparisons.append(paired)
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "left_arm": left["arm"],
        "right_arm": right["arm"],
        "model": left["model"],
        "manifest_content_sha256": left["manifest"]["content_sha256"],
        "tasks": task_comparisons,
    }
    comparison["content_sha256"] = digest_json(comparison)
    return comparison


def checkpoint_assignment_plan(
    checkpoint_state: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> dict[str, str]:
    """Map every checkpoint tensor to the deployed model or fail closed."""
    normalized = {
        (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
        for key, value in checkpoint_state.items()
    }
    assignments: dict[str, str] = {}
    errors: list[str] = []
    for key, value in normalized.items():
        if key.endswith("weight_master"):
            target = key[: -len("weight_master")] + "weight"
        else:
            target = key
        if target not in model_state:
            errors.append(f"checkpoint tensor {key!r} has no model target {target!r}")
            continue
        if tuple(value.shape) != tuple(model_state[target].shape):
            errors.append(
                f"shape mismatch {key!r} -> {target!r}: "
                f"{tuple(value.shape)} != {tuple(model_state[target].shape)}"
            )
            continue
        if target in assignments:
            errors.append(
                f"multiple checkpoint tensors map to {target!r}: "
                f"{assignments[target]!r} and {key!r}"
            )
            continue
        assignments[target] = key
    missing = sorted(set(model_state) - set(assignments))
    if missing:
        errors.append(
            f"{len(missing)} model tensors absent from checkpoint, first: {missing[:8]}"
        )
    if errors:
        raise ValueError("checkpoint/model contract failed:\n- " + "\n- ".join(errors))
    return assignments


def validate_arm(arm: str, checkpoint: Path | None) -> None:
    if arm not in VALID_ARMS:
        raise ValueError(f"arm must be one of {sorted(VALID_ARMS)}")
    needs_checkpoint = arm == "qat_ternary"
    if needs_checkpoint != (checkpoint is not None):
        requirement = "requires" if needs_checkpoint else "must not use"
        raise ValueError(f"arm {arm} {requirement} --ckpt")


def git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_provenance(repo: Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("torch", "transformers", "datasets", "accelerate"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "git_revision": git_revision(repo),
        "pid": os.getpid(),
    }
