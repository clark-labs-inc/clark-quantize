from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from decide_optimizer_replicates import decide


def write_summary(
    root: Path,
    seed: int,
    *,
    first_gt: float,
    first_teacher: float,
    first_state: int,
    second_gt: float,
    second_teacher: float,
    second_state: int,
) -> tuple[Path, dict]:
    manifest = root / f"manifest-{seed}.json"
    results = root / f"results-{seed}.json"
    manifest.write_text(str(seed))
    results.write_text(f"results-{seed}")
    summary = {
        "format": "backdropkit-swin2sr-optimizer-tournament-summary-v1",
        "stage": "optimizer-replicate",
        "manifest_sha256": digest(manifest),
        "results_sha256": digest(results),
        "arms_rejected": 0,
        "ranking": [
            {
                "name": "first",
                "optimizer": "first",
                "psnr_vs_ground_truth": first_gt,
                "psnr_vs_teacher": first_teacher,
                "optimizer_state_bytes": first_state,
            },
            {
                "name": "second",
                "optimizer": "second",
                "psnr_vs_ground_truth": second_gt,
                "psnr_vs_teacher": second_teacher,
                "optimizer_state_bytes": second_state,
            },
        ],
    }
    path = root / f"summary-{seed}.json"
    path.write_text(json.dumps(summary))
    return path, {
        "seed": seed,
        "manifest_sha256": summary["manifest_sha256"],
        "results_sha256": summary["results_sha256"],
    }


def digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_aggregate(
    root: Path,
    runs: list[dict],
    *,
    gt_noninferior: bool,
) -> Path:
    aggregate = {
        "format": "backdropkit-swin2sr-optimizer-replicates-v1",
        "diagnostic_only": True,
        "metric": "psnr_vs_ground_truth",
        "noninferiority_margin_db": 0.10,
        "runs": runs,
        "ranking": [
            {"name": "first", "mean": 42.05},
            {"name": "second", "mean": 42.00},
        ],
        "paired_against_best": [{
            "best": "first",
            "candidate": "second",
            "candidate_noninferior_within_margin": gt_noninferior,
        }],
    }
    path = root / "aggregate.json"
    path.write_text(json.dumps(aggregate))
    return path


class DecideOptimizerReplicatesTest(unittest.TestCase):
    def test_selects_lower_state_when_both_metrics_are_noninferior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summaries = []
            runs = []
            for seed, offset in ((1, 0.00), (2, 0.01), (3, -0.01)):
                path, run = write_summary(
                    root,
                    seed,
                    first_gt=42.05 + offset,
                    first_teacher=48.05 + offset,
                    first_state=1_000,
                    second_gt=42.00 + offset,
                    second_teacher=48.00 + offset,
                    second_state=0,
                )
                summaries.append(path)
                runs.append(run)
            aggregate = write_aggregate(
                root,
                runs,
                gt_noninferior=True,
            )
            receipt = decide(aggregate, summaries, teacher_margin=0.10)
            self.assertEqual(receipt["selected"]["name"], "second")
            self.assertEqual(
                receipt["eligible_low_state_set"],
                ["second", "first"],
            )

    def test_rejects_low_state_arm_when_teacher_metric_is_inferior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summaries = []
            runs = []
            for seed in (1, 2, 3):
                path, run = write_summary(
                    root,
                    seed,
                    first_gt=42.05,
                    first_teacher=48.50,
                    first_state=1_000,
                    second_gt=42.00,
                    second_teacher=48.00,
                    second_state=0,
                )
                summaries.append(path)
                runs.append(run)
            aggregate = write_aggregate(
                root,
                runs,
                gt_noninferior=True,
            )
            receipt = decide(aggregate, summaries, teacher_margin=0.10)
            self.assertEqual(receipt["selected"]["name"], "first")
            second = next(
                item for item in receipt["candidates"]
                if item["name"] == "second"
            )
            self.assertFalse(second["teacher_noninferior"])

    def test_requires_every_aggregate_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, run = write_summary(
                root,
                1,
                first_gt=42,
                first_teacher=48,
                first_state=100,
                second_gt=41.95,
                second_teacher=47.95,
                second_state=0,
            )
            extra = dict(run)
            extra["seed"] = 2
            extra["manifest_sha256"] = "missing"
            aggregate = write_aggregate(
                root,
                [run, extra],
                gt_noninferior=True,
            )
            with self.assertRaisesRegex(RuntimeError, "expected 2 summaries"):
                decide(aggregate, [summary], teacher_margin=0.10)


if __name__ == "__main__":
    unittest.main()
