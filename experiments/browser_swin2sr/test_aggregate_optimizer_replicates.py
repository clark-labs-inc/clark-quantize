from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aggregate_optimizer_replicates import aggregate


def write_run(root: Path, seed: int, first: float, second: float) -> Path:
    run = root / f"seed-{seed}"
    run.mkdir()
    arms = [
        {"name": "first__lr_x4"},
        {"name": "second__lr_x4"},
    ]
    (run / "manifest.json").write_text(json.dumps({
        "protocol": {"seed": seed, "steps": 1000},
        "source_sha256": {"trainer.py": "sealed"},
        "arms": arms,
    }))
    (run / "results.json").write_text(json.dumps({
        "results": [
            {
                "name": "first__lr_x4",
                "optimizer": "first",
                "learning_rate": 8e-5,
                "learning_rate_multiplier": 4,
                "return_code": 0,
                "validation": {
                    "psnr_vs_ground_truth": first,
                },
            },
            {
                "name": "second__lr_x4",
                "optimizer": "second",
                "learning_rate": 8e-5,
                "learning_rate_multiplier": 4,
                "return_code": 0,
                "validation": {
                    "psnr_vs_ground_truth": second,
                },
            },
        ],
    }))
    return run


class AggregateOptimizerReplicatesTest(unittest.TestCase):
    def test_uses_paired_seed_differences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [
                write_run(root, 1, 42.0, 41.0),
                write_run(root, 2, 43.0, 42.0),
                write_run(root, 3, 44.0, 43.0),
            ]
            receipt = aggregate(
                runs,
                "psnr_vs_ground_truth",
                noninferiority_margin=0.1,
            )
            self.assertEqual(receipt["ranking"][0]["name"], "first__lr_x4")
            comparison = receipt["paired_against_best"][0]
            self.assertEqual(comparison["mean"], 1.0)
            self.assertEqual(comparison["ci95_low"], 1.0)
            self.assertTrue(comparison["best_statistically_superior"])
            self.assertFalse(comparison["candidate_noninferior_within_margin"])

    def test_rejects_protocol_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_run(root, 1, 42.0, 41.0)
            second = write_run(root, 2, 43.0, 42.0)
            manifest = json.loads((second / "manifest.json").read_text())
            manifest["protocol"]["steps"] = 2000
            (second / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
                aggregate(
                    [first, second],
                    "psnr_vs_ground_truth",
                    noninferiority_margin=0.1,
                )


if __name__ == "__main__":
    unittest.main()
