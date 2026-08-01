import json
import tempfile
import unittest
from pathlib import Path

from rank_factorial_source_aware import rerank


def report(name, rico, flores):
    return {
        "name": name,
        "status": "complete",
        "heldout": {
            "samples": [
                {
                    "sha256": "r",
                    "source": "rico",
                    "psnr_vs_ground_truth": rico,
                    "edge_psnr_vs_ground_truth": rico,
                    "psnr_vs_teacher": rico,
                },
                {
                    "sha256": "f",
                    "source": "flores",
                    "psnr_vs_ground_truth": flores,
                    "edge_psnr_vs_ground_truth": flores,
                    "psnr_vs_teacher": flores,
                },
            ],
        },
    }


class SourceAwareRankingTests(unittest.TestCase):
    def test_worst_source_precedes_aggregate(self):
        q4 = {
            "samples": [
                {
                    "sha256": key,
                    "psnr_vs_ground_truth": 10.0,
                    "edge_psnr_vs_ground_truth": 10.0,
                    "psnr_vs_teacher": 10.0,
                }
                for key in ("r", "f")
            ],
        }
        with tempfile.TemporaryDirectory() as raw:
            stage = Path(raw)
            for row in (
                report("aggregate", 9.0, 13.0),
                report("safe", 10.1, 10.1),
            ):
                destination = stage / row["name"]
                destination.mkdir()
                (destination / "result.json").write_text(json.dumps(row))
            ranked = rerank(stage, q4)
        self.assertEqual(
            [row["name"] for row in ranked["ranking"]],
            ["safe", "aggregate"],
        )


if __name__ == "__main__":
    unittest.main()
