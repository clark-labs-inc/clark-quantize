from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import summarize_qwen38_27b_paired_throughput as module


class PairedThroughputSummaryTests(unittest.TestCase):
    def test_complete_speedup_matrix_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.gguf"
            candidate = root / "candidate.gguf"
            bench = root / "llama-bench"
            for path in (baseline, candidate, bench):
                path.touch()
            for label, rate in (("baseline", 10.0), ("candidate", 11.0)):
                for threads in (1, 2, 4):
                    prefill = []
                    for prompt in (128, 512, 2048):
                        prefill.append(
                            {
                                "model_filename": str(
                                    (baseline if label == "baseline" else candidate).resolve()
                                ),
                                "n_threads": threads,
                                "cpu_strict": True,
                                "samples_ts": [rate] * 5,
                                "avg_ts": rate,
                                "stddev_ts": 0.0,
                                "n_prompt": prompt,
                                "n_gen": 0,
                            }
                        )
                    decode = [
                        {
                            "model_filename": str(
                                (baseline if label == "baseline" else candidate).resolve()
                            ),
                            "n_threads": threads,
                            "cpu_strict": True,
                            "samples_ts": [rate] * 5,
                            "avg_ts": rate,
                            "stddev_ts": 0.0,
                            "n_prompt": 0,
                            "n_gen": 128,
                        }
                    ]
                    (root / f"{label}-prefill-t{threads}.json").write_text(json.dumps(prefill))
                    (root / f"{label}-decode-t{threads}.json").write_text(json.dumps(decode))
            argv = [
                "summary",
                "--root", str(root),
                "--baseline", str(baseline),
                "--candidate", str(candidate),
                "--baseline-sha256", "a" * 64,
                "--candidate-sha256", "b" * 64,
                "--llama-bench", str(bench),
                "--llama-bench-sha256", "c" * 64,
                "--schema", "test_v1",
                "--claim-boundary", "test only",
            ]
            with patch.object(sys, "argv", argv):
                module.main()
            summary = json.loads((root / "summary.json").read_text())
            self.assertTrue(summary["speed_promotion_pass"])
            self.assertAlmostEqual(summary["geometric_mean_speedup"], 1.1)
            self.assertEqual(len(summary["comparisons"]), 12)


if __name__ == "__main__":
    unittest.main()
