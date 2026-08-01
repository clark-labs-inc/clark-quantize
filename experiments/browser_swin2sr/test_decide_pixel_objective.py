import unittest

from decide_pixel_objective import (
    index_samples,
    paired_interval,
    source_deltas,
)


class PixelObjectiveDecisionTests(unittest.TestCase):
    def test_paired_interval_preserves_constant_effect(self):
        interval = paired_interval(
            [0.25] * 20,
            seed=7,
            samples=100,
        )

        self.assertEqual(interval["mean"], 0.25)
        self.assertEqual(interval["lower_95"], 0.25)
        self.assertEqual(interval["upper_95"], 0.25)
        self.assertEqual(interval["win_fraction"], 1.0)

    def test_source_deltas_are_paired_by_identity(self):
        control = {
            "a": {"source": "ui", "psnr_vs_ground_truth": 30.0},
            "b": {"source": "text", "psnr_vs_ground_truth": 31.0},
            "c": {"source": "ui", "psnr_vs_ground_truth": 32.0},
        }
        candidate = {
            "c": {"source": "ui", "psnr_vs_ground_truth": 32.5},
            "a": {"source": "ui", "psnr_vs_ground_truth": 30.5},
            "b": {"source": "text", "psnr_vs_ground_truth": 30.8},
        }

        self.assertEqual(
            source_deltas(control, candidate),
            {"text": -0.1999999999999993, "ui": 0.5},
        )

    def test_index_rejects_duplicate_identity(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            index_samples({"samples": [
                {"sha256": "same"},
                {"sha256": "same"},
            ]})


if __name__ == "__main__":
    unittest.main()
