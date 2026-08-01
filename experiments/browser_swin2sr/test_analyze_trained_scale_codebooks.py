import unittest

import torch

from analyze_trained_scale_codebooks import error_stats


class TrainedScaleCodebookAnalysisTests(unittest.TestCase):
    def test_error_stats_are_exact(self):
        reference = torch.tensor([3.0, 4.0])
        candidate = torch.tensor([0.0, 4.0])
        result = error_stats(reference, candidate)
        self.assertEqual(result["values"], 2)
        self.assertEqual(result["error_squared"], 9.0)
        self.assertEqual(result["signal_squared"], 25.0)
        self.assertAlmostEqual(result["relative_rmse"], 0.6)
        self.assertEqual(result["max_abs_delta"], 3.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            error_stats(torch.zeros(2), torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
