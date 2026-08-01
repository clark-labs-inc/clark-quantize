import unittest

from analyze_gradient_group_sensitivity import correlation


class GradientGroupSensitivityTests(unittest.TestCase):
    def test_correlation_tracks_aligned_ordering(self):
        self.assertAlmostEqual(
            correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]),
            1.0,
        )

    def test_correlation_rejects_constant_signal(self):
        self.assertIsNone(correlation([1.0, 1.0], [2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
