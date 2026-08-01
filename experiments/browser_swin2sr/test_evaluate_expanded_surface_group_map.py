import unittest

from evaluate_expanded_surface_group_map import (
    effective_ternary_bits,
    quality_gate,
)


class ExpandedSurfaceGroupMapTests(unittest.TestCase):
    def test_effective_bits_include_codes_and_scales(self):
        self.assertEqual(effective_ternary_bits(100, 20, 5), 2.0)

    def test_quality_gate_requires_positive_ground_truth_interval(self):
        passed, failures = quality_gate({
            "psnr_vs_ground_truth": {"lower_95": 0.0},
            "edge_psnr_vs_ground_truth": {"mean": 0.0},
            "psnr_vs_teacher": {"mean": 0.0},
        })

        self.assertFalse(passed)
        self.assertEqual(failures, ["ground_truth_lower_95_not_positive"])


if __name__ == "__main__":
    unittest.main()
