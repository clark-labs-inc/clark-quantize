import unittest

from train_fixed_int7_scales import sealed_gate_failure


class SealedGateTests(unittest.TestCase):
    def test_step_250_requires_a_gain_without_material_regression(self):
        self.assertIsNone(sealed_gate_failure(250, {
            "psnr_vs_ground_truth": 0.001,
            "psnr_vs_teacher": -0.001,
        }))
        self.assertIsNotNone(sealed_gate_failure(250, {
            "psnr_vs_ground_truth": -0.003,
            "psnr_vs_teacher": 0.010,
        }))
        self.assertIsNotNone(sealed_gate_failure(250, {
            "psnr_vs_ground_truth": 0.0,
            "psnr_vs_teacher": 0.0,
        }))

    def test_step_500_requires_both_metrics_nonnegative(self):
        self.assertIsNone(sealed_gate_failure(500, {
            "psnr_vs_ground_truth": 0.0,
            "psnr_vs_teacher": 0.001,
        }))
        self.assertIsNotNone(sealed_gate_failure(500, {
            "psnr_vs_ground_truth": -0.000001,
            "psnr_vs_teacher": 0.100,
        }))


if __name__ == "__main__":
    unittest.main()
