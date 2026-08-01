import unittest

import numpy as np

from score_eval_outputs import score_sample


class ScoreEvalOutputsTests(unittest.TestCase):
    def test_identical_runtime_and_teacher_have_lossless_teacher_psnr(self):
        truth = np.zeros((8, 8, 3), dtype=np.float32)
        teacher = np.full((8, 8, 3), 0.25, dtype=np.float32)
        bicubic = np.full((8, 8, 3), 0.5, dtype=np.float32)

        result = score_sample(teacher.copy(), truth, teacher, bicubic)

        self.assertEqual(result["psnr_vs_teacher"], 99.0)
        self.assertGreater(
            result["psnr_vs_ground_truth"],
            result["bicubic_psnr_vs_ground_truth"],
        )

    def test_edge_metric_penalizes_shifted_boundary(self):
        truth = np.zeros((8, 8, 3), dtype=np.float32)
        truth[:, 4:] = 1
        shifted = np.zeros_like(truth)
        shifted[:, 5:] = 1

        result = score_sample(shifted, truth, truth, shifted)

        self.assertLess(result["edge_psnr_vs_ground_truth"], 30)


if __name__ == "__main__":
    unittest.main()
