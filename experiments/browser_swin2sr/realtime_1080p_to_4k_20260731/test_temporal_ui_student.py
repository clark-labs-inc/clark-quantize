import random
import unittest

import numpy as np
from PIL import Image

from train_temporal_ui_student import degrade_pair, gate_failure, translated_pair


class TemporalUiStudentTests(unittest.TestCase):
    def test_pair_generation_preserves_aligned_shapes(self) -> None:
        y, x = np.mgrid[:160, :192]
        pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
        pair = translated_pair(Image.fromarray(pixels), 128, random.Random(7), "cpu")
        low = degrade_pair(pair, random.Random(8))
        self.assertEqual(tuple(pair.shape), (2, 3, 128, 128))
        self.assertEqual(tuple(low.shape), (2, 3, 64, 64))
        self.assertGreater(float((pair[1] - pair[0]).abs().mean()), 0)

    def test_sealed_gate_requires_temporal_gain_without_spatial_loss(self) -> None:
        passing = {
            "psnr_vs_ground_truth": 0.0,
            "edge_psnr_vs_ground_truth": 0.0,
            "psnr_vs_teacher": 0.0,
            "temporal_delta_psnr": 0.03,
            "static_delta_mae": -0.0001,
        }
        self.assertIsNone(gate_failure(250, passing))
        failing = {**passing, "temporal_delta_psnr": 0.01}
        self.assertIsNotNone(gate_failure(250, failing))
        failing = {**passing, "psnr_vs_ground_truth": -0.021}
        self.assertIsNotNone(gate_failure(250, failing))


if __name__ == "__main__":
    unittest.main()
