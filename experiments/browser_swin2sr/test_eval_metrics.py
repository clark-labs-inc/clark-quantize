import unittest

import numpy as np

from evaluate_stratified import product_pixels


class EvalMetricTests(unittest.TestCase):
    def test_product_pixels_matches_uint8_rounding_and_clamping(self):
        values = np.asarray([-0.2, 0.0, 0.5, 1.0, 1.2], dtype=np.float32)

        result = product_pixels(values)

        np.testing.assert_array_equal(
            result,
            np.asarray([0, 0, 128, 255, 255], dtype=np.float32) / 255,
        )
        self.assertEqual(result.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
