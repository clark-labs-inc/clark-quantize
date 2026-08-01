import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).with_name("materialize_rstb_mixed_precision.py")
SPEC = importlib.util.spec_from_file_location("rstb_mixed", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QuantizeWeightTests(unittest.TestCase):
    def test_int4_per_output_uses_fp16_scale_and_signed_range(self):
        weight = torch.linspace(-1.0, 1.0, 2 * 3 * 3 * 3).reshape(2, 3, 3, 3)
        recovered, receipt = MODULE.quantize_weight(weight, 4, "per_output")
        self.assertEqual(tuple(recovered.shape), tuple(weight.shape))
        self.assertEqual(receipt["scales"], 2)
        self.assertEqual(receipt["code_bytes"], 27)
        self.assertEqual(receipt["scale_bytes"], 4)
        self.assertGreaterEqual(receipt["code_min"], -7)
        self.assertLessEqual(receipt["code_max"], 7)
        self.assertTrue(receipt["scale_fp16_roundtrip_exact"])

    def test_int4_per_kernel_has_one_scale_per_spatial_kernel(self):
        weight = torch.randn(2, 3, 3, 3)
        _recovered, receipt = MODULE.quantize_weight(weight, 4, "per_kernel")
        self.assertEqual(receipt["scales"], 6)
        self.assertEqual(receipt["code_bytes"], 27)
        self.assertEqual(receipt["scale_bytes"], 12)

    def test_int8_per_output_storage(self):
        weight = torch.randn(60, 60, 3, 3)
        _recovered, receipt = MODULE.quantize_weight(weight, 8, "per_output")
        self.assertEqual(receipt["values"], 32_400)
        self.assertEqual(receipt["code_bytes"], 32_400)
        self.assertEqual(receipt["scale_bytes"], 120)

    def test_rejects_non_convolution_shape(self):
        with self.assertRaisesRegex(ValueError, "3x3 convolution"):
            MODULE.quantize_weight(torch.randn(2, 3), 4, "per_output")

    def test_zero_groups_are_exact(self):
        weight = torch.zeros(2, 3, 3, 3)
        recovered, receipt = MODULE.quantize_weight(weight, 4, "per_output")
        self.assertTrue(torch.equal(recovered, weight))
        self.assertEqual(receipt["code_min"], 0)
        self.assertEqual(receipt["code_max"], 0)


if __name__ == "__main__":
    unittest.main()
