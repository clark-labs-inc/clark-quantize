import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).with_name("materialize_rstb_precision_boundary.py")
SPEC = importlib.util.spec_from_file_location("rstb_boundary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QuantizeBoundaryTests(unittest.TestCase):
    def test_all_boundary_arms_use_exact_storage(self):
        weight = torch.randn(60, 60, 3, 3)
        expected = {
            "int5_per_output": (20_250, 120),
            "int6_per_output": (24_300, 120),
            "int5_per_kernel": (20_250, 7_200),
            "int7_per_output": (28_350, 120),
            "int6_per_kernel": (24_300, 7_200),
        }
        for arm, config in MODULE.ARM_SPECS.items():
            with self.subTest(arm=arm):
                recovered, receipt = MODULE.quantize_weight_nbit(weight, **config)
                self.assertEqual(tuple(recovered.shape), tuple(weight.shape))
                self.assertEqual(
                    (receipt["code_bytes"], receipt["scale_bytes"]),
                    expected[arm],
                )
                self.assertTrue(receipt["scale_fp16_roundtrip_exact"])

    def test_signed_range_matches_bit_width(self):
        weight = torch.linspace(-1, 1, 2 * 3 * 3 * 3).reshape(2, 3, 3, 3)
        for bits in (5, 6, 7):
            _recovered, receipt = MODULE.quantize_weight_nbit(
                weight, bits, "per_output"
            )
            limit = (1 << (bits - 1)) - 1
            self.assertGreaterEqual(receipt["code_min"], -limit)
            self.assertLessEqual(receipt["code_max"], limit)

    def test_rejects_nonboundary_width(self):
        with self.assertRaisesRegex(ValueError, "boundary bit width"):
            MODULE.quantize_weight_nbit(torch.randn(2, 3, 3, 3), 8, "per_output")


if __name__ == "__main__":
    unittest.main()
