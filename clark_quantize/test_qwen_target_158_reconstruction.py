from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/qwen38_27b_spot/target_158_reconstruction_gate.py"
SPEC = importlib.util.spec_from_file_location("target_158_reconstruction_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Target158ReconstructionTests(unittest.TestCase):
    def test_projected_ledger_counts_all_rank2_and_retains_others(self) -> None:
        inventory = {
            "matrix_a": {"shape": [3, 2050], "parameters": 6150},
            "matrix_b": {"shape": [2, 10], "parameters": 20},
            "conv": {"shape": [4, 1, 3], "parameters": 12},
        }
        row = MODULE.projected_ledger(inventory, "all_rank2_group1024_fp8_scales")
        self.assertEqual(row["total_parameters"], 6182)
        self.assertEqual(row["rank2_ternary_parameters"], 6170)
        self.assertEqual(row["rank_not_2_bf16_parameters"], 12)
        self.assertEqual(row["scale_count"], 3 * 3 + 2)
        expected = (6170 * math.log2(3) + 12 * 16 + 11 * 8) / 6182
        self.assertAlmostEqual(row["complete_effective_bits_per_parameter"], expected)

    def test_candidate_reconstruction_is_finite_and_exact_ternary(self) -> None:
        torch.manual_seed(7)
        weight = torch.randn(5, 2050)
        for name in (
            "all_rank2_group1024_fp8_scales",
            "all_rank2_group2048_fp8_scales",
            "all_rank2_per_row_fp16_scales",
        ):
            result = MODULE.evaluate_candidate(weight, name)
            self.assertTrue(result["finite"])
            self.assertTrue(result["codes_exact_ternary"])
            self.assertTrue(result["exact_code_roundtrip"])
            self.assertEqual(result["parameters"], weight.numel())
            self.assertGreaterEqual(result["zero_fraction"], 0.0)
            self.assertLessEqual(result["zero_fraction"], 1.0)
            self.assertEqual(len(result["codes_sha256"]), 64)
            self.assertEqual(len(result["stored_scales_sha256"]), 64)
            self.assertEqual(len(result["base3_frame_sha256"]), 64)

    def test_unknown_candidate_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown sealed candidate"):
            MODULE.candidate_spec("invented", 128)

    def test_retained_dtype_ledger_requires_every_nonmatrix_to_be_bfloat16(self) -> None:
        inventory = {
            "matrix": {"shape": [2, 3], "parameters": 6},
            "vector": {"shape": [3], "parameters": 3, "dtype": "torch.bfloat16"},
            "conv": {"shape": [2, 1, 3], "parameters": 6, "dtype": "torch.bfloat16"},
        }
        ledger = MODULE.retained_dtype_ledger(inventory)
        self.assertEqual(ledger["tensor_count"], 2)
        self.assertEqual(ledger["parameters"], 9)
        self.assertEqual(ledger["parameter_count_by_dtype"], {"torch.bfloat16": 9})
        self.assertTrue(ledger["all_bfloat16"])
        inventory["conv"]["dtype"] = "torch.float32"
        self.assertFalse(MODULE.retained_dtype_ledger(inventory)["all_bfloat16"])


if __name__ == "__main__":
    unittest.main()
