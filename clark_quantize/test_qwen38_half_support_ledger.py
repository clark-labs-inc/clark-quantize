from __future__ import annotations

import unittest

from clark_quantize.qwen38_half_support_ledger import (
    half_support_checkpoint_ledger,
    half_support_group_bytes,
    mask_payload_bytes,
    tensor_half_support_ledger,
)
from clark_quantize.qwen38_storage_budget import TensorSpec


class Qwen38HalfSupportLedgerTests(unittest.TestCase):
    def test_group_and_vision_tail_are_exactly_one_point_five_bpw(self) -> None:
        self.assertEqual(half_support_group_bytes(128), 24)
        self.assertEqual(half_support_group_bytes(80), 15)
        self.assertEqual(mask_payload_bytes(rows=1152, columns=4304), 1152 * 807)
        tensor = tensor_half_support_ledger(
            TensorSpec("vision.fc2", (1152, 4304), "BF16")
        )
        self.assertEqual(tensor["tail_width"], 80)
        self.assertEqual(tensor["tail_nonzero_count"], 40)
        self.assertEqual(tensor["raw_mask_bits_per_weight"], 1.5)

    def test_short_conv_and_rank_one_are_retained(self) -> None:
        conv = tensor_half_support_ledger(
            TensorSpec("linear.conv", (10240, 1, 4), "BF16")
        )
        norm = tensor_half_support_ledger(
            TensorSpec("norm", (5120,), "BF16")
        )
        self.assertEqual(conv["policy"], "retained_exact")
        self.assertIn("break-even", conv["reason"])
        self.assertEqual(norm["policy"], "retained_exact")

    def test_complete_ledger_counts_every_parameter_and_byte_class(self) -> None:
        tensors = [
            TensorSpec("gate", (17408, 5120), "BF16"),
            TensorSpec("vision.fc2", (1152, 4304), "BF16"),
            TensorSpec("linear.conv", (10240, 1, 4), "BF16"),
            TensorSpec("norm", (5120,), "BF16"),
        ]
        ledger = half_support_checkpoint_ledger(tensors, auxiliary_bytes=12345)
        self.assertEqual(
            ledger["parameters"]["total"],
            sum(tensor.parameters for tensor in tensors),
        )
        self.assertEqual(ledger["tensors"]["total"], len(tensors))
        self.assertEqual(ledger["tensors"]["compressed"], 2)
        self.assertEqual(ledger["tensors"]["compressed_with_even_tail"], 1)
        self.assertEqual(ledger["bytes"]["auxiliary_model_files"], 12345)
        self.assertEqual(
            ledger["bytes"]["complete_bundle"] * 8
            / ledger["parameters"]["total"],
            ledger["effective_bits_per_total_parameter"],
        )

    def test_invalid_half_support_group_is_rejected(self) -> None:
        for width in (0, 3, 129):
            with self.assertRaises(ValueError):
                half_support_group_bytes(width)


if __name__ == "__main__":
    unittest.main()
