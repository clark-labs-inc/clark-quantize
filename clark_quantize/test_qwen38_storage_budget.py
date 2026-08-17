from __future__ import annotations

import math
import unittest

from clark_quantize.qwen38_storage_budget import (
    LOG2_3,
    StorageStrategy,
    TensorSpec,
    align_up,
    canonical_inventory_digest,
    entropy_lower_bound,
    layout_metadata_bytes,
    strategy_ledger,
    tensor_code_bits,
    tensor_scale_count,
)


class Qwen38StorageBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tensors = [
            TensorSpec("layer.weight", (2, 129), "BF16"),
            TensorSpec("layer.norm", (3,), "BF16"),
        ]

    def test_block_base3_and_scale_counts_are_exact(self) -> None:
        strategy = StorageStrategy("test", "base3", 128, 16)
        self.assertEqual(tensor_code_bits(self.tensors[0], strategy), 2 * (203 + 2))
        self.assertEqual(tensor_scale_count(self.tensors[0], strategy), 4)
        self.assertEqual(math.ceil(128 * LOG2_3), 203)

    def test_complete_ledger_counts_alignment_metadata_and_auxiliary_files(self) -> None:
        strategy = StorageStrategy("test", "base3", 128, 16)
        ledger = strategy_ledger(self.tensors, strategy, auxiliary_bytes=11)
        self.assertEqual(ledger["parameters"], {"total": 261, "ternary": 258, "retained_exact": 3})
        self.assertEqual(ledger["bytes"]["ternary_code_payload"], 52)
        self.assertEqual(ledger["bytes"]["ternary_code_sections_aligned"], 64)
        self.assertEqual(ledger["bytes"]["scale_payload_and_codebooks"], 8)
        self.assertEqual(ledger["bytes"]["scale_sections_aligned"], 64)
        self.assertEqual(ledger["bytes"]["retained_exact_payload"], 6)
        self.assertEqual(ledger["bytes"]["retained_exact_sections_aligned"], 64)
        expected = 64 + 64 + 64 + layout_metadata_bytes(self.tensors) + 11
        self.assertEqual(ledger["bytes"]["complete_bundle"], expected)
        self.assertEqual(ledger["effective_bits_per_total_parameter"], expected * 8 / 261)

    def test_four_bit_scale_codebook_is_counted_per_quantized_tensor(self) -> None:
        strategy = StorageStrategy("test", "base3", 128, 4, True)
        ledger = strategy_ledger(self.tensors, strategy)
        self.assertEqual(ledger["scales"]["index_bytes"], 2)
        self.assertEqual(ledger["scales"]["fp16_codebook_bytes"], 32)

    def test_lower_bound_excludes_scales_and_is_explicitly_unattainable(self) -> None:
        lower = entropy_lower_bound(self.tensors, auxiliary_bytes=7)
        self.assertIn("unattainable", lower["description"])
        actual = strategy_ledger(
            self.tensors, StorageStrategy("test", "base3", 128, 4, True), 7
        )
        self.assertGreater(actual["effective_bits_per_total_parameter"], lower["effective_bits_per_total_parameter"])

    def test_inventory_digest_is_order_independent(self) -> None:
        self.assertEqual(
            canonical_inventory_digest(self.tensors),
            canonical_inventory_digest(list(reversed(self.tensors))),
        )

    def test_invalid_sub_16_bit_scale_without_codebook_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StorageStrategy("bad", "base3", 128, 4, False)

    def test_align_up_rejects_invalid_inputs(self) -> None:
        self.assertEqual(align_up(65), 128)
        with self.assertRaises(ValueError):
            align_up(-1)


if __name__ == "__main__":
    unittest.main()
