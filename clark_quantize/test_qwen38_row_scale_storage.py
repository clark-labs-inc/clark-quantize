from __future__ import annotations

import unittest

from clark_quantize.qwen38_row_scale_storage import row_normalized_scale_ledger
from clark_quantize.qwen38_storage_budget import (
    StorageStrategy,
    TensorSpec,
    align_up,
    strategy_ledger,
)


class Qwen38RowScaleStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tensors = [
            TensorSpec("a.weight", (3, 129), "BF16"),
            TensorSpec("b.weight", (2, 128), "BF16"),
            TensorSpec("norm", (5,), "BF16"),
        ]
        self.base = strategy_ledger(
            self.tensors,
            StorageStrategy("base3_group128_codebook4", "base3", 128, 4, True),
            auxiliary_bytes=17,
        )

    def test_counts_anchors_and_per_tensor_alignment(self) -> None:
        row = row_normalized_scale_ledger(self.tensors, self.base)
        self.assertEqual(row["scales"]["count"], 8)
        self.assertEqual(row["scales"]["index_bytes"], 4)
        self.assertEqual(row["scales"]["fp16_row_anchor_count"], 5)
        self.assertEqual(row["scales"]["fp16_row_anchor_bytes"], 10)
        self.assertEqual(row["scales"]["fp16_codebook_bytes"], 64)
        expected_sections = align_up(3 + 6 + 32) + align_up(1 + 4 + 32)
        self.assertEqual(row["bytes"]["scale_sections_aligned"], expected_sections)
        expected_bundle = (
            self.base["bytes"]["complete_bundle"]
            - self.base["bytes"]["scale_sections_aligned"]
            + expected_sections
        )
        self.assertEqual(row["bytes"]["complete_bundle"], expected_bundle)

    def test_rejects_unrelated_base_strategy(self) -> None:
        bad = strategy_ledger(
            self.tensors, StorageStrategy("base3_group128_fp16", "base3", 128, 16)
        )
        with self.assertRaisesRegex(ValueError, "expected"):
            row_normalized_scale_ledger(self.tensors, bad)


if __name__ == "__main__":
    unittest.main()
