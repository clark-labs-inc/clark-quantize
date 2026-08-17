from __future__ import annotations

import unittest

from clark_quantize.qwen38_hash_storage_budget import (
    HashResidualStorageStrategy,
    complete_hash_storage_ledger,
    hash_tensor_storage_ledger,
    tensor_hash_eligible,
)
from clark_quantize.qwen38_storage_budget import TensorSpec, align_up, layout_metadata_bytes


class Qwen38HashStorageBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = HashResidualStorageStrategy(
            name="hash128_sparse64",
            group_size=128,
            nonzero_per_group=64,
            input_to_sketch_ratio=128,
        )

    def test_exact_small_eligible_tensor_ledger(self) -> None:
        tensor = TensorSpec("projection.weight", (2, 128), "BF16")
        ledger = hash_tensor_storage_ledger(tensor, self.strategy)
        self.assertEqual(ledger["group_width_bits"], 189)
        self.assertEqual(ledger["sketch_dim"], 1)
        self.assertEqual(ledger["bytes"]["fixed_support_ternary_payload"], 48)
        self.assertEqual(ledger["bytes"]["row_anchors_FP16"], 4)
        self.assertEqual(ledger["bytes"]["group_scale_indices_INT4"], 1)
        self.assertEqual(ledger["bytes"]["tensor_ratio_codebook_16xFP16"], 32)
        self.assertEqual(ledger["bytes"]["hash_coefficients_INT4"], 2)
        self.assertEqual(ledger["bytes"]["hash_coefficient_row_scales_FP16"], 4)
        self.assertEqual(ledger["bytes"]["payload"], 155)
        self.assertEqual(ledger["bytes"]["aligned_section"], 192)

    def test_eligibility_requires_rank_group_and_ratio_alignment(self) -> None:
        self.assertTrue(tensor_hash_eligible(TensorSpec("ok", (3, 256), "BF16"), self.strategy))
        self.assertFalse(tensor_hash_eligible(TensorSpec("vector", (256,), "BF16"), self.strategy))
        self.assertFalse(tensor_hash_eligible(TensorSpec("tail", (3, 192), "BF16"), self.strategy))

    def test_complete_ledger_counts_every_tensor_and_byte_class(self) -> None:
        tensors = [
            TensorSpec("eligible", (2, 128), "BF16"),
            TensorSpec("tail", (3, 192), "F16"),
            TensorSpec("norm", (128,), "F32"),
        ]
        ledger = complete_hash_storage_ledger(tensors, self.strategy, auxiliary_bytes=17)
        self.assertEqual(ledger["parameters"]["total"], 960)
        self.assertEqual(ledger["parameters"]["fixed_support_hash"], 256)
        self.assertEqual(ledger["parameters"]["retained_exact"], 704)
        self.assertEqual(ledger["tensors"], {
            "total": 3, "fixed_support_hash": 1, "retained_exact": 2,
        })
        retained = align_up(3 * 192 * 2) + align_up(128 * 4)
        expected = 192 + retained + layout_metadata_bytes(tensors) + 17
        self.assertEqual(ledger["bytes"]["complete_bundle"], expected)
        self.assertEqual(
            ledger["effective_bits_per_total_parameter"], expected * 8 / 960
        )

    def test_load_time_projection_mapping_is_shared_by_shape(self) -> None:
        tensors = [
            TensorSpec("gate", (4, 5120), "BF16"),
            TensorSpec("up", (4, 5120), "BF16"),
            TensorSpec("down", (4, 17408), "BF16"),
        ]
        ledger = complete_hash_storage_ledger(tensors, self.strategy)
        compute = ledger["active_compute_metadata_excluded_from_model_storage"]
        self.assertEqual(compute["unique_projection_shapes"], 2)
        self.assertEqual(compute["mapping_bytes"], (5120 + 17408) * 4 * 3)
        self.assertEqual(compute["maximum_per_thread_sketch_scratch_bytes"], 136 * 4)


if __name__ == "__main__":
    unittest.main()
