from __future__ import annotations

import unittest

import torch

from clark_quantize.qwen38_layer0_qat import (
    CQ3_BLOCK_BYTES,
    RowShardTrainingConfig,
    aggregate_shard_metrics,
    expected_cq3_payload_bytes,
    pack_cq3_rows,
    train_row_shard,
    unpack_cq3_rows,
)


class Qwen38Layer0QATTests(unittest.TestCase):
    def test_cq3_payload_roundtrip_and_layout(self) -> None:
        codes = (torch.arange(2 * 512, dtype=torch.int64).reshape(2, 512) % 3 - 1).to(torch.int8)
        scales = torch.linspace(0.01, 0.08, 8, dtype=torch.float16).reshape(2, 4)
        payload = pack_cq3_rows(codes, scales)
        self.assertEqual(len(payload), 2 * 2 * CQ3_BLOCK_BYTES)
        self.assertEqual(len(payload), expected_cq3_payload_bytes(2, 512))
        restored_codes, restored_scales = unpack_cq3_rows(payload, rows=2, columns=512)
        self.assertTrue(torch.equal(restored_codes, codes))
        self.assertTrue(torch.equal(restored_scales, scales))

    def test_reserved_code_is_rejected(self) -> None:
        payload = bytearray(expected_cq3_payload_bytes(1, 256))
        payload[0] = 3
        with self.assertRaisesRegex(ValueError, "reserved"):
            unpack_cq3_rows(bytes(payload), rows=1, columns=256)

    def test_row_shard_training_is_deterministic_and_guarded(self) -> None:
        torch.manual_seed(7)
        source = torch.randn(4, 256) * 0.03
        calibration = torch.randn(24, 256)
        validation = torch.randn(12, 256)
        config = RowShardTrainingConfig(steps=2, learning_rate=5e-5)
        first = train_row_shard(source, calibration, validation, config=config)
        second = train_row_shard(source, calibration, validation, config=config)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[0].shape, source.shape)
        self.assertEqual(first[1].shape, (4, 2))

    def test_aggregate_uses_selected_baseline_for_rejected_shards(self) -> None:
        rows = [
            {
                "rows": 2,
                "accepted": True,
                "finite": True,
                "initial_validation_nmse": 0.2,
                "final_validation_nmse": 0.1,
                "initial_validation_squared_error_sum": 20.0,
                "final_validation_squared_error_sum": 10.0,
                "validation_target_squared_sum": 100.0,
                "selected_code_churn_fraction": 0.02,
            },
            {
                "rows": 1,
                "accepted": False,
                "finite": True,
                "initial_validation_nmse": 0.3,
                "final_validation_nmse": 0.4,
                "initial_validation_squared_error_sum": 30.0,
                "final_validation_squared_error_sum": 40.0,
                "validation_target_squared_sum": 100.0,
                "selected_code_churn_fraction": 0.0,
            },
        ]
        summary = aggregate_shard_metrics(rows)
        self.assertEqual(summary["accepted_shards"], 1)
        self.assertAlmostEqual(summary["weighted_initial_validation_nmse"], 0.25)
        self.assertAlmostEqual(summary["weighted_selected_validation_nmse"], 0.20)

    def test_distinct_teacher_inputs_are_supported(self) -> None:
        torch.manual_seed(9)
        source = torch.randn(2, 256) * 0.02
        candidate_train = torch.randn(8, 256)
        candidate_validation = torch.randn(4, 256)
        _, _, metrics = train_row_shard(
            source,
            candidate_train,
            candidate_validation,
            config=RowShardTrainingConfig(steps=1, learning_rate=1e-5),
            teacher_calibration_inputs=candidate_train + 0.01,
            teacher_validation_inputs=candidate_validation - 0.01,
        )
        self.assertTrue(metrics["distinct_teacher_inputs"])


if __name__ == "__main__":
    unittest.main()
