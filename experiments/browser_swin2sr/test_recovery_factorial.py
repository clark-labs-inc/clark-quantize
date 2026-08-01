import unittest
from argparse import Namespace
from pathlib import Path

from run_recovery_factorial import (
    Arm,
    arms,
    paired_scores,
    promotion_slacks,
    rank_key,
    train_command,
)


class RecoveryFactorialTest(unittest.TestCase):
    def test_matrix_has_twenty_four_unique_preregistered_arms(self):
        matrix = arms()

        self.assertEqual(len(matrix), 24)
        self.assertEqual(len({arm.name for arm in matrix}), 24)

    def test_paired_score_exposes_source_regression(self):
        reference = {
            "samples": [
                {
                    "sha256": "ui",
                    "source": "ui",
                    "psnr_vs_ground_truth": 30.0,
                    "edge_psnr_vs_ground_truth": 31.0,
                    "psnr_vs_teacher": 40.0,
                },
                {
                    "sha256": "text",
                    "source": "text",
                    "psnr_vs_ground_truth": 30.0,
                    "edge_psnr_vs_ground_truth": 31.0,
                    "psnr_vs_teacher": 40.0,
                },
            ]
        }
        candidate = {
            "samples": [
                {
                    "sha256": "ui",
                    "source": "ui",
                    "psnr_vs_ground_truth": 29.5,
                    "edge_psnr_vs_ground_truth": 30.8,
                    "psnr_vs_teacher": 39.0,
                },
                {
                    "sha256": "text",
                    "source": "text",
                    "psnr_vs_ground_truth": 31.0,
                    "edge_psnr_vs_ground_truth": 31.4,
                    "psnr_vs_teacher": 40.5,
                },
            ]
        }

        scores = paired_scores(candidate, reference)

        self.assertAlmostEqual(
            scores["mean_deltas"]["psnr_vs_ground_truth"],
            0.25,
        )
        self.assertAlmostEqual(scores["worst_source_psnr_delta"], -0.5)

    def test_ranking_prefers_source_safety_before_aggregate_gain(self):
        safer = {
            "paired_q4": {
                "worst_source_psnr_delta": -0.1,
                "mean_deltas": {
                    "psnr_vs_ground_truth": 0.0,
                    "psnr_vs_teacher": -0.2,
                    "edge_psnr_vs_ground_truth": 0.0,
                },
            }
        }
        aggregate_only = {
            "paired_q4": {
                "worst_source_psnr_delta": -0.3,
                "mean_deltas": {
                    "psnr_vs_ground_truth": 0.2,
                    "psnr_vs_teacher": 0.1,
                    "edge_psnr_vs_ground_truth": 0.2,
                },
            }
        }

        self.assertGreater(rank_key(safer), rank_key(aggregate_only))

    def test_ranking_prioritizes_the_weakest_promotion_gate(self):
        source_only = {
            "paired_q4": {
                "worst_source_psnr_delta": -0.05,
                "mean_deltas": {
                    "psnr_vs_ground_truth": 0.2,
                    "psnr_vs_teacher": -1.0,
                    "edge_psnr_vs_ground_truth": 0.2,
                },
            },
        }
        gate_balanced = {
            "paired_q4": {
                "worst_source_psnr_delta": -0.1,
                "mean_deltas": {
                    "psnr_vs_ground_truth": 0.0,
                    "psnr_vs_teacher": -0.2,
                    "edge_psnr_vs_ground_truth": 0.0,
                },
            },
        }

        self.assertLess(promotion_slacks(source_only)["teacher_psnr"], 0)
        self.assertGreater(rank_key(gate_balanced), rank_key(source_only))

    def test_explicit_corpus_transition_resets_only_the_data_stream(self):
        args = Namespace(
            base_checkpoint=Path("/tmp/base.pt"),
            out=Path("/tmp/study"),
            resume_stage="stage1",
            validation_data=Path("/tmp/promos"),
            steps=2000,
            batch_size=8,
            crop_size=128,
            learning_rate=1e-6,
            scalar_lr_multiplier=100,
            localized_stage_steps=100,
            validate_every=250,
            checkpoint_every=500,
            device="mps",
            seed=20260729,
            data=[Path("/tmp/ui"), Path("/tmp/glyph-safe")],
            reset_data_stream=True,
        )
        command = train_command(
            args,
            Arm(60, False, "frozen", "balanced"),
            Path("/tmp/stage2"),
            "bonsai_qat8_muon",
        )

        self.assertIn("--resume-optimizer", command)
        self.assertIn("--reset-data-stream", command)
        self.assertIn(
            "/tmp/study/stage1/"
            "g60__plain__frozen__balanced/checkpoint-final.pt",
            command,
        )


if __name__ == "__main__":
    unittest.main()
