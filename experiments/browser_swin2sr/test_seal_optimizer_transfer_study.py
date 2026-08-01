import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from seal_optimizer_transfer_study import (
    development_quality_screen,
    evaluation_command,
    materialization_command,
    training_commands,
    verify_selected_recipe,
)


class OptimizerTransferStudyTests(unittest.TestCase):
    def decisions(self):
        input_decision = {
            "format": (
                "backdropkit-swin2sr-sensitive-io-development-decision-v1"
            ),
            "sensitive_recipe_selected": True,
            "selected_checkpoint_sha256": "input",
            "model_promotion_authorized": False,
            "final_heldout_locked": True,
        }
        output_decision = {
            "format": "backdropkit-swin2sr-output-conv-decision-v1",
            "output_recipe_selected": True,
            "parent_checkpoint_sha256": "input",
            "selected_checkpoint_sha256": "output",
            "selected_output_group_size": 3,
            "shared_non_output_surface_map": {
                "swin2sr.first_convolution": 2,
            },
            "model_promotion_authorized": False,
            "final_heldout_locked": True,
        }
        return input_decision, output_decision

    def test_requires_exact_selected_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "output.pt"
            checkpoint.write_bytes(b"output")
            input_decision, output_decision = self.decisions()
            from full_surface_provenance import file_sha256
            output_decision["selected_checkpoint_sha256"] = file_sha256(
                checkpoint
            )

            surface_map = verify_selected_recipe(
                input_decision,
                output_decision,
                checkpoint,
            )

            self.assertEqual(surface_map, {
                "swin2sr.first_convolution": 2,
                "upsample.conv": 3,
            })

    def test_rejects_output_from_another_input_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "output.pt"
            checkpoint.write_bytes(b"output")
            input_decision, output_decision = self.decisions()
            output_decision["parent_checkpoint_sha256"] = "other"
            with self.assertRaisesRegex(RuntimeError, "selected input"):
                verify_selected_recipe(
                    input_decision,
                    output_decision,
                    checkpoint,
                )

    def test_commands_compare_continuation_with_fresh_adamw(self):
        args = Namespace(
            validation_data=Path("/promos"),
            out=Path("/study"),
            steps=30000,
            device="mps",
            data=[Path("/images"), Path("/text")],
            qat8_checkpoint=Path("/selected-output.pt"),
        )
        config = {
            "batch_size": 8,
            "crop_size": 128,
            "group_size": 60,
            "quantizer_method": "absmean",
            "conv_quantizer_method": "optimal_mse",
            "conv_grouping": "kernel",
            "schedule_profile": "sensitive_io_v2",
            "core_rotation": "hadamard",
            "learning_rate": 1e-6,
            "convolution_learning_rate_multiplier": 4.0,
            "correction_learning_rate_multiplier": 0.1,
            "loss_recipe": "balanced",
            "hard_shadow_weight_start": 0.05,
            "hard_shadow_weight_end": 0.25,
            "hard_shadow_every": 2,
            "seed": 20260730,
        }
        commands = training_commands(
            args,
            config,
            {
                "swin2sr.first_convolution": 2,
                "upsample.conv": 3,
            },
        )

        self.assertIn("--base-checkpoint", commands["qat8"])
        self.assertNotIn("--initialization", commands["qat8"])
        self.assertEqual(
            commands["adamw"][
                commands["adamw"].index("--optimizer") + 1
            ],
            "adamw",
        )
        self.assertEqual(
            commands["adamw"][
                commands["adamw"].index("--initialization") + 1
            ],
            "fp32",
        )

    def test_evaluation_uses_one_canonical_materialized_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                out=root / "study",
                device="mps",
            )
            materialize = materialization_command(args, "qat8")
            evaluate = evaluation_command(
                args,
                "qat8",
                args.out / "study-manifest.json",
                root / "bundle.json",
                root / "q4.json",
            )

            self.assertIn(
                str(args.out / "qat8" / "run" / "checkpoint-final.pt"),
                materialize,
            )
            manifest = (
                args.out / "qat8" / "canonical" / "materialization.json"
            )
            self.assertIn(str(manifest), evaluate)
            self.assertEqual(
                evaluate[evaluate.index("--materialization") + 1],
                str(manifest),
            )
            self.assertNotIn("--max-samples", evaluate)

    def test_quality_screen_reuses_final_policy_margins_on_development(self):
        q4 = {
            "model": "q4",
            "model_revision": "revision",
            "state_sha256": "q4-hash",
            "artifact_bytes": 100,
            "samples": [{}, {}],
        }
        policy = {
            "format": "backdropkit-swin2sr-promotion-policy-v1",
            "seed": 7,
            "bootstrap_samples": 200,
            "reference": {
                "model": "q4",
                "revision": "revision",
                "model_sha256": "q4-hash",
                "artifact_bytes": 100,
                "samples": 2,
            },
            "paired_quality_gates": {
                "psnr_vs_ground_truth": {
                    "noninferiority_margin": -0.1,
                    "p10_delta_minimum": -0.75,
                },
            },
            "stratified_gates": [{
                "key": "source",
                "metric": "psnr_vs_ground_truth",
                "minimum_samples": 1,
                "mean_delta_minimum": -0.15,
            }],
        }

        screen = development_quality_screen(policy, q4)

        self.assertEqual(screen["seed"], 7)
        self.assertEqual(
            screen["paired_quality_gates"]["psnr_vs_ground_truth"][
                "noninferiority_margin"
            ],
            -0.1,
        )


if __name__ == "__main__":
    unittest.main()
