import tempfile
import unittest
import json
from argparse import Namespace
from pathlib import Path

from full_surface_artifacts import (
    materialization_command,
    portable_export_command,
)
from full_surface_provenance import selection_provenance
from run_full_surface_comparison import (
    BaseQuantization,
    evaluation_command,
    export_command,
    rank_key,
    training_command,
    verify_eval_contract,
    verification_command,
)


class FullSurfaceComparisonTests(unittest.TestCase):
    def arguments(self, root: Path) -> Namespace:
        return Namespace(
            recovered_checkpoint=root / "recovered.pt",
            data=[root / "ui", root / "text"],
            data_receipt=[],
            validation_data=root / "promos",
            steps=4000,
            schedule_steps=3000,
            batch_size=8,
            crop_size=128,
            learning_rate=1e-6,
            loss_recipe="balanced",
            validate_every=250,
            checkpoint_every=500,
            device="mps",
            eval_device="cpu",
            seed=20260729,
            bundle=root / "bundle.json",
            base_onnx=root / "base.onnx",
        )

    def test_lanes_share_every_training_control_except_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root)
            base = BaseQuantization("absmean", 30, "hadamard")
            fresh = training_command(args, "fp32", root / "fp32", base)
            recovered = training_command(
                args,
                "recovered_core",
                root / "recovered_core",
                base,
            )

            self.assertIn("absmean", fresh)
            self.assertIn("30", fresh)
            self.assertIn("hadamard", fresh)
            self.assertNotIn("--base-checkpoint", fresh)
            self.assertIn("--base-checkpoint", recovered)
            self.assertEqual(fresh[fresh.index("--device") + 1], "mps")
            fresh_controls = fresh[:]
            recovered_controls = recovered[:]
            fresh_controls[fresh_controls.index(str(root / "fp32"))] = "LANE"
            recovered_controls[
                recovered_controls.index(str(root / "recovered_core"))
            ] = "LANE"
            fresh_controls[-2:] = ["--initialization", "LANE"]
            recovered_controls[-4:] = ["--initialization", "LANE"]
            self.assertEqual(fresh_controls, recovered_controls)

    def test_evaluation_uses_the_sealed_bundle_and_dense_hard_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = evaluation_command(self.arguments(root), root / "lane")

            self.assertIn(str(root / "bundle.json"), command)
            self.assertIn(
                str(root / "lane" / "canonical" / "dense-hard.pt"),
                command,
            )
            self.assertEqual(command[command.index("--device") + 1], "cpu")

    def test_materialization_export_and_verification_share_canonical_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "lane"
            materialize = materialization_command(
                self.arguments(root),
                destination,
            )
            export = export_command(destination)
            verification = verification_command(
                self.arguments(root),
                destination,
            )

            self.assertIn(
                str(destination / "checkpoint-final.pt"),
                materialize,
            )
            self.assertIn(
                str(destination / "canonical" / "materialization.json"),
                export,
            )
            self.assertIn(
                str(destination / "canonical" / "dense-hard.pt"),
                verification,
            )

    def test_evaluation_device_must_match_q4_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root)
            args.bundle.write_text(json.dumps({
                "manifest_sha256": "bundle",
                "samples": 1,
            }))
            q4 = {
                "bundle_manifest_sha256": "bundle",
                "samples": [{}],
                "device": "mps",
            }
            with self.assertRaisesRegex(ValueError, "evaluation device"):
                verify_eval_contract(args, q4)
            args.eval_device = "mps"
            verify_eval_contract(args, q4)

    def test_portable_export_covers_every_matrix_and_convolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = portable_export_command(
                self.arguments(root),
                root / "lane",
            )

            self.assertIn(str(root / "base.onnx"), command)
            self.assertIn(
                str(root / "lane" / "canonical" / "materialization.json"),
                command,
            )
            self.assertIn("all_matrix_and_conv", command)

    def test_ranking_prefers_weakest_source_before_aggregate(self):
        safer = {
            "paired_q4": {
                "worst_source_psnr_delta": -0.05,
                "mean_deltas": {
                    "psnr_vs_ground_truth": 0.0,
                    "edge_psnr_vs_ground_truth": 0.0,
                    "psnr_vs_teacher": -0.1,
                },
            },
        }
        aggregate = {
            "paired_q4": {
                "worst_source_psnr_delta": -0.2,
                "mean_deltas": {
                    "psnr_vs_ground_truth": 1.0,
                    "edge_psnr_vs_ground_truth": 1.0,
                    "psnr_vs_teacher": 1.0,
                },
            },
        }

        self.assertGreater(rank_key(safer), rank_key(aggregate))

    def test_summary_seals_development_selection_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root)
            args.bundle.write_text(json.dumps({
                "manifest_sha256": "development-bundle",
                "samples": 1,
            }))
            args.q4_reference = root / "q4.json"
            args.q4_reference.write_text(json.dumps({
                "state_sha256": "development-q4",
            }))
            manifest = root / "study-manifest.json"
            manifest.write_text('{"selection_source":"development"}\n')
            provenance = selection_provenance(
                args,
                {"state_sha256": "development-q4"},
                manifest,
            )

            self.assertEqual(provenance["selection_source"], "development")
            self.assertEqual(
                provenance["bundle_manifest_sha256"],
                "development-bundle",
            )
            self.assertEqual(
                provenance["q4_reference_state_sha256"],
                "development-q4",
            )


if __name__ == "__main__":
    unittest.main()
