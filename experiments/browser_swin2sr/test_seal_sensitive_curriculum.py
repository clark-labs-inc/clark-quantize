import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from seal_sensitive_curriculum import (
    training_command,
    verify_development_contract,
)


class SensitiveCurriculumSealTests(unittest.TestCase):
    def test_command_starts_from_fp32_and_includes_sensitive_controls(self):
        args = Namespace(
            validation_data=Path("/promos"),
            out=Path("/run"),
            steps=30000,
            schedule_steps=24000,
            batch_size=8,
            crop_size=128,
            group_size=60,
            quantizer_method="absmean",
            conv_quantizer_method="optimal_mse",
            conv_grouping="kernel",
            schedule_profile="sensitive_io_v2",
            core_rotation="hadamard",
            optimizer="adamw",
            learning_rate=1e-6,
            convolution_learning_rate_multiplier=0.25,
            correction_learning_rate_multiplier=0.10,
            surface_group_sizes={"swin2sr.first_convolution": 2},
            loss_recipe="balanced",
            validate_every=500,
            checkpoint_every=1000,
            hard_shadow_weight_start=0.05,
            hard_shadow_weight_end=0.25,
            hard_shadow_every=2,
            device="mps",
            seed=20260730,
            data=[Path("/ui"), Path("/text")],
        )

        command = training_command(args)

        self.assertEqual(
            command[command.index("--initialization") + 1],
            "fp32",
        )
        self.assertIn("sensitive_io_v2", command)
        self.assertIn("optimal_mse", command)
        self.assertIn("0.25", command)
        self.assertIn("adamw", command)
        self.assertIn("swin2sr.first_convolution=2", command)

    def test_development_bundle_must_match_q4_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.json"
            q4 = root / "q4.json"
            bundle.write_text(json.dumps({
                "manifest_sha256": "bundle",
                "samples": 1,
            }))
            q4.write_text(json.dumps({
                "bundle_manifest_sha256": "bundle",
                "device": "mps",
                "samples": [{}],
            }))

            verify_development_contract(bundle, q4, "mps")
            with self.assertRaisesRegex(ValueError, "device"):
                verify_development_contract(bundle, q4, "cpu")


if __name__ == "__main__":
    unittest.main()
