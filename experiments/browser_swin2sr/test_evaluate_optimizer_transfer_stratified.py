import hashlib
import tempfile
import unittest
from pathlib import Path

from evaluate_optimizer_transfer_stratified import (
    arm_identity,
    verify_materialization_identity,
    verify_optimizer_study,
)


def study() -> dict:
    return {
        "format": "backdropkit-swin2sr-optimizer-transfer-v1",
        "selection_source": "development",
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
        "arms": {"qat8": {}, "adamw": {}},
        "surface_group_sizes": {
            "swin2sr.first_convolution": 2,
            "upsample.conv": 3,
        },
        "qat8_checkpoint_sha256": "selected",
    }


def payload(optimizer: str) -> dict:
    return {
        "surface_step": 30000,
        "quantization": {
            "surface_group_sizes": {
                "swin2sr.first_convolution": 2,
                "upsample.conv": 3,
            },
        },
        "surface_training": {
            "optimizer": {"name": optimizer},
            "resume_checkpoint": "/selected.pt",
            "base_checkpoint": None,
            "initialization": "fp32",
        },
    }


class OptimizerTransferEvaluationTests(unittest.TestCase):
    def test_requires_locked_exact_study(self):
        current = study()
        verify_optimizer_study(current)
        current["final_heldout_locked"] = False
        with self.assertRaisesRegex(RuntimeError, "lock"):
            verify_optimizer_study(current)

    def test_identifies_fresh_adamw(self):
        self.assertEqual(
            arm_identity(payload("adamw"), study(), Path("/")),
            "adamw",
        )

    def test_identifies_selected_qat8_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "selected.pt"
            selected.write_bytes(b"selected")
            current = study()
            from full_surface_provenance import file_sha256
            current["qat8_checkpoint_sha256"] = file_sha256(selected)
            qat8 = payload("bonsai_qat8_muon")
            qat8["surface_training"]["resume_checkpoint"] = str(selected)

            self.assertEqual(
                arm_identity(qat8, current, root),
                "qat8",
            )

    def test_requires_one_canonical_all_hard_materialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "optimizer.pt"
            checkpoint.write_bytes(b"optimizer")
            current = payload("adamw")
            materialization = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(b"optimizer").hexdigest(),
                "surface_step": 30000,
                "all_target_surfaces_materialized_hard": True,
                "surfaces": {
                    f"surface.{index}.weight": {}
                    for index in range(204)
                },
                "quantization": current["quantization"],
                "surface_training": current["surface_training"],
            }

            summary = verify_materialization_identity(
                checkpoint,
                current,
                materialization,
            )

            self.assertEqual(summary["fully_hard_surfaces"], 204)
            self.assertEqual(summary["transitioning_surfaces"], 0)

    def test_rejects_materialization_from_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "optimizer.pt"
            checkpoint.write_bytes(b"optimizer")
            current = payload("adamw")
            materialization = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": "another",
                "surface_step": 30000,
                "all_target_surfaces_materialized_hard": True,
                "surfaces": {
                    f"surface.{index}.weight": {}
                    for index in range(204)
                },
                "quantization": current["quantization"],
                "surface_training": current["surface_training"],
            }

            with self.assertRaisesRegex(RuntimeError, "another optimizer"):
                verify_materialization_identity(
                    checkpoint,
                    current,
                    materialization,
                )


if __name__ == "__main__":
    unittest.main()
