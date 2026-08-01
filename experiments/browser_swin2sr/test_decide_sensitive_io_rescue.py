import json
import tempfile
import unittest
from pathlib import Path

from decide_sensitive_io_rescue import decide


def receipt(path: Path, *, group_sizes, multiplier, blended, hard_gt, hard_teacher):
    payload = {
        "format": "backdropkit-swin2sr-hard-started-validation-v2",
        "diagnostic_only": True,
        "device": "mps",
        "crop_size": 128,
        "validation_files": [{"path": "/promos/a.png", "sha256": "image"}],
        "checkpoints": [{
            "surface_step": 19000,
            "checkpoint": str(path.with_suffix(".pt")),
            "checkpoint_sha256": path.stem,
            "surface_group_sizes": group_sizes,
            "convolution_learning_rate_multiplier": multiplier,
            "fork_provenance": (
                {"parent_checkpoint_sha256": "parent"}
                if group_sizes
                else None
            ),
            "blended": {"psnr_vs_ground_truth": blended},
            "hard_started": {
                "psnr_vs_ground_truth": hard_gt,
                "psnr_vs_teacher": hard_teacher,
            },
        }],
    }
    path.write_text(json.dumps(payload))


class DecideSensitiveIoRescueTests(unittest.TestCase):
    def test_selects_smallest_lr_only_after_all_checkpoint_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "control.json"
            arm025 = root / "arm025.json"
            arm1 = root / "arm1.json"
            receipt(
                control,
                group_sizes={"swin2sr.first_convolution": 9},
                multiplier=0.25,
                blended=34.0,
                hard_gt=30.0,
                hard_teacher=31.0,
            )
            receipt(
                arm025,
                group_sizes={"swin2sr.first_convolution": 2},
                multiplier=0.25,
                blended=34.0,
                hard_gt=32.0,
                hard_teacher=33.0,
            )
            receipt(
                arm1,
                group_sizes={"swin2sr.first_convolution": 2},
                multiplier=1.0,
                blended=34.1,
                hard_gt=32.05,
                hard_teacher=33.2,
            )

            result = decide(
                control,
                [arm025, arm1],
                step=19000,
                margin_db=0.10,
            )

            self.assertEqual(result["checkpoint_shortlist"], [0.25, 1.0])
            self.assertEqual(result["provisional_smallest_lr"], 0.25)
            self.assertFalse(result["final_selection_authorized"])


if __name__ == "__main__":
    unittest.main()
