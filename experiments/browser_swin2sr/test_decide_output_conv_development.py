import json
import tempfile
import unittest
from pathlib import Path

from decide_output_conv_development import decide
from full_surface_provenance import file_sha256


def sample_rows(offset: float) -> list[dict]:
    return [
        {
            "sha256": f"sample-{index}",
            "source": "rico" if index < 2 else "flores",
            "psnr_vs_ground_truth": 30.0 + index + offset,
            "edge_psnr_vs_ground_truth": 32.0 + index + offset,
            "psnr_vs_teacher": 40.0 + index + offset,
        }
        for index in range(4)
    ]


def report(
    q4_hash: str,
    gate_hash: str,
    *,
    state_hash: str,
    output_group: int,
    offset: float,
) -> dict:
    rows = sample_rows(offset)
    return {
        "format": "backdropkit-swin2sr-hard-started-stratified-v1",
        "diagnostic_only": True,
        "selection_source": "development",
        "complete_bundle": True,
        "model_promotion_authorized": False,
        "device": "mps",
        "crop_size": 128,
        "bundle_sha256": "bundle",
        "bundle_manifest_sha256": "manifest",
        "q4_reference_sha256": q4_hash,
        "gate_manifest_sha256": gate_hash,
        "surface_step": 24000,
        "state": f"/{state_hash}.pt",
        "state_sha256": state_hash,
        "surface_group_sizes": {
            "swin2sr.first_convolution": 2,
            "upsample.conv": output_group,
        },
        "convolution_learning_rate_multiplier": 1.0,
        "fork_provenance": {"parent_checkpoint_sha256": "parent"},
        "scheduled_surface_summary": {
            "surfaces": 204,
            "fully_hard_surfaces": 204,
            "transitioning_surfaces": 0,
        },
        "hard_started_surface_summary": {
            "surfaces": 204,
            "fully_hard_surfaces": 204,
            "transitioning_surfaces": 0,
        },
        "overall": {
            metric: {
                "mean": sum(row[metric] for row in rows) / len(rows),
            }
            for metric in (
                "psnr_vs_ground_truth",
                "edge_psnr_vs_ground_truth",
                "psnr_vs_teacher",
            )
        },
        "samples": rows,
    }


class OutputConvDecisionTests(unittest.TestCase):
    def fixture(self, root: Path, *, g2_offset=1.0, g3_offset=0.95):
        q4 = root / "q4.json"
        q4.write_text(json.dumps({"samples": sample_rows(0.0)}))
        q4_hash = file_sha256(q4)
        gate = root / "gate.json"
        gate.write_text(json.dumps({
            "format": "backdropkit-swin2sr-output-conv-development-gate-v1",
            "selection_source": "development",
            "model_promotion_authorized": False,
            "final_heldout_locked": True,
            "q4_reference_sha256": q4_hash,
            "bootstrap_samples": 200,
            "seed": 7,
            "source_margin_db": 0.10,
            "mean_margin_db": 0.10,
            "candidate_group_sizes": [2, 3],
            "control_group_size": 9,
            "selection_step": 24000,
            "parent_checkpoint_sha256": "parent",
            "source_files": {},
        }))
        gate_hash = file_sha256(gate)
        paths = {}
        for name, group, offset in (
            ("control", 9, 0.0),
            ("g2", 2, g2_offset),
            ("g3", 3, g3_offset),
        ):
            current = root / f"{name}.json"
            current.write_text(json.dumps(report(
                q4_hash,
                gate_hash,
                state_hash=name,
                output_group=group,
                offset=offset,
            )))
            paths[name] = current
        return gate, paths, q4

    def test_selects_larger_group_when_within_every_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate, paths, q4 = self.fixture(Path(temporary))

            result = decide(
                gate,
                paths["control"],
                [paths["g2"], paths["g3"]],
                q4,
                bootstrap_samples=200,
                seed=7,
                source_margin_db=0.10,
                mean_margin_db=0.10,
            )

            self.assertTrue(result["output_recipe_selected"])
            self.assertEqual(result["selected_output_group_size"], 3)
            self.assertFalse(result["model_promotion_authorized"])

    def test_selects_g2_when_g3_misses_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate, paths, q4 = self.fixture(
                Path(temporary),
                g2_offset=1.0,
                g3_offset=0.7,
            )

            result = decide(
                gate,
                paths["control"],
                [paths["g2"], paths["g3"]],
                q4,
                bootstrap_samples=200,
                seed=7,
                source_margin_db=0.10,
                mean_margin_db=0.10,
            )

            self.assertEqual(result["selected_output_group_size"], 2)

    def test_selects_none_when_rescues_do_not_beat_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate, paths, q4 = self.fixture(
                Path(temporary),
                g2_offset=-0.2,
                g3_offset=-0.1,
            )

            result = decide(
                gate,
                paths["control"],
                [paths["g2"], paths["g3"]],
                q4,
                bootstrap_samples=200,
                seed=7,
                source_margin_db=0.10,
                mean_margin_db=0.10,
            )

            self.assertFalse(result["output_recipe_selected"])
            self.assertIsNone(result["selected_output_group_size"])


if __name__ == "__main__":
    unittest.main()
