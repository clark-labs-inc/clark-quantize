import json
import tempfile
import unittest
from pathlib import Path

from decide_sensitive_io_development import decide
from full_surface_provenance import file_sha256


def samples(offset: float) -> list[dict]:
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
    group_size: int,
    multiplier: float,
    offset: float,
) -> dict:
    rows = samples(offset)
    return {
        "format": "backdropkit-swin2sr-hard-started-stratified-v1",
        "diagnostic_only": True,
        "selection_source": "development",
        "complete_bundle": True,
        "model_promotion_authorized": False,
        "device": "mps",
        "crop_size": 128,
        "bundle_sha256": "bundle-file",
        "bundle_manifest_sha256": "bundle-manifest",
        "q4_reference_sha256": q4_hash,
        "gate_manifest_sha256": gate_hash,
        "surface_step": 19000,
        "state": f"/checkpoint-{state_hash}.pt",
        "state_sha256": state_hash,
        "surface_group_sizes": {
            "swin2sr.first_convolution": group_size,
        },
        "convolution_learning_rate_multiplier": multiplier,
        "fork_provenance": {
            "parent_checkpoint_sha256": "shared-parent",
        },
        "overall": {
            "psnr_vs_ground_truth": {
                "mean": sum(
                    row["psnr_vs_ground_truth"]
                    for row in rows
                ) / len(rows),
            },
        },
        "samples": rows,
    }


class SensitiveIoDevelopmentDecisionTests(unittest.TestCase):
    def fixture(self, root: Path):
        q4_path = root / "q4.json"
        q4_path.write_text(json.dumps({"samples": samples(0.0)}))
        q4_hash = file_sha256(q4_path)
        gate_path = root / "development-gate.json"
        gate_path.write_text(json.dumps({
            "format": (
                "backdropkit-swin2sr-sensitive-io-development-gate-v1"
            ),
            "selection_source": "development",
            "q4_reference_sha256": q4_hash,
            "bootstrap_samples": 200,
            "seed": 7,
            "source_margin_db": 0.10,
            "mean_margin_db": 0.10,
            "source_files": {},
        }))
        gate_hash = file_sha256(gate_path)
        control_path = root / "control.json"
        control_path.write_text(json.dumps(report(
            q4_hash,
            gate_hash,
            state_hash="control",
            group_size=9,
            multiplier=0.25,
            offset=0.0,
        )))
        candidates = []
        for multiplier, state_hash, offset in (
            (0.25, "candidate-025", 1.95),
            (1.0, "candidate-1", 2.0),
        ):
            path = root / f"candidate-{multiplier}.json"
            path.write_text(json.dumps(report(
                q4_hash,
                gate_hash,
                state_hash=state_hash,
                group_size=2,
                multiplier=multiplier,
                offset=offset,
            )))
            candidates.append(path)
        decision_path = root / "checkpoint-decision.json"
        decision_path.write_text(json.dumps({
            "format": (
                "backdropkit-swin2sr-sensitive-io-checkpoint-decision-v1"
            ),
            "final_selection_authorized": False,
            "shared_fork_parent_sha256": "shared-parent",
            "control": {
                "checkpoint_sha256": "control",
            },
            "arms": [
                {
                    "convolution_learning_rate_multiplier": 0.25,
                    "hard_ground_truth_db": 31.0,
                    "checkpoint_sha256": "candidate-025",
                    "checkpoint_eligible": True,
                },
                {
                    "convolution_learning_rate_multiplier": 1.0,
                    "hard_ground_truth_db": 32.0,
                    "checkpoint_sha256": "candidate-1",
                    "checkpoint_eligible": True,
                },
                {
                    "convolution_learning_rate_multiplier": 4.0,
                    "hard_ground_truth_db": 20.0,
                    "checkpoint_sha256": "candidate-4",
                    "checkpoint_eligible": False,
                },
            ],
        }))
        return decision_path, gate_path, control_path, candidates, q4_path

    def test_selects_best_eligible_full_bundle_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self.fixture(root)

            result = decide(
                *inputs,
                bootstrap_samples=200,
                seed=7,
                source_margin_db=0.10,
                mean_margin_db=0.10,
            )

            self.assertTrue(result["sensitive_recipe_selected"])
            self.assertEqual(
                result["selected_convolution_learning_rate_multiplier"],
                0.25,
            )
            self.assertFalse(result["model_promotion_authorized"])
            self.assertTrue(result["final_heldout_locked"])

    def test_requires_every_frozen_top_two_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                decision_path,
                gate_path,
                control_path,
                candidates,
                q4_path,
            ) = self.fixture(root)

            with self.assertRaisesRegex(RuntimeError, "top two"):
                decide(
                    decision_path,
                    gate_path,
                    control_path,
                    candidates[:1],
                    q4_path,
                    bootstrap_samples=200,
                    seed=7,
                    source_margin_db=0.10,
                    mean_margin_db=0.10,
                )


if __name__ == "__main__":
    unittest.main()
