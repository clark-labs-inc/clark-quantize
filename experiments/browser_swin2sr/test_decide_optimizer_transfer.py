import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from decide_optimizer_transfer import decide
from full_surface_provenance import file_sha256


def samples(offset: float) -> list[dict]:
    return [
        {
            "sha256": f"sample-{index}",
            "source": "rico" if index < 2 else "flores",
            "script": "latin" if index % 2 == 0 else "arabic",
            "language": f"language-{index}",
            "psnr_vs_ground_truth": 30.0 + index + offset,
            "edge_psnr_vs_ground_truth": 32.0 + index + offset,
            "psnr_vs_teacher": 40.0 + index + offset,
        }
        for index in range(4)
    ]


class OptimizerTransferDecisionTests(unittest.TestCase):
    def fixture(
        self,
        root: Path,
        *,
        qat8_offset: float,
        adamw_offset: float,
    ) -> tuple[Path, list[Path], Path, Path]:
        q4 = root / "q4.json"
        q4.write_text(json.dumps({"samples": samples(0.0)}))
        policy = root / "policy.json"
        policy.write_text(json.dumps({
            "format": "backdropkit-swin2sr-promotion-policy-v1",
            "seed": 7,
            "bootstrap_samples": 200,
            "paired_quality_gates": {
                metric: {
                    "noninferiority_margin": -0.1,
                    "p10_delta_minimum": -0.75,
                }
                for metric in (
                    "psnr_vs_ground_truth",
                    "edge_psnr_vs_ground_truth",
                    "psnr_vs_teacher",
                )
            },
            "stratified_gates": [{
                "key": "source",
                "metric": "psnr_vs_ground_truth",
                "minimum_samples": 2,
                "mean_delta_minimum": -0.15,
            }],
        }))
        study = root / "study.json"
        study.write_text(json.dumps({
            "format": "backdropkit-swin2sr-optimizer-transfer-v1",
            "selection_source": "development",
            "model_promotion_authorized": False,
            "final_heldout_locked": True,
            "arms": {"qat8": {}, "adamw": {}},
            "q4_reference_sha256": file_sha256(q4),
            "promotion_policy_sha256": file_sha256(policy),
            "development_quality_screen": {
                "seed": 7,
                "bootstrap_samples": 200,
                "paired_quality_gates": {
                    metric: {
                        "noninferiority_margin": -0.1,
                        "p10_delta_minimum": -0.75,
                    }
                    for metric in (
                        "psnr_vs_ground_truth",
                        "edge_psnr_vs_ground_truth",
                        "psnr_vs_teacher",
                    )
                },
                "stratified_gates": [{
                    "key": "source",
                    "metric": "psnr_vs_ground_truth",
                    "minimum_samples": 2,
                    "mean_delta_minimum": -0.15,
                }],
            },
            "selection_step": 30000,
            "ranking_policy": [
                "worst_source_psnr_delta",
                "overall_psnr_delta",
                "edge_psnr_delta",
                "teacher_psnr_delta",
                "optimizer_arm_stable_tiebreak",
            ],
        }))
        study_hash = file_sha256(study)
        reports = []
        for arm, offset in (
            ("qat8", qat8_offset),
            ("adamw", adamw_offset),
        ):
            arm_root = root / arm
            arm_root.mkdir()
            checkpoint = arm_root / "checkpoint.pt"
            checkpoint.write_bytes(f"{arm}-checkpoint".encode())
            dense = arm_root / "dense.pt"
            dense.write_bytes(f"{arm}-dense".encode())
            runtime = arm_root / "runtime.pt"
            runtime.write_bytes(f"{arm}-runtime".encode())
            materialization = arm_root / "materialization.json"
            materialization.write_text(json.dumps({
                "format": (
                    "backdropkit-swin2sr-surface-materialization-v1"
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "all_target_surfaces_materialized_hard": True,
                "dense_state": {
                    "path": str(dense),
                    "file_sha256": file_sha256(dense),
                    "tensor_sha256": hashlib.sha256(
                        f"{arm}-tensor".encode()
                    ).hexdigest(),
                },
                "runtime_state": {
                    "path": str(runtime),
                    "file_sha256": file_sha256(runtime),
                },
            }))
            rows = samples(offset)
            report = arm_root / "metrics.json"
            report.write_text(json.dumps({
                "format": (
                    "backdropkit-swin2sr-optimizer-transfer-stratified-v1"
                ),
                "diagnostic_only": True,
                "selection_source": "development",
                "complete_bundle": True,
                "model_promotion_authorized": False,
                "final_heldout_locked": True,
                "optimizer_arm": arm,
                "model": "teacher",
                "model_revision": "revision",
                "study_manifest_sha256": study_hash,
                "device": "mps",
                "crop_size": 128,
                "bundle_sha256": "bundle",
                "bundle_manifest_sha256": "bundle-manifest",
                "q4_reference_sha256": file_sha256(q4),
                "surface_step": 30000,
                "surface_group_sizes": {
                    "swin2sr.first_convolution": 2,
                    "upsample.conv": 3,
                },
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "materialization": str(materialization),
                "materialization_sha256": file_sha256(materialization),
                "state": str(dense),
                "state_sha256": file_sha256(dense),
                "state_tensor_sha256": json.loads(
                    materialization.read_text()
                )["dense_state"]["tensor_sha256"],
                "surface_summary": {
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
            }))
            reports.append(report)
        return study, reports, q4, policy

    def test_selects_best_eligible_canonical_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            study, reports, q4, policy = self.fixture(
                Path(temporary),
                qat8_offset=0.2,
                adamw_offset=0.1,
            )

            result = decide(study, reports, q4, policy)

            self.assertTrue(result["optimizer_recipe_selected"])
            self.assertEqual(result["selected_optimizer"], "qat8")
            self.assertTrue(result["selected_materialization_sha256"])
            self.assertFalse(result["model_promotion_authorized"])
            self.assertTrue(result["final_heldout_locked"])

    def test_selects_none_when_both_fail_development_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            study, reports, q4, policy = self.fixture(
                Path(temporary),
                qat8_offset=-1.0,
                adamw_offset=-1.2,
            )

            result = decide(study, reports, q4, policy)

            self.assertFalse(result["optimizer_recipe_selected"])
            self.assertIsNone(result["selected_optimizer"])


if __name__ == "__main__":
    unittest.main()
