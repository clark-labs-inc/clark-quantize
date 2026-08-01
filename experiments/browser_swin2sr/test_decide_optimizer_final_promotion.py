import gzip
import json
import tempfile
import unittest
from pathlib import Path

from decide_optimizer_final_promotion import (
    FIXTURE_SOURCE_FILES,
    promotion_record,
    validate_final_chain,
    validate_runtime_fixture,
)
from final_artifact_contract import file_sha256


class OptimizerFinalPromotionTests(unittest.TestCase):
    def final_chain(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        hard = root / "dense.pt"
        runtime = root / "runtime.pt"
        portable = root / "model.onnx"
        package = root / "model.onnx.gz"
        materialization = root / "materialization.json"
        ledger = root / "model.ledger.json"
        artifact_precommit = root / "artifact-precommit.json"
        candidate_path = root / "final-candidate.json"
        policy = root / "policy.json"
        reference = root / "q4.json"
        hard.write_bytes(b"hard")
        runtime.write_bytes(b"runtime")
        portable.write_bytes(b"portable")
        package.write_bytes(gzip.compress(portable.read_bytes(), mtime=0))
        materialization.write_text(json.dumps({
            "format": "backdropkit-swin2sr-surface-materialization-v1",
            "all_target_surfaces_materialized_hard": True,
            "dense_state": {
                "path": str(hard.resolve()),
                "file_sha256": file_sha256(hard),
            },
            "runtime_state": {
                "path": str(runtime.resolve()),
                "file_sha256": file_sha256(runtime),
            },
        }))
        ledger.write_text(json.dumps({
            "format": "clark-portable-onnx-ternary-v2",
            "materialization_manifest_sha256": file_sha256(materialization),
            "hard_state_sha256": file_sha256(hard),
            "runtime_state_file_sha256": file_sha256(runtime),
            "model_sha256": file_sha256(portable),
            "runtime_package_sha256": file_sha256(package),
            "runtime_package_compression": "gzip",
            "scope": "all_matrix_and_conv",
        }))
        artifact_precommit.write_text('{"format":"artifact"}\n')
        policy.write_text('{"format":"policy"}\n')
        reference.write_text('{"format":"q4"}\n')
        candidate = {
            "format": "backdropkit-swin2sr-final-candidate-v1",
            "final_evaluation_eligible": True,
            "model_promotion_authorized": False,
            "lane": "qat8",
            "optimizer_decision": "decision.json",
            "optimizer_decision_sha256": "decision-sha",
            "materialization": str(materialization),
            "materialization_sha256": file_sha256(materialization),
            "hard_state": str(hard),
            "hard_state_sha256": file_sha256(hard),
            "runtime_state": str(runtime),
            "runtime_state_sha256": file_sha256(runtime),
            "portable_onnx": str(portable),
            "portable_onnx_bytes": portable.stat().st_size,
            "portable_onnx_sha256": file_sha256(portable),
            "portable_ledger": str(ledger),
            "portable_ledger_sha256": file_sha256(ledger),
            "runtime_package": str(package),
            "runtime_package_bytes": package.stat().st_size,
            "runtime_package_sha256": file_sha256(package),
            "runtime_package_compression": "gzip",
            "artifact_precommit": str(artifact_precommit),
            "artifact_precommit_sha256": file_sha256(artifact_precommit),
        }
        candidate_path.write_text(json.dumps(candidate))

        final_root = root / "final"
        final_root.mkdir()
        precommit = final_root / "final-precommit.json"
        report = final_root / "candidate-metrics.json"
        receipt = final_root / "final-evaluation-receipt.json"
        precommit.write_text(json.dumps({
            "format": "backdropkit-swin2sr-final-precommit-v2",
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": file_sha256(candidate_path),
            "optimizer_decision": candidate["optimizer_decision"],
            "optimizer_decision_sha256": candidate[
                "optimizer_decision_sha256"
            ],
            "selected_optimizer": candidate["lane"],
            "materialization": candidate["materialization"],
            "materialization_sha256": candidate["materialization_sha256"],
            "hard_state": candidate["hard_state"],
            "hard_state_sha256": candidate["hard_state_sha256"],
            "runtime_state": candidate["runtime_state"],
            "runtime_state_sha256": candidate["runtime_state_sha256"],
            "runtime_package": candidate["runtime_package"],
            "runtime_package_sha256": candidate["runtime_package_sha256"],
            "runtime_package_bytes": candidate["runtime_package_bytes"],
            "artifact_precommit": candidate["artifact_precommit"],
            "artifact_precommit_sha256": candidate[
                "artifact_precommit_sha256"
            ],
            "promotion_policy": str(policy),
            "promotion_policy_sha256": file_sha256(policy),
            "final_q4": str(reference),
            "final_q4_sha256": file_sha256(reference),
        }))
        report_payload = {
            "selection_source": "development",
            "selection_lane": "qat8",
            "state_sha256": candidate["hard_state_sha256"],
            "hard_state_sha256": candidate["hard_state_sha256"],
            "materialization_sha256": candidate["materialization_sha256"],
            "runtime_state_sha256": candidate["runtime_state_sha256"],
            "runtime_artifact_sha256": candidate[
                "runtime_package_sha256"
            ],
            "runtime_artifact_bytes": candidate["runtime_package_bytes"],
            "runtime_artifact_compression": "gzip",
            "runtime_expanded_artifact_sha256": candidate[
                "portable_onnx_sha256"
            ],
            "runtime_expanded_artifact_bytes": candidate[
                "portable_onnx_bytes"
            ],
            "overall": {"samples": 1},
            "samples": [{"sha256": "sample"}],
        }
        report.write_text(json.dumps(report_payload))
        receipt.write_text(json.dumps({
            "format": "backdropkit-swin2sr-final-evaluation-v1",
            "precommit_sha256": file_sha256(precommit),
            "candidate_metrics": str(report),
            "candidate_metrics_sha256": file_sha256(report),
            "samples": 1,
            "overall": report_payload["overall"],
        }))
        return candidate_path, receipt, policy, reference, report

    def test_final_chain_binds_metrics_to_the_canonical_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.final_chain(Path(temporary))
            candidate, report_path, report = validate_final_chain(
                fixture[0],
                fixture[1],
                fixture[2],
                fixture[3],
            )
            self.assertEqual(candidate["lane"], "qat8")
            self.assertEqual(report_path, fixture[4])
            self.assertEqual(len(report["samples"]), 1)

            fixture[4].write_text('{"samples":[]}\n')
            with self.assertRaisesRegex(ValueError, "metrics identity changed"):
                validate_final_chain(
                    fixture[0],
                    fixture[1],
                    fixture[2],
                    fixture[3],
                )

    def test_partial_final_result_never_authorizes_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for name in (
                "candidate",
                "receipt",
                "policy",
                "q4",
                "fixture",
                "runtime",
            ):
                source = root / f"{name}.json"
                source.write_text("{}\n")
                inputs.append(source)
            result = {
                "passed": False,
                "paired_quality": {"psnr": {"passed": True}},
                "stratified_quality": [{"passed": True}],
                "artifact": {"passed": True},
                "runtime": {"passed": False},
            }
            record = promotion_record(
                candidate_path=inputs[0],
                receipt_path=inputs[1],
                policy_path=inputs[2],
                reference_path=inputs[3],
                fixture_path=inputs[4],
                runtime_path=inputs[5],
                result=result,
                source_files={"evaluator.py": "sha"},
            )

            self.assertTrue(record["quality_passed"])
            self.assertTrue(record["artifact_passed"])
            self.assertFalse(record["runtime_passed"])
            self.assertFalse(record["model_promotion_authorized"])

    def test_runtime_bundle_must_use_the_exact_candidate_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path, _, _, _, _ = self.final_chain(root)
            candidate = json.loads(candidate_path.read_text())
            fixture_path = root / "runtime-fixture.json"
            runtime_path = root / "runtime-receipts.json"
            fixture = {
                "hard_state_sha256": candidate["hard_state_sha256"],
                "model": candidate["portable_onnx"],
                "model_sha256": candidate["portable_onnx_sha256"],
                "model_package": candidate["runtime_package"],
                "model_package_compression": "gzip",
                "model_package_bytes": candidate["runtime_package_bytes"],
                "model_package_sha256": candidate[
                    "runtime_package_sha256"
                ],
                "input_shape": [1, 3, 128, 128],
                "source_files": {
                    name: file_sha256(
                        Path(__file__).resolve().parents[2] / name
                    )
                    for name in FIXTURE_SOURCE_FILES
                },
            }
            fixture_path.write_text(json.dumps(fixture))
            runtime_path.write_text(json.dumps({
                "model_sha256": candidate["runtime_package_sha256"],
                "fixture_sha256": file_sha256(fixture_path),
                "fixture_hard_state_sha256": candidate[
                    "hard_state_sha256"
                ],
                "receipts": [{
                    "fixtureSha256": file_sha256(fixture_path),
                    "fixtureHardStateSha256": candidate[
                        "hard_state_sha256"
                    ],
                }],
            }))
            policy = {"runtime_input_shape": [1, 3, 128, 128]}

            validate_runtime_fixture(
                candidate,
                fixture_path,
                runtime_path,
                policy,
            )
            fixture["input_shape"] = [1, 3, 64, 64]
            fixture_path.write_text(json.dumps(fixture))
            with self.assertRaisesRegex(ValueError, "candidate differ"):
                validate_runtime_fixture(
                    candidate,
                    fixture_path,
                    runtime_path,
                    policy,
                )


if __name__ == "__main__":
    unittest.main()
