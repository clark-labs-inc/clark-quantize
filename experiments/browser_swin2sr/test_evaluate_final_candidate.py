import hashlib
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_final_candidate import (
    candidate_report,
    selected_candidate,
    validate_artifacts,
    validate_development_selection,
    verify_completed_receipt,
    write_or_verify_precommit,
)


class FinalCandidateEvaluationTests(unittest.TestCase):
    def winner(self) -> dict:
        return {
            "lane": "recovered_core",
            "materialization": "/tmp/materialization.json",
            "materialization_sha256": "materialization",
            "hard_state": "/tmp/hard.pt",
            "hard_state_sha256": "hard",
            "runtime_state": "/tmp/runtime.pt",
            "runtime_state_sha256": "runtime",
            "portable_onnx": "/tmp/model.onnx",
            "portable_onnx_bytes": 42,
            "portable_onnx_sha256": "portable",
            "portable_ledger": "/tmp/model.ledger.json",
            "portable_ledger_sha256": "ledger",
            "runtime_package": "/tmp/model.onnx.gz",
            "runtime_package_bytes": 21,
            "runtime_package_sha256": "package",
            "runtime_package_compression": "gzip",
        }

    def test_selection_is_committed_from_development_ranking(self):
        summary = {
            "format": "backdropkit-swin2sr-full-surface-summary-v1",
            "failed": 0,
            "ranking": [self.winner()],
        }
        self.assertEqual(selected_candidate(summary)["lane"], "recovered_core")

    def test_legacy_winner_without_materialization_is_rejected(self):
        winner = self.winner()
        del winner["materialization"]
        summary = {
            "format": "backdropkit-swin2sr-full-surface-summary-v1",
            "failed": 0,
            "ranking": [winner],
        }
        with self.assertRaisesRegex(ValueError, "materialization"):
            selected_candidate(summary)

    def test_candidate_report_preserves_quality_and_runtime_identities(self):
        metrics = {
            "state_sha256": "hard",
            "bundle_manifest_sha256": "bundle",
            "device": "cpu",
            "samples": [{}],
        }
        report = candidate_report(
            metrics,
            self.winner(),
            {"manifest_sha256": "bundle", "samples": 1},
            {"device": "cpu"},
        )
        self.assertEqual(report["state_sha256"], "hard")
        self.assertEqual(report["hard_state_sha256"], "hard")
        self.assertEqual(report["runtime_artifact_sha256"], "package")
        self.assertEqual(report["runtime_artifact_bytes"], 21)
        self.assertEqual(report["runtime_artifact_compression"], "gzip")
        self.assertEqual(
            report["runtime_expanded_artifact_sha256"],
            "portable",
        )
        self.assertEqual(
            report["materialization_sha256"],
            "materialization",
        )

    def test_candidate_report_rejects_a_different_hard_state(self):
        metrics = {
            "state_sha256": "different",
            "bundle_manifest_sha256": "bundle",
            "device": "cpu",
            "samples": [{}],
        }
        with self.assertRaisesRegex(ValueError, "canonical hard state"):
            candidate_report(
                metrics,
                self.winner(),
                {"manifest_sha256": "bundle", "samples": 1},
                {"device": "cpu"},
            )

    def test_artifacts_must_share_one_canonical_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hard = root / "hard.pt"
            runtime = root / "runtime.pt"
            portable = root / "model.onnx"
            package = root / "model.onnx.gz"
            manifest = root / "materialization.json"
            ledger = root / "model.ledger.json"
            hard.write_bytes(b"hard")
            runtime.write_bytes(b"runtime")
            portable.write_bytes(b"portable")
            package.write_bytes(gzip.compress(portable.read_bytes(), mtime=0))

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            manifest.write_text(json.dumps({
                "format": "backdropkit-swin2sr-surface-materialization-v1",
                "all_target_surfaces_materialized_hard": True,
                "dense_state": {
                    "path": str(hard.resolve()),
                    "file_sha256": digest(hard),
                },
                "runtime_state": {
                    "path": str(runtime.resolve()),
                    "file_sha256": digest(runtime),
                },
            }))
            ledger.write_text("{}")
            candidate = {
                "materialization": str(manifest),
                "materialization_sha256": digest(manifest),
                "hard_state": str(hard),
                "hard_state_sha256": digest(hard),
                "runtime_state": str(runtime),
                "runtime_state_sha256": digest(runtime),
                "portable_onnx": str(portable),
                "portable_onnx_bytes": portable.stat().st_size,
                "portable_onnx_sha256": digest(portable),
                "portable_ledger": str(ledger),
                "portable_ledger_sha256": "",
                "runtime_package": str(package),
                "runtime_package_bytes": package.stat().st_size,
                "runtime_package_sha256": digest(package),
                "runtime_package_compression": "gzip",
            }
            ledger.write_text(json.dumps({
                "format": "clark-portable-onnx-ternary-v2",
                "materialization_manifest_sha256": digest(manifest),
                "hard_state_sha256": digest(hard),
                "runtime_state_file_sha256": digest(runtime),
                "model_sha256": digest(portable),
                "runtime_package_sha256": digest(package),
                "runtime_package_compression": "gzip",
                "scope": "all_matrix_and_conv",
            }))
            candidate["portable_ledger_sha256"] = digest(ledger)
            validate_artifacts(candidate)
            bad = json.loads(ledger.read_text())
            bad["materialization_manifest_sha256"] = "different"
            ledger.write_text(json.dumps(bad))
            candidate["portable_ledger_sha256"] = digest(ledger)
            with self.assertRaisesRegex(ValueError, "canonical materialization"):
                validate_artifacts(candidate)

    def test_precommit_cannot_change_after_it_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "precommit.json"
            stable = {"format": "test", "selected_lane": "fp32"}
            first = write_or_verify_precommit(path, stable)
            self.assertEqual(first["selected_lane"], "fp32")
            with self.assertRaisesRegex(ValueError, "differs"):
                write_or_verify_precommit(
                    path,
                    {"format": "test", "selected_lane": "recovered_core"},
                )

    def test_completed_receipt_is_revalidated_before_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            precommit = root / "final-precommit.json"
            report = root / "candidate-metrics.json"
            source = root / "evaluator.py"
            receipt_path = root / "final-evaluation-receipt.json"
            precommit.write_text('{"selected_lane":"recovered_core"}\n')
            report.write_text('{"samples":[{}],"overall":{"psnr":42.0}}\n')
            source.write_text("# sealed evaluator\n")

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            receipt = {
                "format": "backdropkit-swin2sr-final-evaluation-v1",
                "precommit_sha256": digest(precommit),
                "candidate_metrics": str(report),
                "candidate_metrics_sha256": digest(report),
                "samples": 1,
                "overall": {"psnr": 42.0},
                "source_files": {"evaluator.py": digest(source)},
            }
            receipt_path.write_text(json.dumps(receipt))
            verified = verify_completed_receipt(
                receipt_path,
                precommit,
                report,
                root,
                ("evaluator.py",),
            )
            self.assertEqual(verified["samples"], 1)
            report.write_text('{"samples":[],"overall":{"psnr":0.0}}\n')
            with self.assertRaisesRegex(ValueError, "metrics hash mismatch"):
                verify_completed_receipt(
                    receipt_path,
                    precommit,
                    report,
                    root,
                    ("evaluator.py",),
                )

    def test_development_selection_cannot_reuse_final_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development_bundle = root / "development-bundle.json"
            development_q4 = root / "development-q4.json"
            final_bundle = root / "final-bundle.json"
            final_q4 = root / "final-q4.json"
            study = root / "study-manifest.json"
            development_bundle.write_text(
                '{"manifest_sha256":"development","samples":1}\n'
            )
            development_q4.write_text(
                '{"bundle_manifest_sha256":"development",'
                '"state_sha256":"q4-development"}\n'
            )
            final_bundle.write_text(
                '{"manifest_sha256":"final","samples":1}\n'
            )
            final_q4.write_text(
                '{"bundle_manifest_sha256":"final","state_sha256":"q4-final"}\n'
            )

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            study.write_text(json.dumps({
                "format": "backdropkit-swin2sr-full-surface-comparison-v1",
                "selection_source": "development",
                "bundle": str(development_bundle),
                "bundle_sha256": digest(development_bundle),
                "q4_reference": str(development_q4),
                "q4_reference_sha256": digest(development_q4),
            }))
            summary = {
                "selection_source": "development",
                "study_manifest": str(study),
                "study_manifest_sha256": digest(study),
                "bundle": str(development_bundle),
                "bundle_sha256": digest(development_bundle),
                "bundle_manifest_sha256": "development",
                "q4_reference": str(development_q4),
                "q4_reference_sha256": digest(development_q4),
                "q4_reference_state_sha256": "q4-development",
            }
            validate_development_selection(
                summary,
                final_bundle,
                json.loads(final_bundle.read_text()),
                final_q4,
            )
            final_bundle.write_text(
                '{"manifest_sha256":"development","samples":1}\n'
            )
            with self.assertRaisesRegex(ValueError, "final evaluation bundle"):
                validate_development_selection(
                    summary,
                    final_bundle,
                    json.loads(final_bundle.read_text()),
                    final_q4,
                )


if __name__ == "__main__":
    unittest.main()
