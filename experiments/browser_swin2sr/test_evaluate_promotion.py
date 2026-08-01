import unittest

from evaluate_promotion import evaluate, runtime_gate


def report(delta=0.0, artifact_bytes=100, state="reference", samples=12):
    rows = []
    for index in range(samples):
        rows.append({
            "sha256": f"{index:064x}",
            "source": "ui",
            "script": "Latn",
            "language": "eng_Latn",
            "psnr_vs_ground_truth": 30 + delta,
            "edge_psnr_vs_ground_truth": 32 + delta,
            "psnr_vs_teacher": 40 + delta,
        })
    return {
        "model": "reference",
        "model_revision": "revision",
        "state_sha256": state,
        "artifact_bytes": artifact_bytes,
        "hard_state_sha256": f"{state}-hard",
        "runtime_artifact_sha256": state,
        "runtime_artifact_bytes": artifact_bytes,
        "runtime_artifact_compression": "gzip",
        "runtime_expanded_artifact_sha256": f"{state}-expanded",
        "runtime_expanded_artifact_bytes": artifact_bytes * 4,
        "bundle_manifest_sha256": "bundle",
        "teacher_model": "teacher",
        "teacher_model_revision": "teacher-revision",
        "crop_size": 128,
        "device": "cpu",
        "samples": rows,
    }


def policy():
    return {
        "seed": 1,
        "bootstrap_samples": 100,
        "reference": {
            "model": "reference",
            "revision": "revision",
            "model_sha256": "reference",
            "artifact_bytes": 100,
            "bundle_manifest_sha256": "bundle",
            "samples": 12,
        },
        "paired_quality_gates": {
            "psnr_vs_ground_truth": {
                "noninferiority_margin": -0.1,
                "p10_delta_minimum": -0.5,
            },
            "edge_psnr_vs_ground_truth": {
                "noninferiority_margin": -0.1,
                "p10_delta_minimum": -0.5,
            },
            "psnr_vs_teacher": {
                "noninferiority_margin": -0.1,
                "p10_delta_minimum": -0.5,
            },
        },
        "stratified_gates": [{
            "key": "language",
            "metric": "psnr_vs_ground_truth",
            "minimum_samples": 12,
            "mean_delta_minimum": -0.1,
        }],
        "artifact_gate": {
            "candidate_bytes_less_than_reference": True,
            "maximum_fraction_of_reference": 0.80,
        },
        "required_runtime_surfaces": ["chromium-wasm"],
        "runtime_harness_sha256": {
            "full_onnx_smoke_browser.html": "html",
            "runtime_surface.mjs": "surface",
        },
        "runtime_input_shape": [1, 3, 128, 128],
        "runtime_artifact_compression": "gzip",
        "runtime_fixture_contract": {
            "fixture_sha256_required": True,
            "canonical_hard_state_sha256_required": True,
        },
        "runtime_limits": {
            "chromium-wasm": {
                "max_session_ms": 5000,
                "max_warm_inference_ms": 2000,
            },
        },
    }


class EvaluatePromotionTests(unittest.TestCase):
    def test_passes_only_with_quality_size_and_runtime_receipts(self):
        runtime = {
            "model_sha256": "candidate",
            "receipts": [{
                "surface": "chromium-wasm",
                "browser": "chromium",
                "backend": "wasm",
                "modelSha256": "candidate",
                "modelBytes": 80,
                "modelCompression": "gzip",
                "expandedModelSha256": "candidate-expanded",
                "expandedModelBytes": 320,
                "fixtureHardStateSha256": "candidate-hard",
                "fixtureSha256": "fixture",
                "harnessFiles": {
                    "full_onnx_smoke_browser.html": "html",
                    "runtime_surface.mjs": "surface",
                },
                "passed": True,
                "inputShape": [1, 3, 128, 128],
                "sessionMs": 1000,
                "warmInferenceMs": 500,
            }],
        }

        class RuntimePath:
            def read_text(self):
                import json
                return json.dumps(runtime)

        candidate = report(
            delta=0.05,
            artifact_bytes=80,
            state="candidate",
        )
        candidate["state_sha256"] = "quality-state"
        result = evaluate(
            policy(),
            report(),
            candidate,
            RuntimePath(),
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["paired_quality"]["psnr_vs_ground_truth"]["improved"])
        self.assertEqual(result["artifact"]["bytes_saved"], 20)
        self.assertEqual(
            result["artifact"]["candidate_fraction_of_reference"],
            0.8,
        )

    def test_rejects_a_numerically_correct_but_slow_runtime(self):
        runtime = {
            "model_sha256": "candidate",
            "receipts": [{
                "surface": "chromium-wasm",
                "browser": "chromium",
                "backend": "wasm",
                "modelSha256": "candidate",
                "modelBytes": 80,
                "modelCompression": "gzip",
                "expandedModelSha256": "candidate-expanded",
                "expandedModelBytes": 320,
                "fixtureHardStateSha256": "candidate-hard",
                "fixtureSha256": "fixture",
                "harnessFiles": {
                    "full_onnx_smoke_browser.html": "html",
                    "runtime_surface.mjs": "surface",
                },
                "passed": True,
                "inputShape": [1, 3, 128, 128],
                "sessionMs": 1000,
                "warmInferenceMs": 2501,
            }],
        }

        class RuntimePath:
            def read_text(self):
                import json
                return json.dumps(runtime)

        result = evaluate(
            policy(),
            report(),
            report(delta=0.05, artifact_bytes=80, state="candidate"),
            RuntimePath(),
        )

        self.assertFalse(result["passed"])
        self.assertIn("chromium-wasm", result["runtime"]["failed_surfaces"])

    def test_rejects_spoofed_browser_label_or_different_artifact(self):
        runtime = {
            "model_sha256": "candidate",
            "receipts": [{
                "surface": "chromium-wasm",
                "browser": "safari",
                "backend": "wasm",
                "modelSha256": "candidate",
                "modelBytes": 79,
                "modelCompression": "gzip",
                "expandedModelSha256": "candidate-expanded",
                "expandedModelBytes": 320,
                "fixtureHardStateSha256": "candidate-hard",
                "fixtureSha256": "fixture",
                "harnessFiles": {
                    "full_onnx_smoke_browser.html": "html",
                    "runtime_surface.mjs": "surface",
                },
                "passed": True,
                "inputShape": [1, 3, 128, 128],
                "sessionMs": 1000,
                "warmInferenceMs": 500,
            }],
        }

        class RuntimePath:
            def read_text(self):
                import json
                return json.dumps(runtime)

        result = evaluate(
            policy(),
            report(),
            report(delta=0.05, artifact_bytes=80, state="candidate"),
            RuntimePath(),
        )

        reasons = result["runtime"]["violations"]["chromium-wasm"]
        self.assertTrue(any("browser" in reason for reason in reasons))
        self.assertTrue(any("model bytes" in reason for reason in reasons))
        self.assertFalse(result["passed"])

    def test_fails_closed_without_runtime_receipts(self):
        result = evaluate(
            policy(),
            report(),
            report(delta=0.05, artifact_bytes=80, state="candidate"),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["runtime"]["missing_surfaces"], ["chromium-wasm"])

    def test_rejects_a_fixture_from_another_canonical_hard_state(self):
        runtime = {
            "model_sha256": "candidate",
            "receipts": [{
                "surface": "chromium-wasm",
                "browser": "chromium",
                "backend": "wasm",
                "modelSha256": "candidate",
                "modelBytes": 80,
                "modelCompression": "gzip",
                "expandedModelSha256": "candidate-expanded",
                "expandedModelBytes": 320,
                "fixtureHardStateSha256": "another-hard-state",
                "fixtureSha256": "fixture",
                "harnessFiles": {
                    "full_onnx_smoke_browser.html": "html",
                    "runtime_surface.mjs": "surface",
                },
                "passed": True,
                "inputShape": [1, 3, 128, 128],
                "sessionMs": 1000,
                "warmInferenceMs": 500,
            }],
        }

        class RuntimePath:
            def read_text(self):
                import json
                return json.dumps(runtime)

        result = evaluate(
            policy(),
            report(),
            report(delta=0.05, artifact_bytes=80, state="candidate"),
            RuntimePath(),
        )

        self.assertFalse(result["passed"])
        self.assertIn(
            "fixture hard-state hash mismatch",
            result["runtime"]["violations"]["chromium-wasm"],
        )

    def test_rejects_required_surfaces_using_different_fixtures(self):
        current_policy = policy()
        current_policy["required_runtime_surfaces"] = [
            "chromium-wasm",
            "safari-wasm",
        ]
        current_policy["runtime_limits"]["safari-wasm"] = {
            "max_session_ms": 5000,
            "max_warm_inference_ms": 2000,
        }
        receipts = []
        for browser, fixture in (
            ("chromium", "fixture-a"),
            ("safari", "fixture-b"),
        ):
            receipts.append({
                "surface": f"{browser}-wasm",
                "browser": browser,
                "backend": "wasm",
                "modelSha256": "candidate",
                "modelBytes": 80,
                "modelCompression": "gzip",
                "expandedModelSha256": "candidate-expanded",
                "expandedModelBytes": 320,
                "fixtureHardStateSha256": "candidate-hard",
                "fixtureSha256": fixture,
                "harnessFiles": current_policy[
                    "runtime_harness_sha256"
                ],
                "passed": True,
                "inputShape": [1, 3, 128, 128],
                "sessionMs": 1000,
                "warmInferenceMs": 500,
            })

        class RuntimePath:
            def read_text(self):
                import json
                return json.dumps({
                    "model_sha256": "candidate",
                    "receipts": receipts,
                })

        result = runtime_gate(
            current_policy,
            RuntimePath(),
            "candidate",
            80,
            "gzip",
            "candidate-expanded",
            320,
            "candidate-hard",
        )

        self.assertFalse(result["passed"])
        self.assertIn("shared-fixture", result["violations"])

    def test_rejects_quality_regression_even_when_smaller(self):
        result = evaluate(
            policy(),
            report(),
            report(delta=-0.2, artifact_bytes=50, state="candidate"),
        )

        self.assertFalse(result["paired_quality"]["psnr_vs_ground_truth"]["passed"])
        self.assertFalse(result["passed"])

    def test_rejects_an_artifact_that_is_smaller_but_not_materially_smaller(self):
        candidate = report(
            delta=0.05,
            artifact_bytes=81,
            state="candidate",
        )
        result = evaluate(policy(), report(), candidate)

        self.assertFalse(result["artifact"]["passed"])
        self.assertEqual(
            result["artifact"]["candidate_fraction_of_reference"],
            0.81,
        )
        self.assertFalse(result["passed"])

    def test_rejects_a_policy_without_a_quantitative_size_gate(self):
        incomplete = policy()
        del incomplete["artifact_gate"]

        with self.assertRaisesRegex(ValueError, "artifact_gate"):
            evaluate(
                incomplete,
                report(),
                report(delta=0.05, artifact_bytes=50, state="candidate"),
            )

    def test_rejects_candidate_scored_with_a_different_teacher_device(self):
        candidate = report(delta=0.05, artifact_bytes=80, state="candidate")
        candidate["device"] = "mps"
        result = evaluate(policy(), report(), candidate)

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("device" in failure for failure in result["candidate_failures"])
        )


if __name__ == "__main__":
    unittest.main()
