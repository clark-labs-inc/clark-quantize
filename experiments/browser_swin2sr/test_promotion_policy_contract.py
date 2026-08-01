import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PromotionPolicyContractTests(unittest.TestCase):
    def test_v5_preserves_its_historical_runtime_harness(self):
        policy = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v5.json").read_text()
        )

        self.assertEqual(
            policy["runtime_harness_sha256"],
            {
                "full_onnx_smoke_browser.html": (
                    "d2939e8ae632701bcfbaa47e1cc3608840886e5d40cdcdac0b47d5f951947a5f"
                ),
                "runtime_surface.mjs": (
                    "34042f57cfd9e88d72dcf05ea768c2ae05fbb8799236dfaccaa716d12b227d40"
                ),
            },
        )
        self.assertEqual(policy["runtime_input_shape"], [1, 3, 128, 128])

    def test_v5_changes_only_the_declared_harness_identity(self):
        v4 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v4.json").read_text()
        )
        v5 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v5.json").read_text()
        )
        amendment = v5.pop("amendment")
        v5["runtime_harness_sha256"] = v4["runtime_harness_sha256"]

        self.assertEqual(v5, v4)
        self.assertEqual(
            amendment["superseded_policy_sha256"],
            sha256(ROOT / "promotion_policy_glyphsafe_v4.json"),
        )
        self.assertTrue(amendment["quality_contract_unchanged"])

    def test_v6_changes_only_the_declared_material_size_threshold(self):
        v5 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v5.json").read_text()
        )
        v6 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v6.json").read_text()
        )
        amendment = v6.pop("amendment")
        v5.pop("amendment")
        v6["artifact_gate"] = v5["artifact_gate"]

        self.assertEqual(v6, v5)
        self.assertEqual(
            amendment["superseded_policy_sha256"],
            sha256(ROOT / "promotion_policy_glyphsafe_v5.json"),
        )
        self.assertTrue(amendment["quality_contract_unchanged"])

    def test_v6_requires_at_least_a_two_x_artifact_reduction(self):
        policy = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v6.json").read_text()
        )

        self.assertEqual(
            policy["artifact_gate"]["maximum_fraction_of_reference"],
            0.5,
        )

    def test_v7_changes_only_the_compressed_runtime_identity(self):
        v6 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v6.json").read_text()
        )
        v7 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v7.json").read_text()
        )
        amendment = v7.pop("amendment")
        v6.pop("amendment")
        v7.pop("runtime_artifact_compression")
        v7["runtime_harness_sha256"] = v6["runtime_harness_sha256"]

        self.assertEqual(v7, v6)
        self.assertEqual(
            amendment["superseded_policy_sha256"],
            sha256(ROOT / "promotion_policy_glyphsafe_v6.json"),
        )
        self.assertTrue(amendment["quality_contract_unchanged"])

    def test_v7_pins_gzip_and_the_current_runtime_harness(self):
        policy = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v7.json").read_text()
        )

        self.assertEqual(policy["runtime_artifact_compression"], "gzip")
        self.assertEqual(
            policy["runtime_harness_sha256"],
            {
                "full_onnx_smoke_browser.html": (
                    "598496a941325d0925589d73dfae03932f8afd43469dd0d282af0e2400800923"
                ),
                "runtime_surface.mjs": sha256(ROOT / "runtime_surface.mjs"),
            },
        )

    def test_v8_changes_only_fixture_identity_and_harness(self):
        v7 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v7.json").read_text()
        )
        v8 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v8.json").read_text()
        )
        amendment = v8.pop("amendment")
        fixture_contract = v8.pop("runtime_fixture_contract")
        v7.pop("amendment")
        v8["runtime_harness_sha256"] = v7["runtime_harness_sha256"]

        self.assertEqual(v8, v7)
        self.assertEqual(
            amendment["superseded_policy_sha256"],
            sha256(ROOT / "promotion_policy_glyphsafe_v7.json"),
        )
        self.assertTrue(amendment["quality_contract_unchanged"])
        self.assertEqual(
            fixture_contract,
            {
                "fixture_sha256_required": True,
                "canonical_hard_state_sha256_required": True,
            },
        )

    def test_v8_pins_the_fixture_aware_runtime_harness(self):
        policy = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v8.json").read_text()
        )

        self.assertEqual(
            policy["runtime_harness_sha256"],
            {
                "full_onnx_smoke_browser.html": sha256(
                    ROOT / "full_onnx_smoke_browser.html"
                ),
                "runtime_surface.mjs": sha256(ROOT / "runtime_surface.mjs"),
            },
        )

    def test_v9_changes_only_the_packed_runtime_harness_identity(self):
        v8 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v8.json").read_text()
        )
        v9 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v9.json").read_text()
        )
        amendment = v9.pop("amendment")
        v8.pop("amendment")
        v9["runtime_harness_sha256"] = v8["runtime_harness_sha256"]

        self.assertEqual(v9, v8)
        self.assertEqual(amendment["supersedes"], "promotion_policy_glyphsafe_v8.json")
        self.assertEqual(
            amendment["superseded_policy_sha256"],
            sha256(ROOT / "promotion_policy_glyphsafe_v8.json"),
        )
        self.assertTrue(amendment["quality_contract_unchanged"])

    def test_v9_preserves_its_historical_packed_runtime_harness(self):
        policy = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v9.json").read_text()
        )

        self.assertEqual(
            policy["runtime_harness_sha256"],
            {
                "packed_v3_smoke_browser.html": "1a44e25d1ae9801e5b24761d5060c9fae6287ca215beb1b3c0ba8f59626c02ba",
                "runtime_surface.mjs": "b7ad36ac79e7c4ba4ebda964c879e4a061144c4c8eecfccad7dfb877c4b9516c",
                "runtime/packedBundle.js": "152abf74eac52f76b8f4ae62d5d2f454413ade3d2c121fe05aa15a3504a0a138",
                "runtime/packedRuntime.js": "1b807b3e4b9708c2ab64d87214f6f2ee67a13f9d3ca581acf6db4b21efc4fd4d",
                "runtime/pkg/clark_air_browser_runtime.js": "060ae64714b73e024d07ec7cff48fa5b689e89734d4511404e051050bc87cd16",
                "runtime/pkg/clark_air_browser_runtime_bg.wasm": "ec17c23c00f8f85a138c07b4e4609435b55af4e40f768a4fcc02d5d010e08748",
                "runtime/pkg-simd/clark_air_browser_runtime.js": "060ae64714b73e024d07ec7cff48fa5b689e89734d4511404e051050bc87cd16",
                "runtime/pkg-simd/clark_air_browser_runtime_bg.wasm": "f3d1e7488dcce5cb6896d853cb9225282870e9ed87c1d2c18615e8a1f5a28139",
            },
        )

    def test_v10_changes_only_current_wasm_runtime_identity(self):
        v9 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v9.json").read_text()
        )
        v10 = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v10.json").read_text()
        )
        amendment = v10.pop("amendment")
        v9.pop("amendment")
        v10["runtime_harness_sha256"] = v9["runtime_harness_sha256"]

        self.assertEqual(v10, v9)
        self.assertEqual(amendment["supersedes"], "promotion_policy_glyphsafe_v9.json")
        self.assertEqual(
            amendment["superseded_policy_sha256"],
            sha256(ROOT / "promotion_policy_glyphsafe_v9.json"),
        )
        self.assertTrue(amendment["quality_contract_unchanged"])

    def test_v10_pins_the_current_packed_runtime_harness(self):
        policy = json.loads(
            (ROOT / "promotion_policy_glyphsafe_v10.json").read_text()
        )
        expected = {
            "packed_v3_smoke_browser.html": ROOT / "packed_v3_smoke_browser.html",
            "runtime_surface.mjs": ROOT / "runtime_surface.mjs",
            "runtime/packedBundle.js": ROOT / "runtime/packedBundle.js",
            "runtime/packedRuntime.js": ROOT / "runtime/packedRuntime.js",
            "runtime/pkg/clark_air_browser_runtime.js": ROOT / "runtime/pkg/clark_air_browser_runtime.js",
            "runtime/pkg/clark_air_browser_runtime_bg.wasm": ROOT / "runtime/pkg/clark_air_browser_runtime_bg.wasm",
            "runtime/pkg-simd/clark_air_browser_runtime.js": ROOT / "runtime/pkg-simd/clark_air_browser_runtime.js",
            "runtime/pkg-simd/clark_air_browser_runtime_bg.wasm": ROOT / "runtime/pkg-simd/clark_air_browser_runtime_bg.wasm",
        }

        self.assertEqual(
            policy["runtime_harness_sha256"],
            {relative: sha256(path) for relative, path in expected.items()},
        )


if __name__ == "__main__":
    unittest.main()
