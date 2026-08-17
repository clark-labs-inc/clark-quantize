from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38Mask64GEMVPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_mask64_gemv_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_is_only_qwen38_27b(self) -> None:
        scope = self.prereg["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(scope["qwen_4b_compute_forbidden"])

    def test_cases_are_full_qwen_mlp_projection_shapes(self) -> None:
        self.assertEqual(
            {
                (case["rows"], case["columns"])
                for case in self.prereg["cases"]
            },
            {(17408, 5120), (5120, 17408)},
        )

    def test_runtime_rate_and_gate_are_explicit(self) -> None:
        representation = self.prereg["representation"]
        gate = self.prereg["gate"]
        self.assertEqual(representation["raw_code_bits_per_weight"], 1.5)
        self.assertEqual(representation["group_bytes"], 24)
        self.assertGreaterEqual(
            gate["minimum_throughput_fraction_of_aligned2"], 0.5
        )
        self.assertLessEqual(
            gate["maximum_complete_component_bits_per_weight"], 1.57
        )
        self.assertIn(
            "not a complete layer", self.prereg["scope"]["claim_boundary"]
        )

    def test_every_frozen_source_hash_matches(self) -> None:
        for descriptor in self.prereg["frozen_inputs"].values():
            source = self.repo / descriptor["path"]
            self.assertNotEqual(descriptor["sha256"], "PLACEHOLDER")
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                descriptor["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
