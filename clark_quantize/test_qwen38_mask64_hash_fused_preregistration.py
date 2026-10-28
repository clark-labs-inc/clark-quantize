from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38Mask64HashFusedPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_mask64_hash_fused_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_is_only_qwen38_27b(self) -> None:
        scope = self.prereg["scope"]
        self.assertEqual(scope["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(scope["qwen_4b_compute_forbidden"])

    def test_cases_are_exact_ratio128_qwen_mlp_shapes(self) -> None:
        cases = {
            (case["rows"], case["columns"]): case["sketch_dim"]
            for case in self.prereg["cases"]
        }
        self.assertEqual(cases, {(17408, 5120): 40, (5120, 17408): 136})
        for (_, columns), sketch_dim in cases.items():
            self.assertEqual(columns // sketch_dim, 128)

    def test_gate_requires_low_fusion_overhead_and_exact_formula(self) -> None:
        gate = self.prereg["gate"]
        self.assertGreaterEqual(
            gate["minimum_fused_throughput_fraction_of_mask64_only"], 0.95
        )
        self.assertLessEqual(gate["maximum_absolute_error"], 0.0005)
        self.assertLessEqual(
            gate["maximum_complete_component_bits_per_weight"], 1.57
        )
        self.assertIn("T x plus A times R x", self.prereg["promotion_rules"][2])

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
