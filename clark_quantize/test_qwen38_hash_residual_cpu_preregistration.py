from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38HashResidualCPUPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_hash_residual_cpu_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_is_qwen38_27b_only(self) -> None:
        self.assertEqual(self.prereg["scope"]["active_model_only"], "Qwen/Qwen3.8-27B")
        self.assertTrue(self.prereg["scope"]["qwen_4b_compute_forbidden"])

    def test_exact_layer0_mlp_shapes_are_frozen(self) -> None:
        self.assertEqual(
            self.prereg["cases"],
            [
                {"name": "gate", "input_dim": 5120, "output_rows": 17408},
                {"name": "up", "input_dim": 5120, "output_rows": 17408},
                {"name": "down", "input_dim": 17408, "output_rows": 5120},
            ],
        )

    def test_gate_is_bounded_and_not_a_model_claim(self) -> None:
        gate = self.prereg["gate"]
        self.assertEqual(gate["required_qwen_projection_cases"], 3)
        self.assertLessEqual(gate["maximum_projected_component_bits_per_weight"], 1.57)
        self.assertGreater(gate["minimum_avx2_gemv_speedup_vs_scalar"], 1.0)
        self.assertIn("not a complete layer", self.prereg["scope"]["claim_boundary"])

    def test_every_frozen_source_hash_matches(self) -> None:
        for descriptor in self.prereg["frozen_inputs"].values():
            source = self.repo / descriptor["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), descriptor["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
