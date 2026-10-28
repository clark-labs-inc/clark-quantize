from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38Mask64GainHashFusedPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.path = cls.repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_mask64_gain_hash_fused_preregistration_v1.json"
        )
        cls.prereg = json.loads(cls.path.read_text())

    def test_scope_and_exact_qwen_shapes_are_frozen(self) -> None:
        self.assertEqual(
            self.prereg["scope"]["active_model_only"], "Qwen/Qwen3.8-27B"
        )
        self.assertTrue(self.prereg["scope"]["qwen_4b_compute_forbidden"])
        shapes = {
            (case["rows"], case["columns"], case["sketch_dim"])
            for case in self.prereg["cases"]
        }
        self.assertEqual(shapes, {(17408, 5120, 40), (5120, 17408, 136)})

    def test_gate_covers_numerics_throughput_and_storage(self) -> None:
        gate = self.prereg["gate"]
        self.assertLessEqual(gate["maximum_absolute_error"], 0.0005)
        self.assertGreaterEqual(
            gate["minimum_gain_throughput_fraction_of_sign_only"], 0.95
        )
        self.assertLessEqual(
            gate["maximum_complete_component_bits_per_weight"], 1.57
        )

    def test_disk_and_compute_route_layouts_are_both_counted(self) -> None:
        layout = self.prereg["gain_route"]
        self.assertEqual(layout["disk_gain_index_bits"], 3)
        self.assertEqual(layout["compute_gain_index_bits"], 8)
        self.assertTrue(layout["record_compute_expansion_bytes"])

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
