from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class Qwen38ActivationHashPreregistrationTests(unittest.TestCase):
    def test_frozen_sources_and_qwen_scope_are_exact(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        path = repo / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
            "qwen38_27b_activation_hash_residual_preregistration_v1.json"
        )
        preregistration = json.loads(path.read_text())
        self.assertEqual(
            preregistration["scope"]["active_model_only"], "Qwen/Qwen3.8-27B"
        )
        self.assertTrue(preregistration["scope"]["qwen_4b_compute_forbidden"])
        self.assertEqual(preregistration["training"]["steps"], 12)
        self.assertFalse(preregistration["training"]["ternary_codes_and_scales_trainable"])
        self.assertEqual(
            preregistration["gate"]["maximum_projected_component_bits_per_weight"],
            1.57,
        )
        for descriptor in preregistration["frozen_inputs"].values():
            source = repo / descriptor["path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), descriptor["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
