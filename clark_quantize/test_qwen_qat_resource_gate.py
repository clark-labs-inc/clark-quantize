from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/qwen35_qat_resource_gate.py"
)
SPEC = importlib.util.spec_from_file_location("qwen35_qat_resource_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QwenQATResourceGateTests(unittest.TestCase):
    def test_inventory_accounting_is_explicit(self) -> None:
        tensors = [
            {
                "name": "model.language_model.layers.0.mlp.up_proj.weight",
                "shape": [10, 128], "dtype": "BF16", "numel": 1280, "bytes": 2560,
            },
            {
                "name": "model.language_model.layers.0.input_layernorm.weight",
                "shape": [64], "dtype": "BF16", "numel": 64, "bytes": 128,
            },
            {
                "name": "model.language_model.embed_tokens.weight",
                "shape": [100, 64], "dtype": "BF16", "numel": 6400, "bytes": 12800,
            },
        ]
        rules = {
            "ternary_master_bytes_per_parameter": 4,
            "hard_forward_cache_bytes_per_parameter": 2,
            "gradient_bytes_per_parameter": 4,
            "muon_momentum_bytes_per_parameter": 2,
            "teacher_bytes_per_parameter": 2,
        }
        result = MODULE.estimate(tensors, rules)
        self.assertEqual(result["total_parameters"], 7744)
        self.assertEqual(result["ternary_master_parameters"], 1280)
        self.assertEqual(result["retained_source_bytes"], 12928)
        self.assertEqual(result["teacher_cache_persistent_lower_bound_bytes"], 28800)
        self.assertEqual(result["full_teacher_persistent_lower_bound_bytes"], 44288)
        self.assertEqual(result["one_checkpoint_estimated_tensor_bytes"], 20992)

    def test_excluded_islands_are_not_counted_as_masters(self) -> None:
        self.assertTrue(MODULE.is_ternary_master(
            "model.language_model.layers.0.self_attn.q_proj.weight", [128, 128]
        ))
        self.assertFalse(MODULE.is_ternary_master(
            "model.language_model.layers.0.linear_attn.conv1d.weight", [128, 128]
        ))
        self.assertFalse(MODULE.is_ternary_master(
            "model.visual.layers.0.mlp.up_proj.weight", [128, 128]
        ))

    def test_linux_memory_gate_uses_memavailable(self) -> None:
        fake = "MemTotal:       32000000 kB\nMemFree:         7000000 kB\nMemAvailable:   30000000 kB\n"
        with mock.patch.object(MODULE.Path, "is_file", return_value=True), mock.patch.object(
            MODULE.Path, "read_text", return_value=fake
        ):
            self.assertEqual(MODULE.available_memory_bytes(), 30_000_000 * 1024)


if __name__ == "__main__":
    unittest.main()
