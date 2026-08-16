from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from clark_quantize.qwen_qat import HardTernaryLinear
from clark_quantize.ternary import ternary_codes_scales


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/qwen38_27b_spot/one_layer_qat.py"
SPEC = importlib.util.spec_from_file_location("qwen38_one_layer_qat", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Reader:
    def __init__(self, name: str, baseline_codes: np.ndarray) -> None:
        self.records_by_tensor = {
            name: [{"descriptor": {"row_start": 0, "row_stop": baseline_codes.shape[0]}}]
        }
        self._baseline_codes = baseline_codes

    def raw_batch(self, record):
        del record
        scales = np.ones((self._baseline_codes.shape[0], 2), dtype=np.float16)
        return self._baseline_codes.copy(), scales


class _LogitModel(torch.nn.Module):
    def __init__(self, tokens: int, vocab: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.randn(1, tokens, vocab))

    def forward(self, **kwargs):
        del kwargs
        return type("Outputs", (), {"logits": self.logits})()


class OneLayerQATTests(unittest.TestCase):
    def test_gradient_stop_uses_actual_post_clip_norm(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(8))
        parameter.grad = torch.full_like(parameter, 100.0)
        before, after = MODULE.clip_and_measure_gradient_norm([parameter], 1.0)
        self.assertGreater(float(before), 10.0)
        self.assertLessEqual(float(after), 1.00001)
        self.assertTrue(torch.isfinite(after))

    def test_step10_finalization_requires_complete_bounded_history(self) -> None:
        state = {
            "schema": "clark_qwen38_one_layer_qat_state_v1",
            "step": 10,
            "model_revision": "a" * 40,
            "selected_layer": 3,
            "masters": {"weight": torch.ones(1)},
            "history": [
                {"step": step, "gradient_norm_after_clip": 1.0}
                for step in range(1, 11)
            ],
        }
        history = MODULE.validate_step10_checkpoint_state(
            state,
            revision="a" * 40,
            selected_layer=3,
            wrapper_names={"weight"},
        )
        self.assertEqual(len(history), 10)
        state["history"][4]["gradient_norm_after_clip"] = float("nan")
        with self.assertRaisesRegex(ValueError, "history is invalid"):
            MODULE.validate_step10_checkpoint_state(
                state,
                revision="a" * 40,
                selected_layer=3,
                wrapper_names={"weight"},
            )

    def test_projection_metrics_measure_exact_step_zero_and_churn(self) -> None:
        name = "model.language_model.layers.3.mlp.up_proj.weight"
        source = torch.randn(4, 7, generator=torch.Generator().manual_seed(17))
        baseline, _ = ternary_codes_scales(source, group=4, method="optimal")
        reader = _Reader(name, baseline.cpu().numpy())
        wrapper = HardTernaryLinear(source, group_size=4, row_batch=2)
        initial = MODULE.projection_metrics({name: wrapper}, reader)
        self.assertEqual(initial["parameters"], source.numel())
        self.assertEqual(initial["changed_codes_from_step_zero"], 0)
        self.assertGreaterEqual(initial["ternary_zero_fraction"], 0.0)
        self.assertLessEqual(initial["ternary_zero_fraction"], 1.0)
        self.assertGreaterEqual(initial["master_to_hard_relative_l2"], 0.0)

        with torch.no_grad():
            wrapper.master.mul_(-1)
        changed = MODULE.projection_metrics({name: wrapper}, reader)
        self.assertGreater(changed["changed_codes_from_step_zero"], 0)
        self.assertGreater(changed["code_churn_fraction_from_step_zero"], 0.0)

    def test_calibration_loss_is_finite_and_differentiable(self) -> None:
        tokens, vocab, top_k = 5, 11, 4
        model = _LogitModel(tokens, vocab)
        teacher = model.logits.detach()[0].clone()
        values, indices = torch.topk(teacher, k=top_k, dim=-1)
        row = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "top_values": values.to(torch.bfloat16),
            "top_indices": indices.to(torch.int32),
            "teacher_logsumexp": torch.logsumexp(teacher.float(), dim=-1),
        }
        loss, metrics = MODULE.calibration_loss(model, row, kl_weight=1.0, ce_weight=0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        loss.backward()
        self.assertIsNotNone(model.logits.grad)


if __name__ == "__main__":
    unittest.main()
