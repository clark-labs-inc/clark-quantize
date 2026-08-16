from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import torch

from clark_quantize.qwen_qat import (
    HardTernaryLinear,
    checkpoint_rng_state_valid,
    hard_ternary_ste,
    hard_ternary_weight,
    load_qat_checkpoint,
    save_qat_checkpoint,
    state_tree_equal,
    topk_tail_teacher_kl,
)
from clark_quantize.ternary import ternary_codes_scales


class QwenQATPrimitiveTests(unittest.TestCase):
    def test_hard_weight_matches_fp16_artifact_reconstruction(self) -> None:
        source = torch.randn(7, 131, generator=torch.Generator().manual_seed(11))
        actual = hard_ternary_weight(source, group_size=128, row_batch=3)
        codes, scales = ternary_codes_scales(source, group=128, method="optimal")
        padded = torch.nn.functional.pad(codes, (0, 125))
        expected = (
            padded.reshape(7, 2, 128).to(torch.float16)
            * scales.to(torch.float16).unsqueeze(-1)
        ).reshape(7, 256)[:, :131].float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_ste_passes_gradient_and_linear_runs(self) -> None:
        master = torch.nn.Parameter(torch.randn(4, 8))
        hard_ternary_ste(master, group_size=4, row_batch=2).sum().backward()
        torch.testing.assert_close(master.grad, torch.ones_like(master))

        layer = HardTernaryLinear(master.detach(), group_size=4, row_batch=2)
        inputs = torch.randn(2, 3, 8)
        output = layer(inputs)
        self.assertEqual(tuple(output.shape), (2, 3, 4))
        output.square().mean().backward()
        self.assertIsNotNone(layer.master.grad)
        self.assertTrue(torch.isfinite(layer.master.grad).all())

    def test_topk_tail_kl_is_zero_for_matching_distributions(self) -> None:
        logits = torch.randn(1, 5, 13, generator=torch.Generator().manual_seed(9))
        lse = torch.logsumexp(logits[0].float(), dim=-1)
        values, indices = torch.topk(logits[0], k=5, dim=-1)
        loss, stats = topk_tail_teacher_kl(logits, values, indices, lse)
        self.assertAlmostEqual(float(loss), 0.0, places=5)
        self.assertEqual(stats["positions"], 4)

    def test_topk_tail_kl_rejects_misaligned_cache(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not match"):
            topk_tail_teacher_kl(
                torch.zeros(1, 3, 7),
                torch.zeros(2, 4),
                torch.zeros(2, 4, dtype=torch.long),
                torch.zeros(2),
            )

    def test_atomic_checkpoint_roundtrip_and_corruption_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            masters = {"model.layers.3.mlp.up_proj.weight": torch.arange(12).reshape(3, 4).float()}
            tensor_metadata = {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in masters.items()
            }
            path = save_qat_checkpoint(
                root,
                step=10,
                state={"masters": masters, "optimizer": {"step": 10}},
                tensor_metadata=tensor_metadata,
                metadata={"model_revision": "test", "selected_layer": 3},
            )
            self.assertEqual(path.name, "checkpoint-10")
            state, manifest = load_qat_checkpoint(path)
            torch.testing.assert_close(state["masters"][next(iter(masters))], next(iter(masters.values())))
            self.assertTrue(manifest["complete"])
            with self.assertRaises(FileExistsError):
                save_qat_checkpoint(
                    root,
                    step=10,
                    state={"masters": masters},
                    tensor_metadata=tensor_metadata,
                    metadata={},
                )
            state_path = path / "state.pt"
            damaged = bytearray(state_path.read_bytes())
            damaged[-1] ^= 1
            state_path.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_qat_checkpoint(path)

    def test_checkpoint_refuses_existing_incomplete_save(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / ".checkpoint-10.interrupted").mkdir()
            with self.assertRaisesRegex(RuntimeError, "requires inspection"):
                save_qat_checkpoint(
                    root,
                    step=10,
                    state={"masters": {"weight": torch.zeros(1, 1)}},
                    tensor_metadata={"weight": {"shape": [1, 1], "dtype": "torch.float32"}},
                    metadata={},
                )

    def test_nested_optimizer_state_comparison_is_exact_across_devices(self) -> None:
        left = {
            "state": {0: {"step": 3, "m_q": torch.tensor([1, -2], dtype=torch.int8)}},
            "param_groups": [{"params": [0], "betas": (0.9, 0.95)}],
            "bonsai_global_step": 3,
        }
        right = {
            "state": {0: {"step": 3, "m_q": torch.tensor([1, -2], dtype=torch.int8)}},
            "param_groups": [{"params": [0], "betas": (0.9, 0.95)}],
            "bonsai_global_step": 3,
        }
        self.assertTrue(state_tree_equal(left, right))
        right["state"][0]["m_q"][1] = -1
        self.assertFalse(state_tree_equal(left, right))

    def test_checkpoint_rng_state_requires_complete_device_list(self) -> None:
        state = {
            "cpu_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": [torch.ones(8, dtype=torch.uint8)],
        }
        self.assertTrue(checkpoint_rng_state_valid(state, cuda_devices=1))
        self.assertFalse(checkpoint_rng_state_valid(state, cuda_devices=2))
        state["cpu_rng_state"] = torch.ones(8, dtype=torch.int32)
        self.assertFalse(checkpoint_rng_state_valid(state, cuda_devices=1))


if __name__ == "__main__":
    unittest.main()
