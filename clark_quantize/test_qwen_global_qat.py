from __future__ import annotations

from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from clark_quantize.qwen_global_qat import (
    capture_replay_state,
    join_model_state,
    load_global_qat_checkpoint,
    restore_process_rng,
    save_global_qat_checkpoint,
    split_model_state,
    tensor_tree_nbytes,
)
from clark_quantize.ternary_kd import BonsaiMuonFast


class QwenGlobalQATCheckpointTests(unittest.TestCase):
    def replay(self) -> dict:
        generator = np.random.default_rng(17)
        generator.random()
        return capture_replay_state(
            arm_rng_states={"math": generator.bit_generator.state},
            bandit_kl={"math": 0.5},
            bandit_rng_state=random.Random(5).getstate(),
            servo_state={"lr_mult": 1.25, "integral": 0.1},
            kl_history=[(0.1, 0.9)],
        )

    def test_model_split_is_complete_without_overlap(self) -> None:
        state = {
            "layer.weight_master": torch.arange(6).reshape(2, 3).float(),
            "norm.weight": torch.ones(3),
        }
        masters, retained = split_model_state(state)
        self.assertEqual(list(masters), ["layer.weight_master"])
        self.assertEqual(list(retained), ["norm.weight"])
        joined = join_model_state(
            {"masters": masters, "retained_model_state": retained}
        )
        self.assertEqual(joined.keys(), state.keys())
        for name in state:
            torch.testing.assert_close(joined[name], state[name])

    def test_tensor_byte_count_deduplicates_shared_storage(self) -> None:
        tensor = torch.zeros(17, dtype=torch.float32)
        self.assertEqual(
            tensor_tree_nbytes({"first": tensor, "again": tensor, "view": tensor[:3]}),
            tensor.untyped_storage().nbytes(),
        )

    def test_atomic_global_roundtrip_and_identity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            model = {
                "layer.weight_master": torch.arange(6).reshape(2, 3).float(),
                "norm.weight": torch.ones(3),
            }
            identity = {"model": "test", "revision": "abc", "seq_len": 8}
            checkpoint = save_global_qat_checkpoint(
                root,
                step=7,
                model_state=model,
                optimizer_state={"state": {}, "param_groups": []},
                replay_state=self.replay(),
                run_identity=identity,
            )
            state, manifest = load_global_qat_checkpoint(
                checkpoint,
                run_identity=identity,
                arms=["math"],
                cuda_devices=0,
            )
            self.assertEqual(state["step"], 7)
            self.assertTrue(manifest["complete"])
            joined = join_model_state(state)
            for name in model:
                torch.testing.assert_close(joined[name], model[name])
            with self.assertRaisesRegex(ValueError, "identity"):
                load_global_qat_checkpoint(
                    checkpoint,
                    run_identity={**identity, "seq_len": 16},
                    arms=["math"],
                    cuda_devices=0,
                )

    def test_disk_reserve_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(OSError, "admission"):
                save_global_qat_checkpoint(
                    root,
                    step=1,
                    model_state={"layer.weight_master": torch.ones(2, 2)},
                    optimizer_state={},
                    replay_state=self.replay(),
                    run_identity={"model": "test"},
                    reserve_bytes=10**18,
                )
            self.assertFalse(any(path.name.startswith("checkpoint-") for path in root.iterdir()))

    def test_muon_resume_preserves_bf16_momentum_contract(self) -> None:
        parameter = torch.nn.Parameter(torch.arange(6).reshape(2, 3).float())
        optimizer = BonsaiMuonFast([{
            "params": [parameter],
            "lr": 0.01,
            "role": "muon",
            "state_dtype": torch.bfloat16,
            "ns_dtype": torch.bfloat16,
        }])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        saved = optimizer.state_dict()
        self.assertEqual(
            next(iter(saved["state"].values()))["momentum_buffer"].dtype,
            torch.bfloat16,
        )

        resumed_parameter = torch.nn.Parameter(torch.zeros_like(parameter))
        resumed = BonsaiMuonFast([{
            "params": [resumed_parameter],
            "lr": 0.01,
            "role": "muon",
            "state_dtype": torch.bfloat16,
            "ns_dtype": torch.bfloat16,
        }])
        resumed.load_state_dict(saved)
        resumed_buffer = next(iter(resumed.state.values()))["momentum_buffer"]
        self.assertEqual(resumed_buffer.dtype, torch.bfloat16)
        torch.testing.assert_close(
            resumed_buffer,
            next(iter(saved["state"].values()))["momentum_buffer"],
        )

    def test_replay_restores_process_and_arm_rng_exactly(self) -> None:
        original_torch = torch.random.get_rng_state()
        original_numpy = np.random.get_state()
        original_python = random.getstate()
        try:
            torch.manual_seed(101)
            np.random.seed(102)
            random.seed(103)
            arm = np.random.default_rng(104)
            replay = capture_replay_state(
                arm_rng_states={"math": arm.bit_generator.state},
                bandit_kl={"math": 0.5},
                bandit_rng_state=random.Random(105).getstate(),
                servo_state={"lr_mult": 1.0},
                kl_history=[],
            )
            expected = (
                torch.rand(4),
                np.random.random(4),
                [random.random() for _ in range(4)],
                arm.random(4),
            )
            restore_process_rng(replay, cuda_devices=0)
            resumed_arm = np.random.default_rng()
            resumed_arm.bit_generator.state = replay["arm_rng_states"]["math"]
            actual = (
                torch.rand(4),
                np.random.random(4),
                [random.random() for _ in range(4)],
                resumed_arm.random(4),
            )
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertEqual(actual[2], expected[2])
            np.testing.assert_array_equal(actual[3], expected[3])
        finally:
            torch.random.set_rng_state(original_torch)
            np.random.set_state(original_numpy)
            random.setstate(original_python)


if __name__ == "__main__":
    unittest.main()
