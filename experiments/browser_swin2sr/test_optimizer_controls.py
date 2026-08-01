from __future__ import annotations

import copy
import unittest

import torch

from optimizer_controls import (
    OPTIMIZERS,
    apply_proximal_pull,
    build_optimizer,
    configure_resumed_optimizer,
    scheduled_prox_rho,
)


class _SignProjector(torch.nn.Module):
    def project(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.sign(weight)


class OptimizerControlTests(unittest.TestCase):
    def test_adamw_control_has_no_proximal_schedule(self) -> None:
        self.assertEqual(scheduled_prox_rho("adamw", 1.0), 0.0)

    def test_bonsai_controls_do_not_inherit_proximal_schedule(self) -> None:
        self.assertEqual(scheduled_prox_rho("bonsai_qat8", 1.0), 0.0)
        self.assertEqual(scheduled_prox_rho("bonsai_qat8_muon", 1.0), 0.0)

    def test_proximal_schedule_starts_only_after_hard_endpoint(self) -> None:
        self.assertEqual(scheduled_prox_rho("adamw_prox", 0.27), 0.0)
        self.assertAlmostEqual(scheduled_prox_rho("adamw_prox", 0.28), 0.005)
        self.assertAlmostEqual(scheduled_prox_rho("adamw_prox", 0.60), 0.02)
        self.assertAlmostEqual(scheduled_prox_rho("adamw_prox", 0.90), 0.05)

    def test_proximal_pull_moves_master_toward_hard_projection(self) -> None:
        master = torch.nn.Parameter(torch.tensor([[0.2, -0.4]]))
        apply_proximal_pull([(master, _SignProjector())], 0.25)
        torch.testing.assert_close(master, torch.tensor([[0.4, -0.55]]))

    def test_bonsai_optimizer_separates_matrix_and_other_state(self) -> None:
        master = torch.nn.Parameter(torch.ones(3, 4))
        scale = torch.nn.Parameter(torch.ones(3))
        optimizer = build_optimizer(
            "bonsai_qat8",
            [master],
            [scale],
            learning_rate=1e-3,
            weight_decay=0.0,
        )
        master.grad = torch.ones_like(master)
        scale.grad = torch.ones_like(scale)
        optimizer.step()

        self.assertEqual(optimizer.param_groups[0]["role"], "matrix_qat8")
        self.assertEqual(optimizer.param_groups[1]["role"], "adamw")
        self.assertEqual(optimizer.state[master]["m_q"].dtype, torch.int8)
        self.assertIn("v", optimizer.state[scale])

    def test_every_optimizer_takes_finite_cpu_steps(self) -> None:
        for name in OPTIMIZERS:
            with self.subTest(optimizer=name):
                torch.manual_seed(7)
                master = torch.nn.Parameter(torch.randn(8, 8))
                scale = torch.nn.Parameter(torch.randn(8))
                optimizer = build_optimizer(
                    name,
                    [master, master],
                    [master, scale, scale],
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    group_size=4,
                    muon_every=1,
                    muon_steps=2,
                )
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    (master.square().mean() + scale.square().mean()).backward()
                    optimizer.step()
                self.assertTrue(torch.isfinite(master).all())
                self.assertTrue(torch.isfinite(scale).all())
                parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ]
                self.assertEqual(len(parameters), len({id(item) for item in parameters}))

    def test_low_state_variants_create_expected_state(self) -> None:
        for name, expected_key in (
            ("mirror_sgd_adagrad", "group_accum"),
            ("factored_adafactor", "row_accum"),
            ("bonsai_muon", "momentum_buffer"),
            ("bonsai_muon_fast", "momentum_buffer"),
        ):
            with self.subTest(optimizer=name):
                master = torch.nn.Parameter(torch.randn(8, 8))
                optimizer = build_optimizer(
                    name,
                    [master],
                    [],
                    learning_rate=1e-4,
                    weight_decay=0.0,
                    group_size=4,
                )
                master.grad = torch.ones_like(master)
                optimizer.step()
                self.assertIn(expected_key, optimizer.state[master])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_every_optimizer_takes_a_finite_mps_step(self) -> None:
        for name in OPTIMIZERS:
            with self.subTest(optimizer=name):
                master = torch.nn.Parameter(torch.randn(8, 8, device="mps"))
                scale = torch.nn.Parameter(torch.randn(8, device="mps"))
                convolution = torch.nn.Parameter(
                    torch.randn(4, 4, 3, 3, device="mps")
                )
                optimizer = build_optimizer(
                    name,
                    [master],
                    [scale, convolution],
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    group_size=4,
                    muon_every=1,
                    muon_steps=2,
                )
                (
                    master.square().mean()
                    + scale.square().mean()
                    + convolution.square().mean()
                ).backward()
                optimizer.step()
                torch.mps.synchronize()
                self.assertTrue(torch.isfinite(master).all().item())
                self.assertTrue(torch.isfinite(scale).all().item())
                self.assertTrue(torch.isfinite(convolution).all().item())

    def test_every_optimizer_resumes_with_exact_next_cpu_step(self) -> None:
        for name in OPTIMIZERS:
            with self.subTest(optimizer=name):
                torch.manual_seed(11)
                master = torch.nn.Parameter(torch.randn(8, 8))
                scale = torch.nn.Parameter(torch.randn(8))
                optimizer = build_optimizer(
                    name,
                    [master],
                    [scale],
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    group_size=4,
                    muon_every=1,
                    muon_steps=2,
                )
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    (master.square().mean() + scale.square().mean()).backward()
                    optimizer.step()
                resumed_master = torch.nn.Parameter(master.detach().clone())
                resumed_scale = torch.nn.Parameter(scale.detach().clone())
                resumed = build_optimizer(
                    name,
                    [resumed_master],
                    [resumed_scale],
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    group_size=4,
                    muon_every=1,
                    muon_steps=2,
                )
                resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                optimizer.zero_grad(set_to_none=True)
                resumed.zero_grad(set_to_none=True)
                (master.square().mean() + scale.square().mean()).backward()
                (
                    resumed_master.square().mean()
                    + resumed_scale.square().mean()
                ).backward()
                optimizer.step()
                resumed.step()
                torch.testing.assert_close(master, resumed_master)
                torch.testing.assert_close(scale, resumed_scale)

    def test_resume_rejects_optimizer_switch(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = torch.optim.AdamW([parameter])
        with self.assertRaisesRegex(ValueError, "resume optimizer mismatch"):
            configure_resumed_optimizer(
                optimizer,
                {"name": "adamw"},
                {
                    "name": "bonsai_qat8",
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "muon_every": 16,
                    "muon_steps": 2,
                },
            )


if __name__ == "__main__":
    unittest.main()
