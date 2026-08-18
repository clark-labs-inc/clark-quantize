from __future__ import annotations

import numpy as np
import torch

from clark_quantize.qwen38_affine_butterfly import (
    AffineTernary,
    deterministic_givens_pairings,
)
from clark_quantize.qwen38_affine_codebook import compress_affine_parameters
from clark_quantize.qwen38_affine_qat import (
    StoredAffineConvWeight,
    StoredAffineLinear,
    TrainableAffinePayload,
    TrainableStoredRotation,
    hard_fp16_ste,
    qat_parameter_groups,
)


def payload() -> TrainableAffinePayload:
    source = AffineTernary(
        codes=torch.tensor(
            [[-1, 0, 1, 0, -1, 1], [1, 0, -1, 1, 0, -1]], dtype=torch.int8
        ),
        alpha=torch.tensor([[0.25, 0.5], [0.75, 1.0]]),
        mu=torch.tensor([[-0.2, 0.1], [0.0, 0.3]]),
        group_size=3,
    )
    return TrainableAffinePayload(compress_affine_parameters(source, chunk_rows=1))


def test_hard_forward_initially_matches_the_packed_payload() -> None:
    trained = payload()
    exported = trained.export_compressed()
    torch.testing.assert_close(
        trained.decoded_rotated_weight(), exported.payload.reconstruct(), rtol=0, atol=0
    )
    assert trained.decoded_rotated_weight().dtype == torch.float32
    assert set(trained.hard_codes().unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert np.any(exported.mu.values == np.float16(0))


def test_stored_qat_state_gets_finite_gradients_and_projects() -> None:
    trained = payload()
    pairings = deterministic_givens_pairings(6, stages=2, seed=4)
    rotation = TrainableStoredRotation(torch.zeros(2, 3), pairings)
    groups = qat_parameter_groups(
        [trained],
        [rotation],
        ternary_learning_rate=0.1,
        index_learning_rate=0.25,
        continuous_learning_rate=0.001,
    )
    optimizer = torch.optim.Adam(groups.groups)
    inputs = torch.randn(5, 6)
    target = torch.randn(5, 2)
    observed = torch.nn.functional.linear(
        inputs, rotation.effective_weight(trained.decoded_rotated_weight())
    )
    loss = (observed - target).square().mean()
    loss.backward()
    assert groups.ternary_parameters == 12
    assert groups.index_parameters == 8
    assert groups.continuous_parameters == 2 * (2 + 16) + 6
    for parameter_group in optimizer.param_groups:
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in parameter_group["params"]
        )
    optimizer.step()
    trained.project_training_state()
    assert bool((trained.codes_shadow >= -1).all())
    assert bool((trained.codes_shadow <= 1).all())
    assert bool((trained.alpha_index_shadow >= 0).all())
    assert bool((trained.alpha_index_shadow <= 15).all())


def test_fp16_ste_has_frozen_forward_and_identity_gradient() -> None:
    value = torch.tensor([0.123456], requires_grad=True)
    frozen = hard_fp16_ste(value)
    assert torch.equal(frozen.detach(), value.detach().to(torch.float16).to(torch.float32))
    frozen.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_runtime_modules_use_rotation_then_stored_weight_and_keep_bf16_boundary() -> None:
    trained = payload()
    pairings = deterministic_givens_pairings(6, stages=2, seed=9)
    rotation = TrainableStoredRotation(torch.randn(2, 3) * 0.01, pairings)
    linear = StoredAffineLinear(trained, rotation)
    inputs = torch.randn(4, 6).to(torch.bfloat16)
    observed = linear(inputs)
    rotated = torch.nn.functional.linear(
        inputs.float(),
        rotation.effective_weight(trained.decoded_rotated_weight()),
    ).to(torch.bfloat16)
    torch.testing.assert_close(observed, rotated, rtol=1e-2, atol=2e-2)
    assert observed.dtype == torch.bfloat16

    conv_payload = TrainableAffinePayload(
        compress_affine_parameters(
            AffineTernary(
                codes=torch.randint(-1, 2, (3, 4), dtype=torch.int8),
                alpha=torch.rand(3, 1),
                mu=torch.randn(3, 1),
                group_size=128,
            )
        )
    )
    proxy = StoredAffineConvWeight(conv_payload, (3, 1, 4))
    assert proxy.weight.shape == (3, 1, 4)
    assert proxy.weight.dtype == torch.bfloat16
    assert proxy.bias is None
