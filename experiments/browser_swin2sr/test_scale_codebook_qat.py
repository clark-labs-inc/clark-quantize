from types import SimpleNamespace

import torch
from torch.nn.utils import parametrize

from quantizer_controls import ProgressiveLowBit
from scale_codebook_qat import (
    ScaleCodebookProjection,
    add_per_surface_scale_codebooks,
    add_scale_codebooks,
    scale_codebook_parameter_group,
    scale_codebook_summary,
    set_scale_codebook_progress,
)


def test_scale_projection_blends_gradually_and_keeps_weight_ste():
    projected = torch.tensor([0.25, -0.25, 0.0, 1.0, -1.0, 0.0], requires_grad=True)
    projection = ScaleCodebookProjection(projected.detach(), 3, 1)
    projection.amount.fill_(0.0)
    assert torch.equal(projection(projected), projected)
    projection.amount.fill_(1.0)
    output = projection(projected)
    assert output.unique().numel() <= 5
    output.sum().backward()
    assert torch.equal(projected.grad, torch.ones_like(projected))
    assert projection.raw_centers.grad is not None
    assert torch.all(projection.centers > 0)
    assert torch.equal(
        projection.stored_centers,
        projection.stored_centers.to(torch.float16).to(torch.float32),
    )


def _surface(amount: float = 1.0):
    module = torch.nn.Linear(4, 2, bias=False)
    quantizer = ProgressiveLowBit(
        "ternary", "absmean", 2, 0.0, 1.0, module.weight
    )
    parametrize.register_parametrization(module, "weight", quantizer)
    quantizer.amount.fill_(amount)
    return SimpleNamespace(
        name="layer",
        category="core_linear",
        quantizer=quantizer,
        module=module,
    )


def test_attach_requires_hard_ternary_and_exposes_centers():
    try:
        add_scale_codebooks([_surface(0.5)], target_bits_by_category={"core_linear": 1})
    except ValueError as error:
        assert "not hard" in str(error)
    else:
        raise AssertionError("transitioning ternary surface must be rejected")
    surfaces = add_scale_codebooks(
        [_surface()], target_bits_by_category={"core_linear": 1}
    )
    surfaces[0].module.requires_grad_(False)
    parameters = scale_codebook_parameter_group(surfaces)
    assert len(parameters) == 1
    assert parameters[0].requires_grad
    set_scale_codebook_progress(surfaces, 0.2)
    assert float(surfaces[0].projection.amount) == 0.5
    summary = scale_codebook_summary(surfaces)
    assert summary["transitioning_surfaces"] == 1
    assert summary["target_bits_by_category"] == {"core_linear": [1]}
    assert summary["target_bit_counts"] == {"1": 1}


def test_per_surface_map_requires_exact_names_and_assigns_independently():
    first = _surface()
    second = _surface()
    second.name = "other"
    surfaces = add_per_surface_scale_codebooks(
        [first, second],
        target_bits_by_surface={"layer": 1, "other": 4},
    )
    assert [surface.target_bits for surface in surfaces] == [1, 4]
    try:
        add_per_surface_scale_codebooks(
            [_surface()], target_bits_by_surface={"wrong": 1}
        )
    except ValueError as error:
        assert "per-surface scale map differs" in str(error)
    else:
        raise AssertionError("incomplete per-surface map must be rejected")


def test_center_parameters_learn_toward_task_scale():
    target = torch.tensor([0.5, -0.5, 0.0, 1.0, -1.0, 0.0])
    projection = ScaleCodebookProjection(target, 3, 1)
    projection.set_centers(torch.tensor([0.1, 0.2]))
    projection.amount.fill_(1.0)
    optimizer = torch.optim.SGD([projection.raw_centers], lr=0.05)
    initial = torch.nn.functional.mse_loss(projection(target), target).item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(projection(target), target)
        loss.backward()
        optimizer.step()
    assert torch.nn.functional.mse_loss(projection(target), target).item() < initial


def test_chained_collapse_starts_exactly_from_a_frozen_bridge():
    surface = _surface()
    bridge = add_scale_codebooks(
        [surface], target_bits_by_category={"core_linear": 2}
    )[0]
    bridge.projection.amount.fill_(1.0)
    bridge_output = surface.module.weight.detach().clone()
    bridge.projection.raw_centers.requires_grad_(False)
    collapse = add_scale_codebooks(
        [surface], target_bits_by_category={"core_linear": 1}
    )[0]
    collapse.projection.amount.fill_(0.0)
    assert torch.equal(surface.module.weight, bridge_output)
    collapse.projection.amount.fill_(1.0)
    surface.module.parametrizations.weight.original.requires_grad_(True)
    collapse.projection.raw_centers.requires_grad_(True)
    surface.module.weight.square().sum().backward()
    assert bridge.projection.raw_centers.grad is None
    assert collapse.projection.raw_centers.grad is not None
    assert surface.module.parametrizations.weight.original.grad is not None
    assert len(surface.module.parametrizations.weight) == 3
