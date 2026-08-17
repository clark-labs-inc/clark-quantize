from types import SimpleNamespace

import torch

from clark_quantize.qwen_qat import hard_ternary_weight
from experiments.llm_ternary_158bit_dossier.cpu_runtime.eval_mlp_triplet_reconstruction import (
    normalized_loss_from_weights,
)
from train_mlp_triplet_qat import (
    QuantMLPTriplet,
    activation_position_indices,
    parse_row_indices,
    train_layer,
)
from train_mlp_triplet_survival_qat import QuantSwiGLUMLP


def test_artifact_exact_hard_weights_match_training_forward_quantizer() -> None:
    torch.manual_seed(31)
    gate = torch.randn(6, 4)
    up = torch.randn(6, 4)
    down = torch.randn(4, 6)
    model = QuantMLPTriplet(
        gate,
        up,
        down,
        mode="ternary",
        group_size=2,
        artifact_exact_ternary=True,
    )

    exported = model.hard_weights()
    for name in ("gate", "up", "down"):
        expected_float = hard_ternary_weight(
            getattr(model, name), group_size=2, row_batch=128
        ).to(torch.bfloat16).float()
        torch.testing.assert_close(
            model.quantized(getattr(model, name)), expected_float, rtol=0, atol=0
        )
        expected = expected_float.to(torch.bfloat16)
        torch.testing.assert_close(exported[name], expected, rtol=0, atol=0)


def test_survival_qat_artifact_exact_export_matches_training_quantizer() -> None:
    torch.manual_seed(32)
    source = torch.nn.Module()
    source.gate_proj = torch.nn.Linear(4, 6, bias=False)
    source.up_proj = torch.nn.Linear(4, 6, bias=False)
    source.down_proj = torch.nn.Linear(6, 4, bias=False)
    model = QuantSwiGLUMLP(
        source,
        mode="ternary",
        group_size=2,
        artifact_exact_ternary=True,
    )

    exported = model.hard_weights()
    for name in ("gate", "up", "down"):
        expected_float = hard_ternary_weight(
            getattr(model, name), group_size=2, row_batch=128
        ).to(torch.bfloat16).float()
        torch.testing.assert_close(
            model.quantized(getattr(model, name)), expected_float, rtol=0, atol=0
        )
        expected = expected_float.to(torch.bfloat16)
        torch.testing.assert_close(exported[name], expected, rtol=0, atol=0)


def test_completed_checkpoint_resumes_without_changing_hard_weights(tmp_path) -> None:
    torch.manual_seed(9)
    x = torch.randn(8, 4)
    gate = torch.randn(6, 4)
    up = torch.randn(6, 4)
    down = torch.randn(4, 6)
    y = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, gate))
        * torch.nn.functional.linear(x, up),
        down,
    )
    args = SimpleNamespace(
        mode="ternary",
        group_size=2,
        artifact_exact_ternary=True,
        lr=1e-4,
        weight_decay=0.0,
        seed=17,
        steps_per_layer=2,
        batch_size=4,
        code_weight=0.001,
        grad_clip=1.0,
        log_every=1,
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
        resume=False,
        activation_cache_sha256="activation",
        inventory_sha256="inventory",
        base_patch_sha256=None,
    )

    first, first_report = train_layer(
        layer_id=3,
        x=x,
        y=y,
        gate=gate,
        up=up,
        down=down,
        args=args,
        device=torch.device("cpu"),
    )
    args.resume = True
    resumed, resumed_report = train_layer(
        layer_id=3,
        x=x,
        y=y,
        gate=gate,
        up=up,
        down=down,
        args=args,
        device=torch.device("cpu"),
    )

    assert first_report["checkpoint_sha256"] == resumed_report["checkpoint_sha256"]
    assert resumed_report["resumed"] is True
    assert len(resumed_report["history"]) == 2
    for name in ("gate", "up", "down"):
        torch.testing.assert_close(first[name], resumed[name], rtol=0, atol=0)


def test_activation_row_selection_is_exact_and_rejects_duplicates() -> None:
    rows = [{"positions": 2}, {"positions": 3}, {"positions": 1}]
    assert parse_row_indices("0,2", 3, allow_empty=False) == [0, 2]
    assert activation_position_indices(rows, [0, 2], 6).tolist() == [0, 1, 5]

    try:
        parse_row_indices("1,1", 3, allow_empty=False)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate row indices must fail")


def test_validation_selection_can_keep_and_resume_the_initial_exact_artifact(
    tmp_path,
) -> None:
    torch.manual_seed(41)
    train_x = torch.randn(6, 4)
    validation_x = torch.randn(4, 4)
    gate = torch.randn(6, 4)
    up = torch.randn(6, 4)
    down = torch.randn(4, 6)

    def target(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(x, gate))
            * torch.nn.functional.linear(x, up),
            down,
        )

    initial = QuantMLPTriplet(
        gate,
        up,
        down,
        mode="ternary",
        group_size=2,
        artifact_exact_ternary=True,
    ).hard_weights()
    args = SimpleNamespace(
        mode="ternary",
        group_size=2,
        artifact_exact_ternary=True,
        lr=0.0,
        weight_decay=0.0,
        seed=23,
        steps_per_layer=1,
        batch_size=3,
        code_weight=0.001,
        grad_clip=1.0,
        log_every=1,
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
        resume=False,
        activation_cache_sha256="activation",
        inventory_sha256="inventory",
        base_patch_sha256=None,
        validation_every=1,
        train_row_indices_resolved=[0],
        validation_row_indices_resolved=[1],
    )
    selected, report = train_layer(
        layer_id=3,
        x=train_x,
        y=target(train_x),
        gate=gate,
        up=up,
        down=down,
        args=args,
        device=torch.device("cpu"),
        validation_x=validation_x,
        validation_y=target(validation_x),
    )

    assert report["selected_best_step"] == 0
    assert [row["step"] for row in report["validation_history"]] == [0, 1]
    for name in ("gate", "up", "down"):
        torch.testing.assert_close(selected[name], initial[name], rtol=0, atol=0)

    args.resume = True
    resumed, resumed_report = train_layer(
        layer_id=3,
        x=train_x,
        y=target(train_x),
        gate=gate,
        up=up,
        down=down,
        args=args,
        device=torch.device("cpu"),
        validation_x=validation_x,
        validation_y=target(validation_x),
    )
    assert resumed_report["resumed"] is True
    assert resumed_report["selected_best_step"] == 0
    for name in ("gate", "up", "down"):
        torch.testing.assert_close(resumed[name], initial[name], rtol=0, atol=0)


def test_exact_weight_reconstruction_metric_is_zero_for_its_own_target() -> None:
    torch.manual_seed(53)
    x = torch.randn(5, 4)
    weights = {
        "gate": torch.randn(6, 4),
        "up": torch.randn(6, 4),
        "down": torch.randn(4, 6),
    }
    y = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, weights["gate"]))
        * torch.nn.functional.linear(x, weights["up"]),
        weights["down"],
    )
    loss = normalized_loss_from_weights(x, y, weights, batch_size=2)
    assert loss == 0.0
