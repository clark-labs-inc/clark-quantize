from types import SimpleNamespace

import torch

from clark_quantize.qwen_qat import hard_ternary_weight
from train_mlp_triplet_qat import QuantMLPTriplet, train_layer


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
        expected = hard_ternary_weight(
            getattr(model, name), group_size=2, row_batch=128
        ).to(torch.bfloat16)
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
