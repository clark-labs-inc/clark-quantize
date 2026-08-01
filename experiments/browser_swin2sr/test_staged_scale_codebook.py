from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.utils import parametrize

from full_surface_provenance import file_sha256
from quantizer_controls import ProgressiveLowBit
from scale_codebook_qat import add_scale_codebooks
from scale_codebook_state import (
    MATERIALIZATION_FORMAT,
    load_scale_codebook_materialized_state,
)
from staged_scale_codebook import (
    add_staged_collapse,
    bridge_summary,
    require_hard_bridge,
    set_staged_collapse_progress,
)
from staged_scale_codebook_state import (
    STAGED_CHECKPOINT_FORMAT,
    load_staged_scale_codebook_student,
    materialize_staged_scale_codebook_checkpoint,
)


def make_bridge():
    torch.manual_seed(17)
    model = torch.nn.Module()
    model.add_module("layer", torch.nn.Linear(4, 2, bias=False))
    quantizer = ProgressiveLowBit(
        "ternary", "absmean", 2, 0.0, 1.0, model.layer.weight
    )
    parametrize.register_parametrization(model.layer, "weight", quantizer)
    quantizer.amount.fill_(1.0)
    surface = SimpleNamespace(
        name="layer",
        category="core_linear",
        quantizer=quantizer,
        module=model.layer,
    )
    bridge = add_scale_codebooks(
        [surface], target_bits_by_category={"core_linear": 2}
    )
    bridge[0].projection.amount.fill_(1.0)
    bridge_payload = {
        "parent_checkpoint": "/frozen/hard-parent.pt",
        "parent_checkpoint_sha256": "parent-sha",
        "scale_codebook": {
            "target_bits_by_category": {"core_linear": 2},
            "steps": 4,
        },
    }
    return model, [surface], bridge, bridge_payload, {"parent": True}


def test_staged_collapse_is_exact_at_zero_and_freezes_bridge():
    model, surfaces, bridge, _payload, _parent = make_bridge()
    bridge_output = model.layer.weight.detach().clone()
    collapse = add_staged_collapse(
        surfaces,
        bridge,
        target_bits_by_category={"core_linear": 1},
    )
    assert torch.equal(model.layer.weight, bridge_output)
    assert not bridge[0].projection.raw_centers.requires_grad
    assert collapse[0].projection.raw_centers.requires_grad
    assert float(collapse[0].projection.amount) == 0.0
    assert len(model.layer.parametrizations.weight) == 3
    assert bridge_summary(bridge)["frozen_center_parameters"] == (
        bridge[0].projection.raw_centers.numel()
    )


def test_staged_collapse_rejects_a_soft_bridge():
    _model, _surfaces, bridge, _payload, _parent = make_bridge()
    bridge[0].projection.amount.fill_(0.5)
    try:
        require_hard_bridge(bridge)
    except ValueError as error:
        assert "not hard" in str(error)
    else:
        raise AssertionError("a transitioning bridge must be rejected")


def test_gentle_io_schedule_transitions_sensitive_io_for_half_the_run():
    _model, surfaces, bridge, _payload, _parent = make_bridge()
    surfaces[0].category = "image_io_conv"
    bridge[0] = type(bridge[0])(
        bridge[0].name,
        "image_io_conv",
        bridge[0].target_bits,
        bridge[0].projection,
        bridge[0].module,
    )
    collapse = add_staged_collapse(
        surfaces,
        bridge,
        target_bits_by_category={"image_io_conv": 1},
    )
    set_staged_collapse_progress(collapse, 0.50, "gentle_io_v1")
    assert float(collapse[0].projection.amount) == 0.0
    set_staged_collapse_progress(collapse, 0.75, "gentle_io_v1")
    assert float(collapse[0].projection.amount) == 0.5
    set_staged_collapse_progress(collapse, 1.00, "gentle_io_v1")
    assert float(collapse[0].projection.amount) == 1.0


def test_staged_checkpoint_round_trip_and_materialization(tmp_path, monkeypatch):
    bridge_path = tmp_path / "bridge.pt"
    bridge_path.write_bytes(b"sealed bridge identity")
    model, surfaces, bridge, bridge_payload, parent_payload = make_bridge()
    collapse = add_staged_collapse(
        surfaces,
        bridge,
        target_bits_by_category={"core_linear": 1},
    )
    collapse[0].projection.amount.fill_(0.75)
    expected = model.layer.weight.detach().clone()
    payload = {
        "format": STAGED_CHECKPOINT_FORMAT,
        "bridge_checkpoint": str(bridge_path),
        "bridge_checkpoint_sha256": file_sha256(bridge_path),
        "model": model.state_dict(),
        "collapse_step": 3,
        "quantization": {
            "mode": "ternary",
            "scope": "full_surface",
            "method": "absmean",
            "group_size": 2,
        },
        "surface_training": {},
        "staged_scale_codebook": {
            "target_bits_by_category": {"core_linear": 1},
        },
        "surfaces": {
            "layer.weight": {
                "category": "core_linear",
                "group_size": 2,
                "method": "absmean",
                "rotation": "none",
                "rotation_block": 4,
                "target_bits": 1,
            },
        },
    }
    checkpoint = tmp_path / "collapse.pt"
    torch.save(payload, checkpoint)

    def fake_bridge_loader(_path: Path, _device: str):
        return make_bridge()

    monkeypatch.setattr(
        "staged_scale_codebook_state.load_scale_codebook_student",
        fake_bridge_loader,
    )
    loaded = load_staged_scale_codebook_student(checkpoint, "cpu")
    assert torch.equal(loaded[0].layer.weight, expected)
    assert not loaded[2][0].projection.raw_centers.requires_grad
    assert float(loaded[3][0].projection.amount) == 0.75

    materialized = materialize_staged_scale_codebook_checkpoint(
        checkpoint,
        tmp_path / "materialized",
        "cpu",
    )
    assert materialized["format"] == MATERIALIZATION_FORMAT
    assert materialized["all_scale_codebooks_materialized_hard"]
    assert materialized["surfaces"]["layer.weight"]["target_bits"] == 1
    manifest, dense = load_scale_codebook_materialized_state(
        Path(materialized["manifest"]), "dense_state"
    )
    assert manifest["scale_codebook"]["target_bits_by_category"] == {
        "core_linear": 1
    }
    assert set(dense) == {"layer.weight"}
