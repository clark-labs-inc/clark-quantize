import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).with_name("materialize_image_io_ablation.py")
SPEC = importlib.util.spec_from_file_location("materialize_image_io_ablation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def paired_states():
    names = (*MODULE.KEYS.values(), "unrelated.weight")
    ternary = {name: torch.zeros(2) for name in names}
    fp16 = {name: value.clone() for name, value in ternary.items()}
    fp16[MODULE.KEYS["input"]].fill_(1)
    fp16[MODULE.KEYS["reconstruction"]].fill_(2)
    return ternary, fp16


@pytest.mark.parametrize("choice", ["input", "reconstruction"])
def test_ablation_changes_exactly_one_declared_tensor(choice):
    ternary, fp16 = paired_states()
    output, receipt = MODULE.ablate_states(ternary, fp16, choice)
    selected = MODULE.KEYS[choice]
    assert torch.equal(output[selected], fp16[selected])
    assert receipt["changed_from_ternary"] == [selected]
    for name in output:
        if name != selected:
            assert torch.equal(output[name], ternary[name])


def test_ablation_rejects_cross_contaminated_pair():
    ternary, fp16 = paired_states()
    fp16["unrelated.weight"].fill_(3)
    with pytest.raises(ValueError, match="outside the exact image-I/O pair"):
        MODULE.ablate_states(ternary, fp16, "input")


def test_storage_projection_matches_sealed_group_two_geometry():
    assert MODULE.projected_delta_bytes(1_620)["projected_data_byte_delta"] == 1_296
    assert MODULE.projected_delta_bytes(6_480)["projected_data_byte_delta"] == 5_184
