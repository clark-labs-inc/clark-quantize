import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).with_name("compose_disjoint_ternary_recovery.py")
SPEC = importlib.util.spec_from_file_location(
    "compose_disjoint_ternary_recovery", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def states():
    names = (*MODULE.IMAGE_KEYS, *MODULE.RSTB_KEYS, "unrelated.weight")
    parent = {name: torch.zeros(2) for name in names}
    image = {name: value.clone() for name, value in parent.items()}
    rstb = {name: value.clone() for name, value in parent.items()}
    for index, name in enumerate(MODULE.IMAGE_KEYS, 1):
        image[name].fill_(index)
    for index, name in enumerate(MODULE.RSTB_KEYS, 1):
        rstb[name].fill_(-index)
    return parent, image, rstb


def test_composition_is_the_exact_disjoint_union():
    parent, image, rstb = states()
    composed, receipt = MODULE.compose_states(parent, image, rstb)
    for name in MODULE.IMAGE_KEYS:
        assert torch.equal(composed[name], image[name])
    for name in MODULE.RSTB_KEYS:
        assert torch.equal(composed[name], rstb[name])
    assert torch.equal(composed["unrelated.weight"], parent["unrelated.weight"])
    assert set(receipt["changed_from_parent"]) == {
        *MODULE.IMAGE_KEYS,
        *MODULE.RSTB_KEYS,
    }


def test_image_cross_contamination_is_rejected():
    parent, image, rstb = states()
    image["unrelated.weight"].fill_(1)
    with pytest.raises(ValueError, match="image source changed undeclared"):
        MODULE.compose_states(parent, image, rstb)


def test_rstb_cross_contamination_is_rejected():
    parent, image, rstb = states()
    rstb["unrelated.weight"].fill_(1)
    with pytest.raises(ValueError, match="RSTB source changed undeclared"):
        MODULE.compose_states(parent, image, rstb)


def test_noop_source_is_rejected():
    parent, image, rstb = states()
    with pytest.raises(ValueError, match="image source made no exact"):
        MODULE.compose_states(parent, parent, rstb)
