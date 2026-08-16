from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qwen_agentworld.train_all2d_ternary_qat import (
    FrozenQuantLinear,
    TrainQuantLinear,
    replace_all_2d_modules,
)


class TinyMultimodal(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = torch.nn.Module()
        self.visual.proj = torch.nn.Linear(8, 8, bias=False)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"proj": torch.nn.Linear(8, 8, bias=False)})]
        )
        self.mtp = torch.nn.Linear(8, 8, bias=False)


def main() -> None:
    model = TinyMultimodal()
    args = argparse.Namespace(
        skip_quant_embed=False,
        skip_quant_head=False,
        skip_quant_regex=r"(?:^|\.)(?:visual|mtp)(?:\.|$)",
        train_regex=None,
        train_embed=False,
        train_head=False,
        train_all_layers=True,
        train_layer_set=set(),
        mode="ternary",
        group_size=8,
        alpha_start=1.0,
        master_dtype=torch.bfloat16,
    )
    all_quant, train_quant = replace_all_2d_modules(model, args)
    assert set(all_quant) == {"model.layers.0.proj"}
    assert set(train_quant) == {"model.layers.0.proj"}
    assert isinstance(model.model.layers[0]["proj"], TrainQuantLinear)
    assert isinstance(model.visual.proj, torch.nn.Linear)
    assert isinstance(model.mtp, torch.nn.Linear)

    args.train_all_layers = False
    model = TinyMultimodal()
    all_quant, train_quant = replace_all_2d_modules(model, args)
    assert isinstance(model.model.layers[0]["proj"], FrozenQuantLinear)
    assert not train_quant
    print("all2d scope tests passed")


if __name__ == "__main__":
    main()
