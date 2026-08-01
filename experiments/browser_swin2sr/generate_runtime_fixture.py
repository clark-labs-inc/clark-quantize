#!/usr/bin/env python3
"""Create a browser-runtime fixture from a real Swin2SR attention matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Swin2SRForImageSuperResolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clark_air.quant.optimal_ternary import quantize_ternary_mse
from clark_air.quant.packing import pack_trits_20


MODEL_ID = "caidas/swin2SR-lightweight-x2-64"
MODEL_REVISION = "07e8e61492ce6daf4f34c651622960faf110c17f"
WEIGHT_NAME = "swin2sr.encoder.stages.0.layers.0.attention.self.query.weight"
BIAS_NAME = WEIGHT_NAME.removesuffix("weight") + "bias"
GROUP_SIZE = 60
BATCH_SIZE = 3


def main() -> None:
    model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).eval()
    state = model.state_dict()
    weight = state[WEIGHT_NAME].float()
    bias = state[BIAS_NAME].float()
    quantized = quantize_ternary_mse(weight, group_size=GROUP_SIZE)
    packed, _, pad = pack_trits_20(quantized.codes.numpy())
    generator = torch.Generator().manual_seed(20260729)
    input_batch = torch.randn(
        BATCH_SIZE,
        weight.shape[1],
        generator=generator,
    )
    expected = torch.nn.functional.linear(
        input_batch,
        quantized.dequantize(),
        bias,
    )

    fixture = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tensor": WEIGHT_NAME,
        "format": "clark-air-trit20-v1",
        "shape": list(weight.shape),
        "batch_size": BATCH_SIZE,
        "pad": pad,
        "group_size": GROUP_SIZE,
        "packed": packed.astype(np.uint32).tolist(),
        "scales": quantized.scales.float().tolist(),
        "bias": bias.tolist(),
        "input": input_batch.reshape(-1).tolist(),
        "expected": expected.reshape(-1).tolist(),
    }
    output = Path(__file__).parent / "runtime" / "fixture.json"
    output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(output)


if __name__ == "__main__":
    main()
