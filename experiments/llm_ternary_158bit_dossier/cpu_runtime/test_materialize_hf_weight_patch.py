import json
import os

import torch
from safetensors.torch import load_file, save_file

from materialize_hf_weight_patch import materialize


def test_materialize_rewrites_only_patch_shard(tmp_path) -> None:
    model_dir = tmp_path / "parent"
    model_dir.mkdir()
    shard_a = model_dir / "model-00001-of-00002.safetensors"
    shard_b = model_dir / "model-00002-of-00002.safetensors"
    save_file({"a": torch.arange(4, dtype=torch.bfloat16), "b": torch.ones(3, dtype=torch.bfloat16)}, shard_a)
    save_file({"c": torch.zeros(5, dtype=torch.bfloat16)}, shard_b)
    index = {
        "metadata": {"total_size": 24},
        "weight_map": {
            "a": shard_a.name,
            "b": shard_a.name,
            "c": shard_b.name,
        },
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    (model_dir / "config.json").write_text("{}\n")
    patch_path = tmp_path / "patch.safetensors"
    save_file({"b": torch.full((3,), 2, dtype=torch.bfloat16)}, patch_path)
    out_dir = tmp_path / "candidate"
    receipt_path = tmp_path / "receipt.json"

    receipt = materialize(model_dir, patch_path, out_dir, receipt_path)

    assert torch.equal(load_file(out_dir / shard_a.name)["b"], torch.full((3,), 2, dtype=torch.bfloat16))
    assert torch.equal(load_file(model_dir / shard_a.name)["b"], torch.ones(3, dtype=torch.bfloat16))
    assert os.stat(model_dir / shard_b.name).st_ino == os.stat(out_dir / shard_b.name).st_ino
    assert os.stat(model_dir / shard_a.name).st_ino != os.stat(out_dir / shard_a.name).st_ino
    assert receipt["patch_tensor_count"] == 1
    assert receipt["verification"].startswith("all patched tensors exact")
