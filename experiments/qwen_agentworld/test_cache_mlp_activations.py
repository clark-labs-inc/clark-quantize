import torch

from cache_mlp_activations import atomic_torch_save


def test_atomic_activation_cache_save_leaves_only_complete_target(tmp_path) -> None:
    target = tmp_path / "activations.pt"
    atomic_torch_save({"value": torch.tensor([1, 2, 3])}, target)

    loaded = torch.load(target, map_location="cpu")
    assert loaded["value"].tolist() == [1, 2, 3]
    assert not (tmp_path / ".activations.pt.tmp").exists()
