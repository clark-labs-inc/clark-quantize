import json

import pytest

from train_scale_codebook_recovery import load_target_map


def test_load_target_map_requires_versioned_valid_surface_bits(tmp_path):
    path = tmp_path / "target-map.json"
    path.write_text(json.dumps({
        "format": "backdropkit-swin2sr-per-surface-scale-map-v1",
        "target_bits_by_surface": {"z": 4, "a": 1},
    }))
    payload, mapping = load_target_map(path)
    assert payload["format"].endswith("-v1")
    assert mapping == {"a": 1, "z": 4}

    path.write_text(json.dumps({
        "format": "backdropkit-swin2sr-per-surface-scale-map-v1",
        "target_bits_by_surface": {"a": 9},
    }))
    with pytest.raises(ValueError, match="between one and eight"):
        load_target_map(path)
