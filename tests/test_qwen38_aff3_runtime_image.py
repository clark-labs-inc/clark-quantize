from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clark_quantize.qwen38_aff3_runtime_image import (
    Aff3RuntimeImage,
    Aff3RuntimeImageError,
    build_sections,
    pack_x16_weights,
    unpack_x16_weights,
    write_runtime_image,
)


def fixture() -> tuple[dict[str, bytes], dict[str, int], dict[str, np.ndarray]]:
    rows = 16
    columns = 256
    groups = columns // 128
    codes = np.asarray(
        (
            np.arange(rows, dtype=np.int64)[:, None] * 17
            + np.arange(columns, dtype=np.int64)[None, :] * 29
        )
        % 3
        - 1,
        dtype=np.int8,
    )
    alpha_indices = np.asarray(
        (
            np.arange(rows, dtype=np.uint16)[:, None]
            + np.arange(groups, dtype=np.uint16)[None, :]
        )
        % 16,
        dtype=np.uint8,
    )
    mu_indices = np.asarray((alpha_indices * 7 + 3) % 16, dtype=np.uint8)
    pairings = np.empty((2, columns // 2, 2), dtype=np.uint16)
    pairings[0] = np.arange(columns, dtype=np.uint16).reshape(-1, 2)
    pairings[1, :, 0] = np.arange(0, columns, 2, dtype=np.uint16)
    pairings[1, :, 1] = np.roll(np.arange(1, columns, 2, dtype=np.uint16), 1)
    angles = np.linspace(-0.1, 0.1, pairings.shape[0] * pairings.shape[1]).reshape(
        pairings.shape[:2]
    )
    source = {
        "codes": codes,
        "alpha_indices": alpha_indices,
        "mu_indices": mu_indices,
        "alpha_anchors": np.linspace(0.01, 0.02, rows),
        "mu_anchors": np.linspace(0.001, 0.002, rows),
        "alpha_codebook": np.linspace(0.5, 1.5, 16),
        "mu_codebook": np.linspace(-1.0, 1.0, 16),
        "angles": angles,
        "pairings": pairings,
    }
    sections, dimensions = build_sections(**source)
    return sections, dimensions, source


def test_x16_transcode_roundtrips_every_code_and_index() -> None:
    _, _, source = fixture()
    packed = pack_x16_weights(
        source["codes"], source["alpha_indices"], source["mu_indices"]
    )
    codes, alpha, mu = unpack_x16_weights(packed, rows=16, columns=256)
    assert np.array_equal(codes, source["codes"])
    assert np.array_equal(alpha, source["alpha_indices"])
    assert np.array_equal(mu, source["mu_indices"])


def test_runtime_image_is_self_authenticating_and_complete(tmp_path: Path) -> None:
    sections, dimensions, source = fixture()
    path = tmp_path / "layer-00.aff3x16"
    receipt = write_runtime_image(
        path, layer=0, sections=sections, dimensions=dimensions
    )
    assert receipt["shape"] == [16, 256]
    assert receipt["persistent_dense_weight_bytes"] == 0
    assert receipt["decoded_group_scale_matrix_bytes"] == 0
    with Aff3RuntimeImage(path) as image:
        image.verify_all_sections()
        assert image.read_section("weights_x16") == sections["weights_x16"]
        assert np.array_equal(
            np.frombuffer(image.read_section("angles"), dtype="<f2"),
            np.asarray(source["angles"], dtype="<f2").reshape(-1),
        )


def test_runtime_image_rejects_payload_corruption(tmp_path: Path) -> None:
    sections, dimensions, _ = fixture()
    path = tmp_path / "layer-00.aff3x16"
    receipt = write_runtime_image(
        path, layer=0, sections=sections, dimensions=dimensions
    )
    offset = receipt["sections"]["weights_x16"]["offset"]
    with path.open("r+b") as handle:
        handle.seek(offset)
        original = handle.read(1)
        handle.seek(offset)
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises(Aff3RuntimeImageError, match="root hash"):
        Aff3RuntimeImage(path)


def test_runtime_image_rejects_nonmatching_rotation() -> None:
    _, _, source = fixture()
    source["pairings"] = source["pairings"].copy()
    source["pairings"][0, 0, 0] = source["pairings"][0, 0, 1]
    with pytest.raises(ValueError, match="perfect matching"):
        build_sections(**source)
