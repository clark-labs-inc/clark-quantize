from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from materialize_surface_states import (
    FORMAT,
    load_materialized_state,
    state_receipt,
    state_sha256,
)


class MaterializeSurfaceStatesTest(unittest.TestCase):
    def test_state_hash_is_mapping_order_independent(self) -> None:
        first = {
            "a": torch.tensor([1.0]),
            "b": torch.tensor([2.0]),
        }
        second = {"b": first["b"], "a": first["a"]}
        self.assertEqual(state_sha256(first), state_sha256(second))

    def test_loads_state_only_when_file_and_tensor_identity_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {"weight": torch.tensor([1.0, -1.0])}
            state_path = root / "runtime.pt"
            torch.save(state, state_path)
            manifest = {
                "format": FORMAT,
                "runtime_state": state_receipt(state_path, state),
            }
            manifest_path = root / "materialization.json"
            manifest_path.write_text(json.dumps(manifest))

            loaded_manifest, loaded = load_materialized_state(
                manifest_path,
                "runtime_state",
            )

            self.assertEqual(loaded_manifest["format"], FORMAT)
            torch.testing.assert_close(loaded["weight"], state["weight"])

    def test_rejects_tampered_serialized_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {"weight": torch.tensor([1.0, -1.0])}
            state_path = root / "runtime.pt"
            torch.save(state, state_path)
            manifest = {
                "format": FORMAT,
                "runtime_state": state_receipt(state_path, state),
            }
            manifest_path = root / "materialization.json"
            manifest_path.write_text(json.dumps(manifest))
            state_path.write_bytes(state_path.read_bytes() + b"tampered")

            with self.assertRaisesRegex(ValueError, "file identity mismatch"):
                load_materialized_state(
                    manifest_path,
                    "runtime_state",
                )


if __name__ == "__main__":
    unittest.main()
