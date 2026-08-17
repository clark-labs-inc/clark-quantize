from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from clark_quantize.base3_block203 import pack_row, unpack_row
from clark_quantize.qwen38_cq3_container import (
    CQ3ContainerError,
    HEADER_BYTES,
    Qwen38CQ3Reader,
    Qwen38CQ3Writer,
)
from clark_quantize.scale_codebook import pack_nibbles, unpack_nibbles


class Qwen38CQ3ContainerTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, np.ndarray, np.ndarray]:
        path = root / "qwen38-test.cq3"
        codes = (np.arange(256, dtype=np.int16) % 3 - 1).astype(np.int8)
        scale_indices = np.asarray([1, 2], dtype=np.uint8)
        codebook = np.linspace(0.0, 0.25, 16, dtype=np.float16)
        retained = np.asarray([1.0, -2.0], dtype=np.float32).tobytes()
        with Qwen38CQ3Writer(
            path,
            model={"id": "Qwen/Qwen3.8-27B", "revision": "test-revision"},
            strategy={
                "name": "base3_group128_codebook4",
                "group_size": 128,
                "scale_bits": 4,
            },
            parameter_count=258,
        ) as writer:
            writer.add_section(
                "matrix.codes", "ternary_block203", pack_row(codes),
                metadata={"trits": 256, "group_bits": 203},
            )
            writer.add_section(
                "matrix.scale_indices", "scale_indices_u4", pack_nibbles(scale_indices),
                metadata={"indices": 2},
            )
            writer.add_section(
                "matrix.scale_codebook", "scale_codebook_f16", codebook.astype("<f2").tobytes(),
                metadata={"entries": 16},
            )
            writer.add_section(
                "norm.retained", "retained_f32", retained,
                metadata={"dtype": "F32"},
            )
            auxiliary = writer.add_section(
                "aux.config", "auxiliary_file", b"{}\n",
                metadata={"name": "config.json"},
            )
            receipt = writer.finalize(
                tensors=[
                    {
                        "name": "matrix.weight",
                        "shape": [2, 128],
                        "parameters": 256,
                        "encoding": "base3_group128_codebook4",
                        "sections": [
                            "matrix.codes",
                            "matrix.scale_indices",
                            "matrix.scale_codebook",
                        ],
                    },
                    {
                        "name": "norm.weight",
                        "shape": [2],
                        "parameters": 2,
                        "encoding": "retained_f32",
                        "sections": ["norm.retained"],
                    },
                ],
                auxiliary_files=[{
                    "name": "config.json",
                    "section": auxiliary["id"],
                    "bytes": auxiliary["bytes"],
                    "sha256": auxiliary["sha256"],
                }],
            )
        self.assertEqual(receipt["parameters"], 258)
        self.assertEqual(receipt["sections"], 5)
        return path, codes, scale_indices

    def test_atomic_roundtrip_and_complete_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, codes, scale_indices = self.fixture(Path(temporary))
            with Qwen38CQ3Reader(path) as reader:
                self.assertEqual(reader.header["parameter_count"], 258)
                self.assertEqual(len(reader.directory["tensors"]), 2)
                np.testing.assert_array_equal(
                    unpack_row(reader.read_section("matrix.codes"), 256), codes,
                )
                np.testing.assert_array_equal(
                    unpack_nibbles(
                        reader.read_section("matrix.scale_indices"), 2
                    ),
                    scale_indices,
                )
                result = reader.verify_all()
                self.assertEqual(result["sections_verified"], 5)

    def test_section_corruption_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self.fixture(Path(temporary))
            with Qwen38CQ3Reader(path) as reader:
                offset = reader.sections["matrix.codes"]["offset"]
            with path.open("r+b") as handle:
                handle.seek(offset)
                value = handle.read(1)
                handle.seek(offset)
                handle.write(bytes([value[0] ^ 1]))
            with Qwen38CQ3Reader(path) as reader:
                with self.assertRaisesRegex(CQ3ContainerError, "SHA-256"):
                    reader.verify_section("matrix.codes")

    def test_directory_and_header_corruption_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self.fixture(Path(temporary))
            content = bytearray(path.read_bytes())
            content[HEADER_BYTES - 1] ^= 1
            path.write_bytes(content)
            with self.assertRaisesRegex(CQ3ContainerError, "header"):
                Qwen38CQ3Reader(path)

    def test_inventory_and_atomic_abort_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "bad.cq3"
            with self.assertRaisesRegex(CQ3ContainerError, "section references"):
                with Qwen38CQ3Writer(
                    path,
                    model={"id": "Qwen/Qwen3.8-27B", "revision": "test"},
                    strategy={"name": "test"},
                    parameter_count=1,
                ) as writer:
                    writer.add_section("present", "retained", b"x")
                    writer.finalize(tensors=[{
                        "name": "weight",
                        "shape": [1],
                        "parameters": 1,
                        "sections": ["missing"],
                    }])
            self.assertFalse(path.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_orphan_sections_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "orphan.cq3"
            with self.assertRaisesRegex(CQ3ContainerError, "exactly once"):
                with Qwen38CQ3Writer(
                    path,
                    model={"id": "Qwen/Qwen3.8-27B", "revision": "test"},
                    strategy={"name": "test"},
                    parameter_count=1,
                ) as writer:
                    writer.add_section("used", "retained", b"x")
                    writer.add_section("orphan", "retained", b"y")
                    writer.finalize(tensors=[{
                        "name": "weight",
                        "shape": [1],
                        "parameters": 1,
                        "sections": ["used"],
                    }])
            self.assertFalse(path.exists())
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
