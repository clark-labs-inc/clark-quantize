from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from clark_quantize.kd_data_manifest import seal_kd_arms, verify_kd_arms


class KDDataManifestTests(unittest.TestCase):
    def test_seal_verify_and_corruption_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            np.arange(32, dtype=np.uint32).tofile(root / "math.bin")
            manifest_path = root / "manifest.json"
            sealed = seal_kd_arms(
                root,
                arms=["math"],
                model="Qwen/test",
                revision="abc",
                probe_tokens=8,
                output=manifest_path,
            )
            self.assertEqual(sealed["arms"]["math"]["tokens"], 32)
            verified = verify_kd_arms(
                manifest_path,
                root=root,
                arms=["math"],
                model="Qwen/test",
                revision="abc",
                probe_tokens=8,
            )
            self.assertEqual(verified, sealed)
            path = root / "math.bin"
            damaged = bytearray(path.read_bytes())
            damaged[3] ^= 1
            path.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_kd_arms(
                    manifest_path,
                    root=root,
                    arms=["math"],
                    model="Qwen/test",
                    revision="abc",
                    probe_tokens=8,
                )

    def test_rejects_manifest_outside_root_and_short_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            np.arange(4, dtype=np.uint32).tofile(root / "math.bin")
            with self.assertRaisesRegex(ValueError, "directly inside"):
                seal_kd_arms(
                    root,
                    arms=["math"],
                    model="test",
                    revision="abc",
                    probe_tokens=2,
                    output=root.parent / "manifest.json",
                )
            with self.assertRaisesRegex(ValueError, "no training tokens"):
                seal_kd_arms(
                    root,
                    arms=["math"],
                    model="test",
                    revision="abc",
                    probe_tokens=4,
                    output=root / "manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
