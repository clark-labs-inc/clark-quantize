from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.torch import save_file
import torch

from clark_quantize.base3 import frame_metadata, pack_base3


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/qwen38_27b_spot/audit_completion.py"
SPEC = importlib.util.spec_from_file_location("qwen_completion_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompletionAuditTests(unittest.TestCase):
    def test_git_revision_drops_root_to_the_exact_source_owner(self) -> None:
        source = Path("/workspace/git/clark-autoresearch")
        self.assertEqual(
            MODULE.git_revision_command(
                source,
                effective_uid=0,
                owner_uid=1001,
                owner_name="ubuntu",
            ),
            [
                "runuser",
                "--user",
                "ubuntu",
                "--",
                "git",
                "-C",
                "/workspace/git/clark-autoresearch",
                "rev-parse",
                "HEAD",
            ],
        )

    def test_git_revision_runs_directly_for_the_repository_owner(self) -> None:
        source = Path("/workspace/git/clark-autoresearch")
        self.assertEqual(
            MODULE.git_revision_command(
                source,
                effective_uid=1001,
                owner_uid=1001,
                owner_name="ubuntu",
            ),
            ["git", "-C", str(source), "rev-parse", "HEAD"],
        )

    def fixture(self, root: Path) -> dict:
        workspace = root / "workspace"
        state = workspace / "quant-state"
        model = workspace / "models/Qwen3.8-27B"
        runs = workspace / "quant-runs"
        checkpoints = runs / "qwen38-qat/checkpoints"
        artifact = runs / "qwen38-base3-full/artifacts/qwen38-optimal.cq3q"
        for path in (state, model, checkpoints, artifact / "codes.cq3d/chunks", artifact / "scales"):
            path.mkdir(parents=True, exist_ok=True)
        revision = "a" * 40
        index = model / "model.safetensors.index.json"
        index.write_text("{}")
        preflight = {
            "schema": "clark_qwen38_27b_preflight_v1",
            "download_receipt": {"revision": revision, "index_sha256": digest(index)},
            "parameter_ledger": {
                "total": 11,
                "ternary_payload": 9,
                "retained_high_precision": 2,
                "fp16_scale_count": 1,
                "base3_payload_bytes": 8,
            },
        }
        (state / "qwen38-preflight.json").write_text(json.dumps(preflight))

        manifest = artifact / "codes.cq3d/manifest.json"
        codes = np.asarray([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int8)
        frame = pack_base3(codes)
        frame_path = artifact / "codes.cq3d/chunks/00000000.cq3a"
        frame_path.write_bytes(frame)
        scales = np.asarray([0.5], dtype="<f2").tobytes()
        scale_path = artifact / "scales/00000000.f16"
        scale_path.write_bytes(scales)
        identity = {
            "schema": "qwen38_full_base3_identity_v1",
            "model_revision": revision,
            "index_sha256": digest(index),
            "quantizer": {"method": "optimal", "group_size": 128, "scale_dtype": "float16"},
            "row_batch": 128,
            "mapped_sources": [{
                "name": "mapped.weight",
                "shard": "model.safetensors",
                "shape": [1, 9],
                "category": "language",
                "trits": 9,
            }],
        }
        identity_bytes = (
            json.dumps(identity, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
        ).encode()
        record = {
            "index": 0,
            "descriptor": {
                "tensor": "mapped.weight",
                "category": "language",
                "row_start": 0,
                "row_stop": 1,
                "tensor_shape": [1, 9],
                "scale_shape": [1, 1],
                "scale_file": scale_path.name,
                "scale_bytes": len(scales),
                "scale_sha256": hashlib.sha256(scales).hexdigest(),
            },
            "trits": 9,
            "raw_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
            "frame_sha256": hashlib.sha256(frame).hexdigest(),
            "frame_metadata": frame_metadata(frame),
            "verified_roundtrip": True,
        }
        code_ledger = {
            "chunks": 1,
            "trits": 9,
            "payload_bits": record["frame_metadata"]["payload_bits"],
            "frame_bytes": len(frame),
        }
        manifest_payload = {
            "schema": "clark_chunked_base3_v1",
            "identity": identity,
            "identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "chunks": [record],
            "complete": True,
            "source_sha256": "b" * 64,
            "ledger": code_ledger,
            "extra": {
                "scale_source_sha256": "c" * 64,
                "scale_bytes": len(scales),
                "scale_count": 1,
                "parallel_independently_verified_trits": 9,
            },
        }
        manifest.write_text(json.dumps(manifest_payload))
        retained_file = runs / "retained.safetensors"
        retained_manifest = runs / "retained.manifest.json"
        heldout = runs / "heldout.pt"
        calibration = runs / "calibration.pt"
        retained_tensor = torch.tensor([1, 2], dtype=torch.bfloat16)
        save_file(
            {"retained.weight": retained_tensor},
            retained_file,
            metadata={"schema": "clark_qwen38_retained_tensors_v1", "model_revision": revision},
        )
        retained_raw = retained_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        retained_manifest.write_text(json.dumps({
            "schema": "clark_qwen38_retained_tensor_manifest_v1",
            "model_revision": revision,
            "tensor_count": 1,
            "parameters": 2,
            "tensor_bytes": 4,
            "entries": [{"name": "retained.weight", "shape": [2], "dtype": "torch.bfloat16", "parameters": 2, "bytes": 4, "sha256": hashlib.sha256(retained_raw).hexdigest()}],
        }))
        for path, content in ((heldout, b"h"), (calibration, b"c")):
            path.write_bytes(content)
        checkpoint = checkpoints / "checkpoint-10"
        checkpoint.mkdir()
        checkpoint_state = checkpoint / "state.pt"
        checkpoint_state.write_bytes(b"q")
        (checkpoint / "checkpoint.json").write_text(json.dumps({
            "schema": "clark_qwen38_qat_checkpoint_v1", "complete": True, "step": 10,
            "state_file": "state.pt", "state_sha256": digest(checkpoint_state),
        }))

        retained_bytes = 4
        complete_payload_bytes = len(frame) + len(scales) + retained_bytes
        artifact_storage_bytes = sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())
        complete_transport_bytes = artifact_storage_bytes + retained_bytes
        effective_bpp = complete_payload_bytes * 8 / 11
        base = {"promotion": {"passed": True}, "model_revision": revision}
        receipts = {
            "qwen38-bf16-smoke.json": {"schema": MODULE.EXPECTED_SCHEMAS["bf16_smoke"], "dtype": "torch.bfloat16", "completion": "ok", "new_tokens": 1},
            "qwen38-base3-reference.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["reference"]},
            "qwen38-base3-resume.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["resume"], "identity": {"model_revision": revision}},
            "qwen38-base3-full.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["full_pack"], "identity": identity, "artifact": str(artifact), "code_manifest_sha256": digest(manifest), "code_source_sha256": "b" * 64, "scale_source_sha256": "c" * 64, "code_ledger": {**code_ledger, "payload_bits_per_trit": code_ledger["payload_bits"] / 9, "frame_bits_per_trit": len(frame) * 8 / 9, "delta_payload_bits_per_trit": code_ledger["payload_bits"] / 9 - math.log2(3)}, "complete_model_ledger": {"ternary_code_frame_bytes": len(frame), "fp16_scale_bytes": len(scales), "retained_high_precision_bytes": retained_bytes, "complete_payload_bytes": complete_payload_bytes, "payload_effective_bits_per_total_parameter": effective_bpp, "artifact_storage_bytes_excluding_retained_tensors": artifact_storage_bytes, "artifact_metadata_bytes": artifact_storage_bytes - len(frame) - len(scales), "complete_transport_bytes_including_referenced_retained_tensors": complete_transport_bytes, "transport_effective_bits_per_total_parameter": complete_transport_bytes * 8 / 11, "preflight_ideal_payload_bytes": 8, "payload_overhead_bytes_vs_preflight": complete_payload_bytes - 8, "transport_overhead_bytes_vs_preflight": complete_transport_bytes - 8, "retained_tensor_storage": "referenced from immutable source checkpoint; bytes included in ledger, not duplicated"}, "performance": {"parallel_independently_verified_trits": 9}},
            "qwen38-retained-export.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["retained_export"], "retained": {"path": str(retained_file), "file_sha256": digest(retained_file), "manifest": str(retained_manifest), "manifest_sha256": digest(retained_manifest), "tensor_count": 1, "parameters": 2, "tensor_bytes": 4}, "complete_parameter_bundle": {"ternary_parameters": 9, "retained_parameters": 2, "total_parameters": 11, "ternary_code_frame_bytes": len(frame), "fp16_scale_bytes": len(scales), "retained_tensor_bytes": retained_bytes, "complete_payload_bytes": complete_payload_bytes, "effective_bits_per_total_parameter": effective_bpp}},
            "qwen38-base3-quality.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["quality"], "promotion": {"passed": False}, "teacher_cache": {"path": str(heldout), "sha256": digest(heldout)}, "applied": {"parameters": 9}, "artifact_integrity": {"valid": True, "artifact": str(artifact), "manifest_sha256": digest(manifest), "code_source_sha256": "b" * 64, "scale_source_sha256": "c" * 64, "model_revision": revision, "index_sha256": digest(index), "trits": 9}, "bf16": {"likelihood": {"mean_nll": 1.0, "perplexity": 2.0}}, "ternary": {"likelihood": {"mean_nll": 7.0, "perplexity": 1000.0}}, "comparison": {"perplexity_ratio": 500.0, "mean_teacher_kl": 5.0, "all_bf16_and_ternary_completions_nonempty": True, "bf16_generations_deterministic": True, "ternary_generations_deterministic": True, "bf16_generation_sha256": "f" * 64, "bf16_generation_repeat_sha256": "f" * 64, "ternary_generation_sha256": "0" * 64, "ternary_generation_repeat_sha256": "0" * 64}},
            "qwen38-radix5-performance.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["radix5"], "sample": {"trits": 10000000, "source_sha256": "e" * 64, "sources": [{"used_trits": 10000000}]}, "format": {"payload_bits_per_trit": 1.6, "delta_bits_per_trit": 1.6 - math.log2(3), "exact_roundtrip": True}, "performance": {"encode_mtrits_per_second": 60.0, "decode_mtrits_per_second": 70.0}, "full_model_projection": {"extra_bytes_vs_ideal_base3": 1, "complete_payload_bytes": 9, "effective_bits_per_total_parameter": 72 / 11}},
            "qwen38-layer-sensitivity.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["sensitivity"]},
            "qwen38-qat-teacher-cache.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["teacher_cache"], "cache": str(calibration), "cache_sha256": digest(calibration)},
            "qwen38-one-layer-qat.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["qat"], "promotion": {"passed": False}, "checkpoint": {"path": str(checkpoint), "state_sha256": digest(checkpoint_state), "exact_resume_smoke": True}},
            "qwen38-target-158-reconstruction.json": {**base, "schema": MODULE.EXPECTED_SCHEMAS["target_158"], "selected_candidate": "all_rank2_per_row_fp16_scales", "ledgers": {"all_rank2_per_row_fp16_scales": {"complete_effective_bits_per_parameter": 1.5902}}, "coverage": {"language": True, "vision": True, "mtp": True, "linear_attention": True, "convolution_retained_control": True, "all_rank_not_2_bfloat16": True}, "promotion": {"passed": True, "authorizes_full_pack": False}},
        }
        for name, receipt in receipts.items():
            (state / name).write_text(json.dumps(receipt))

        source_root = workspace / "git"
        archive_root = workspace / "source-archives"
        archive_root.mkdir()
        source_rows = {
            "clark_private_product": ("1" * 40, source_root / "clark", "clark.tar"),
            "clark_desktop_foundation": ("2" * 40, source_root / "clark-desktop", "desktop.tar"),
            "clark_autoresearch": ("3" * 40, source_root / "clark-autoresearch", "autoresearch.tar"),
            "clark_agent_compaction": ("4" * 40, source_root / "clark-agent-compaction", "compaction.tar"),
        }
        provenance = {"schema": "clark_qwen38_source_provenance_v1"}
        for key, (source_revision, source_path, archive_name) in source_rows.items():
            source_path.mkdir(parents=True)
            archive = archive_root / archive_name
            archive.write_bytes(key.encode())
            provenance[key] = {
                "revision": source_revision,
                "remote_path": str(source_path),
                "archive_path": str(archive),
                "archive_sha256": digest(archive),
            }
        provenance["clark_private_product"]["version"] = "0.1.155"
        source_provenance = workspace / "source_provenance.json"
        source_provenance.write_text(json.dumps(provenance))

        clark = workspace / "bin/clark-code-headless"
        clark.parent.mkdir()
        selftest = {
            "status": "passed",
            "worker": "clark-code-headless",
            "protocol_version": 1,
            "specialist_count": 2,
            "worker_version": "0.1.155",
            "catalog_sha256": "f" * 64,
        }
        clark.write_text("#!/bin/sh\nprintf '%s\\n' '" + json.dumps(selftest) + "'\n")
        clark.chmod(0o755)
        (state / "clark-scientist-update.json").write_text(json.dumps({
            "schema": "clark_scientist_quant_update_v1", "installed_path": str(clark),
            "clark_revision": source_rows["clark_private_product"][0],
            "clark_desktop_revision": source_rows["clark_desktop_foundation"][0],
            "autoresearch_revision": source_rows["clark_autoresearch"][0],
            "compaction_revision": source_rows["clark_agent_compaction"][0],
            "worker_sha256": digest(clark), "self_test": selftest,
            "canonical_source": str(source_rows["clark_private_product"][1] / "crates/clark-desktop-product/scientist"),
        }))
        return {"state": state, "model_dir": model, "checkpoints": checkpoints, "workspace": workspace, "clark_binary": clark, "source_provenance": source_provenance}

    def test_accepts_valid_nonpromoting_qat_as_completed_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            report = MODULE.audit(**args)
            self.assertTrue(report["complete"])
            self.assertEqual(report["quality_outcome"], "ptq_valid_nonpromotion")
            self.assertEqual(report["qat_outcome"], "ten_step_valid_nonpromotion")

    def test_rejects_nonfinite_or_nondeterministic_quality_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            path = args["state"] / "qwen38-base3-quality.json"
            receipt = json.loads(path.read_text())
            receipt["comparison"]["mean_teacher_kl"] = float("nan")
            receipt["comparison"]["ternary_generations_deterministic"] = False
            path.write_text(json.dumps(receipt))
            report = MODULE.audit(**args)
            failed = {row["name"] for row in report["checks"] if not row["passed"]}
            self.assertIn("quality_terminal_science", failed)
            self.assertFalse(report["complete"])

    def test_requires_100_step_receipt_only_after_ten_step_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            path = args["state"] / "qwen38-one-layer-qat.json"
            receipt = json.loads(path.read_text())
            receipt["promotion"]["passed"] = True
            path.write_text(json.dumps(receipt))
            report = MODULE.audit(**args)
            failed = {row["name"] for row in report["checks"] if not row["passed"]}
            self.assertIn("qat100_receipt", failed)
            self.assertFalse(report["complete"])

    def test_fails_on_artifact_tamper_or_checkpoint_excess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            artifact = args["workspace"] / "quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q/codes.cq3d/manifest.json"
            artifact.write_text("changed")
            for step in (20, 30):
                target = args["checkpoints"] / f"checkpoint-{step}"
                target.mkdir()
                state = target / "state.pt"
                state.write_bytes(str(step).encode())
                (target / "checkpoint.json").write_text(json.dumps({
                    "schema": "clark_qwen38_qat_checkpoint_v1", "complete": True, "step": step,
                    "state_file": "state.pt", "state_sha256": digest(state),
                }))
            report = MODULE.audit(**args)
            failed = {row["name"] for row in report["checks"] if not row["passed"]}
            self.assertIn("full_manifest_hash", failed)
            self.assertIn("checkpoint_retention", failed)
            self.assertFalse(report["complete"])

    def test_fails_when_frame_content_or_complete_byte_ledger_is_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            frame = args["workspace"] / "quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q/codes.cq3d/chunks/00000000.cq3a"
            damaged = bytearray(frame.read_bytes())
            damaged[-1] ^= 1
            frame.write_bytes(damaged)
            report = MODULE.audit(**args)
            failed = {row["name"] for row in report["checks"] if not row["passed"]}
            self.assertIn("full_exact_artifact_and_bit_ledger", failed)
            self.assertFalse(report["complete"])

        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            receipt_path = args["state"] / "qwen38-base3-full.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["complete_model_ledger"]["complete_payload_bytes"] += 1
            receipt_path.write_text(json.dumps(receipt))
            report = MODULE.audit(**args)
            failed = {row["name"] for row in report["checks"] if not row["passed"]}
            self.assertIn("full_exact_artifact_and_bit_ledger", failed)
            self.assertFalse(report["complete"])

    def test_transfer_scan_uses_executable_not_argument_substrings(self) -> None:
        processes = "\n".join((
            "101 bash bash -c 'run audit, then rsync source'",
            "102 python3 python3 audit_completion.py --note scp",
            "103 /usr/bin/rsync rsync -a source destination",
            "104 rclone rclone copy source destination",
            "105 /usr/bin/scp scp source destination",
        ))
        self.assertEqual(
            MODULE.active_transfer_processes(processes),
            [
                "103 /usr/bin/rsync rsync -a source destination",
                "104 rclone rclone copy source destination",
                "105 /usr/bin/scp scp source destination",
            ],
        )

    def test_fails_when_radix_throughput_or_projection_is_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            path = args["state"] / "qwen38-radix5-performance.json"
            receipt = json.loads(path.read_text())
            receipt["performance"]["encode_mtrits_per_second"] = 49.0
            receipt["full_model_projection"]["complete_payload_bytes"] += 1
            path.write_text(json.dumps(receipt))
            report = MODULE.audit(**args)
            failed = {row["name"] for row in report["checks"] if not row["passed"]}
            self.assertIn("radix5_exact_performance_and_projection", failed)
            self.assertFalse(report["complete"])

    def test_fails_when_retained_tensor_content_is_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            path = args["workspace"] / "quant-runs/retained.safetensors"
            damaged = bytearray(path.read_bytes())
            damaged[-1] ^= 1
            path.write_bytes(damaged)
            report = MODULE.audit(**args)
            failed = {row["name"] for row in report["checks"] if not row["passed"]}
            self.assertIn("retained_bundle_hash", failed)
            self.assertIn("retained_tensor_content_and_ledger", failed)
            self.assertFalse(report["complete"])

    def test_fails_when_clark_source_archive_or_revision_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            provenance = json.loads(args["source_provenance"].read_text())
            Path(provenance["clark_private_product"]["archive_path"]).write_bytes(b"changed")
            receipt_path = args["state"] / "clark-scientist-update.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["compaction_revision"] = "9" * 40
            receipt_path.write_text(json.dumps(receipt))
            report = MODULE.audit(**args)
            failed = {row["name"] for row in report["checks"] if not row["passed"]}
            self.assertIn("clark_source_provenance", failed)
            self.assertIn("clark_worker", failed)
            self.assertFalse(report["complete"])


if __name__ == "__main__":
    unittest.main()
