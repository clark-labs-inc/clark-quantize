#!/usr/bin/env python3
"""Fail-closed completion audit for the A100 Qwen3.8-27B study.

The individual experiment stages already validate their own expensive work.
This audit binds their receipts back to one pinned model revision and to the
files that must still exist locally.  A scientifically valid QAT non-promotion
is a completed result; it is reported separately rather than rewritten as an
infrastructure failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import struct
import subprocess
import sys
from typing import Any
import zlib


EXPECTED_SCHEMAS = {
    "bf16_smoke": "clark_qwen38_bf16_smoke_v1",
    "reference": "clark_qwen38_base3_reference_v1",
    "resume": "clark_qwen38_base3_resume_gate_v1",
    "full_pack": "clark_qwen38_full_base3_pack_v1",
    "retained_export": "clark_qwen38_retained_export_receipt_v1",
    "quality": "clark_qwen38_base3_behavioral_quality_v1",
    "radix5": "clark_qwen38_radix5_performance_v1",
    "sensitivity": "clark_qwen38_layer_sensitivity_v1",
    "teacher_cache": "clark_qwen38_qat_teacher_cache_receipt_v1",
    "qat": "clark_qwen38_one_layer_qat_v1",
    "target_158": "clark_qwen38_target_158_reconstruction_v1",
}
CHECKPOINT_NAME = re.compile(r"checkpoint-([0-9]+)$")
BASE3_HEADER = struct.Struct(">5sQQI")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def is_local_workspace_path(value: Any, workspace: Path) -> bool:
    if not isinstance(value, str):
        return False
    path = Path(value)
    if not path.is_absolute():
        return False
    try:
        path.resolve(strict=False).relative_to(workspace.resolve())
    except ValueError:
        return False
    return True


def active_transfer_processes(process_text: str) -> list[str]:
    """Return actual transfer executables, not shell arguments mentioning one."""
    matches: list[str] = []
    for line in process_text.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 2:
            continue
        executable = Path(fields[1]).name
        if executable in {"rsync", "rclone", "scp"}:
            matches.append(line.strip())
    return matches


def git_revision_command(
    source: Path,
    *,
    effective_uid: int,
    owner_uid: int,
    owner_name: str,
) -> list[str]:
    command = ["git", "-C", str(source), "rev-parse", "HEAD"]
    if effective_uid == owner_uid:
        return command
    if effective_uid == 0:
        return ["runuser", "--user", owner_name, "--", *command]
    raise PermissionError(
        f"audit uid {effective_uid} cannot verify Git repository owned by uid {owner_uid}: {source}"
    )


def git_head_revision(source: Path) -> str:
    """Read HEAD as the retained repository owner.

    The completion audit is launched by a root-owned systemd service while the
    retained source repositories are owned by the unprivileged experiment
    user.  Dropping to that owner preserves Git's ownership protection without
    changing global configuration or declaring any directory globally safe.
    """
    source = source.resolve()
    owner_uid = source.stat().st_uid
    owner_name = pwd.getpwuid(owner_uid).pw_name
    command = git_revision_command(
        source,
        effective_uid=os.geteuid(),
        owner_uid=owner_uid,
        owner_name=owner_name,
    )
    return subprocess.check_output(command, text=True).strip()


def verify_full_artifact_ledger(
    artifact: Path,
    receipt: dict[str, Any],
    preflight_ledger: dict[str, Any],
    revision: str,
    index_sha256: str,
    *,
    verify_content: bool,
) -> dict[str, Any]:
    """Recompute the framed-code, scale, and complete-model byte ledger.

    This deliberately does not trust aggregate fields in the pack receipt.  It
    walks the finalized manifest, requires exact row coverage, checks the exact
    chunk/scale file sets, parses every frame header, and (for a production
    audit) hashes every frame and scale block and verifies each frame CRC.
    """
    try:
        manifest_path = artifact / "codes.cq3d" / "manifest.json"
        manifest = read_json(manifest_path)
        identity = manifest.get("identity")
        receipt_identity = receipt.get("identity")
        if not isinstance(identity, dict) or identity != receipt_identity:
            raise ValueError("full artifact identity differs from pack receipt")
        identity_bytes = (
            json.dumps(identity, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
        ).encode()
        if (
            manifest.get("schema") != "clark_chunked_base3_v1"
            or manifest.get("complete") is not True
            or manifest.get("identity_sha256") != hashlib.sha256(identity_bytes).hexdigest()
            or identity.get("schema") != "qwen38_full_base3_identity_v1"
            or identity.get("model_revision") != revision
            or identity.get("index_sha256") != index_sha256
            or identity.get("quantizer")
            != {"method": "optimal", "group_size": 128, "scale_dtype": "float16"}
        ):
            raise ValueError("full artifact manifest identity is invalid")

        sources = identity.get("mapped_sources")
        chunks = manifest.get("chunks")
        if not isinstance(sources, list) or not sources or not isinstance(chunks, list) or not chunks:
            raise ValueError("full artifact source or chunk inventory is empty")
        source_by_name: dict[str, dict[str, Any]] = {}
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("name"), str):
                raise ValueError("malformed mapped-source entry")
            name = source["name"]
            shape = source.get("shape")
            if name in source_by_name or not isinstance(shape, list) or len(shape) != 2:
                raise ValueError("duplicate or non-matrix mapped source")
            if not all(isinstance(value, int) and value > 0 for value in shape):
                raise ValueError("invalid mapped-source shape")
            if source.get("trits") != shape[0] * shape[1]:
                raise ValueError("mapped-source trit count mismatch")
            source_by_name[name] = source

        expected_row = {name: 0 for name in source_by_name}
        expected_chunk_files: set[str] = set()
        expected_scale_files: set[str] = set()
        trits = payload_bits = payload_bytes = frame_bytes = scale_bytes = scale_count = 0
        header_bytes = 0
        for expected_index, record in enumerate(chunks):
            if not isinstance(record, dict) or record.get("index") != expected_index:
                raise ValueError("chunk records are not a contiguous zero-based sequence")
            if record.get("verified_roundtrip") is not True:
                raise ValueError(f"chunk {expected_index} lacks encoder round-trip evidence")
            descriptor = record.get("descriptor")
            metadata = record.get("frame_metadata")
            if not isinstance(descriptor, dict) or not isinstance(metadata, dict):
                raise ValueError(f"chunk {expected_index} metadata is malformed")
            name = descriptor.get("tensor")
            source = source_by_name.get(name)
            if source is None or descriptor.get("tensor_shape") != source["shape"]:
                raise ValueError(f"chunk {expected_index} tensor identity mismatch")
            if descriptor.get("category") != source.get("category"):
                raise ValueError(f"chunk {expected_index} tensor category mismatch")
            row_start = descriptor.get("row_start")
            row_stop = descriptor.get("row_stop")
            if (
                not isinstance(row_start, int)
                or not isinstance(row_stop, int)
                or row_start != expected_row[name]
                or not row_start < row_stop <= source["shape"][0]
            ):
                raise ValueError(f"chunk {expected_index} row coverage mismatch")
            expected_row[name] = row_stop
            rows = row_stop - row_start
            columns = source["shape"][1]
            record_trits = rows * columns
            expected_scale_shape = [rows, math.ceil(columns / 128)]
            if record.get("trits") != record_trits or metadata.get("trits") != record_trits:
                raise ValueError(f"chunk {expected_index} trit count mismatch")
            if descriptor.get("scale_shape") != expected_scale_shape:
                raise ValueError(f"chunk {expected_index} scale shape mismatch")
            expected_scale_bytes = math.prod(expected_scale_shape) * 2
            if descriptor.get("scale_bytes") != expected_scale_bytes:
                raise ValueError(f"chunk {expected_index} scale-byte mismatch")

            bit_count = metadata.get("payload_bits")
            byte_count = metadata.get("payload_bytes")
            frame_count = metadata.get("frame_bytes")
            per_frame_header = metadata.get("header_bytes")
            if (
                not isinstance(bit_count, int)
                or bit_count <= 0
                or byte_count != math.ceil(bit_count / 8)
                or per_frame_header != BASE3_HEADER.size
                or frame_count != BASE3_HEADER.size + byte_count
            ):
                raise ValueError(f"chunk {expected_index} frame ledger mismatch")
            chunk_name = f"{expected_index:08d}.cq3a"
            chunk_path = artifact / "codes.cq3d" / "chunks" / chunk_name
            expected_chunk_files.add(chunk_name)
            if not chunk_path.is_file() or chunk_path.stat().st_size != frame_count:
                raise ValueError(f"chunk {expected_index} file length mismatch")
            if verify_content:
                content = chunk_path.read_bytes()
                if hashlib.sha256(content).hexdigest() != record.get("frame_sha256"):
                    raise ValueError(f"chunk {expected_index} frame SHA-256 mismatch")
                magic, framed_trits, framed_bits, checksum = BASE3_HEADER.unpack_from(content)
                frame_payload = content[BASE3_HEADER.size :]
                if (
                    magic != b"CQ3A1"
                    or framed_trits != record_trits
                    or framed_bits != bit_count
                    or (zlib.crc32(frame_payload) & 0xFFFFFFFF) != checksum
                    or checksum != metadata.get("crc32")
                ):
                    raise ValueError(f"chunk {expected_index} frame header or CRC mismatch")
                unused_bits = (-bit_count) % 8
                if unused_bits and frame_payload[-1] & ((1 << unused_bits) - 1):
                    raise ValueError(f"chunk {expected_index} has nonzero payload padding")

            scale_name = descriptor.get("scale_file")
            if not isinstance(scale_name, str) or Path(scale_name).name != scale_name:
                raise ValueError(f"chunk {expected_index} scale filename is invalid")
            if scale_name in expected_scale_files:
                raise ValueError(f"chunk {expected_index} reuses a scale filename")
            expected_scale_files.add(scale_name)
            scale_path = artifact / "scales" / scale_name
            if not scale_path.is_file() or scale_path.stat().st_size != expected_scale_bytes:
                raise ValueError(f"chunk {expected_index} scale file length mismatch")
            if verify_content and sha256_file(scale_path) != descriptor.get("scale_sha256"):
                raise ValueError(f"chunk {expected_index} scale SHA-256 mismatch")

            trits += record_trits
            payload_bits += bit_count
            payload_bytes += byte_count
            frame_bytes += frame_count
            header_bytes += per_frame_header
            scale_bytes += expected_scale_bytes
            scale_count += math.prod(expected_scale_shape)

        if any(expected_row[name] != source["shape"][0] for name, source in source_by_name.items()):
            raise ValueError("mapped tensor row coverage is incomplete")
        actual_chunk_files = {path.name for path in (artifact / "codes.cq3d/chunks").iterdir() if path.is_file()}
        actual_scale_files = {path.name for path in (artifact / "scales").iterdir() if path.is_file()}
        if actual_chunk_files != expected_chunk_files or actual_scale_files != expected_scale_files:
            raise ValueError("artifact contains missing or unmanifested chunk/scale files")

        calculated_code_ledger = {
            "chunks": len(chunks),
            "trits": trits,
            "payload_bits": payload_bits,
            "frame_bytes": frame_bytes,
        }
        manifest_ledger = manifest.get("ledger")
        receipt_code = receipt.get("code_ledger", {})
        if manifest_ledger != calculated_code_ledger:
            raise ValueError("manifest aggregate code ledger mismatch")
        if any(receipt_code.get(key) != value for key, value in calculated_code_ledger.items()):
            raise ValueError("pack-receipt aggregate code ledger mismatch")
        if (
            receipt_code.get("payload_bits_per_trit") != payload_bits / trits
            or receipt_code.get("frame_bits_per_trit") != frame_bytes * 8 / trits
            or receipt_code.get("delta_payload_bits_per_trit") != payload_bits / trits - math.log2(3)
        ):
            raise ValueError("pack-receipt derived code ledger mismatch")

        manifest_extra = manifest.get("extra", {})
        if (
            manifest.get("source_sha256") != receipt.get("code_source_sha256")
            or manifest_extra.get("scale_source_sha256") != receipt.get("scale_source_sha256")
            or manifest_extra.get("scale_bytes") != scale_bytes
            or manifest_extra.get("scale_count") != scale_count
            or manifest_extra.get("parallel_independently_verified_trits") != trits
            or preflight_ledger.get("ternary_payload") != trits
            or preflight_ledger.get("fp16_scale_count") != scale_count
        ):
            raise ValueError("manifest source, scale, or preflight ledger mismatch")

        retained_bytes = preflight_ledger.get("retained_high_precision", -1) * 2
        total_parameters = preflight_ledger.get("total")
        complete_payload_bytes = frame_bytes + scale_bytes + retained_bytes
        artifact_storage_bytes = sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())
        artifact_metadata_bytes = artifact_storage_bytes - frame_bytes - scale_bytes
        complete_transport_bytes = artifact_storage_bytes + retained_bytes
        complete = receipt.get("complete_model_ledger", {})
        expected_complete = {
            "ternary_code_frame_bytes": frame_bytes,
            "fp16_scale_bytes": scale_bytes,
            "retained_high_precision_bytes": retained_bytes,
            "complete_payload_bytes": complete_payload_bytes,
            "payload_effective_bits_per_total_parameter": complete_payload_bytes * 8 / total_parameters,
            "artifact_storage_bytes_excluding_retained_tensors": artifact_storage_bytes,
            "artifact_metadata_bytes": artifact_metadata_bytes,
            "complete_transport_bytes_including_referenced_retained_tensors": complete_transport_bytes,
            "transport_effective_bits_per_total_parameter": complete_transport_bytes * 8 / total_parameters,
            "preflight_ideal_payload_bytes": preflight_ledger.get("base3_payload_bytes"),
            "payload_overhead_bytes_vs_preflight": complete_payload_bytes - preflight_ledger.get("base3_payload_bytes"),
            "transport_overhead_bytes_vs_preflight": complete_transport_bytes - preflight_ledger.get("base3_payload_bytes"),
        }
        if any(complete.get(key) != value for key, value in expected_complete.items()):
            raise ValueError("complete-model pack ledger differs from actual artifact files")
        if complete.get("retained_tensor_storage") != "referenced from immutable source checkpoint; bytes included in ledger, not duplicated":
            raise ValueError("retained-tensor storage policy differs from the preregistered pack")
        return {
            "valid": True,
            "content_hashes_verified": verify_content,
            "sources": len(sources),
            "chunks": len(chunks),
            "trits": trits,
            "payload_bits": payload_bits,
            "payload_bytes": payload_bytes,
            "frame_header_bytes": header_bytes,
            "frame_bytes": frame_bytes,
            "scale_count": scale_count,
            "scale_bytes": scale_bytes,
            "retained_bytes": retained_bytes,
            "complete_payload_bytes": complete_payload_bytes,
            "effective_bits_per_total_parameter": complete_payload_bytes * 8 / total_parameters,
            "artifact_storage_bytes": artifact_storage_bytes,
        }
    except Exception as error:
        return {"valid": False, "content_hashes_verified": False, "error": str(error)}


def verify_retained_bundle(bundle: Path, manifest_path: Path, revision: str) -> dict[str, Any]:
    """Independently reopen and hash every retained tensor against its manifest."""
    try:
        import torch
        from safetensors import safe_open

        manifest = read_json(manifest_path)
        entries = manifest.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("retained manifest has no entries")
        by_name = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError("retained manifest entry is malformed")
            if entry["name"] in by_name:
                raise ValueError(f"duplicate retained tensor {entry['name']}")
            by_name[entry["name"]] = entry
        if (
            manifest.get("schema") != "clark_qwen38_retained_tensor_manifest_v1"
            or manifest.get("model_revision") != revision
            or manifest.get("tensor_count") != len(entries)
            or manifest.get("parameters") != sum(int(row.get("parameters", -1)) for row in entries)
            or manifest.get("tensor_bytes") != sum(int(row.get("bytes", -1)) for row in entries)
        ):
            raise ValueError("retained manifest aggregate mismatch")
        with safe_open(bundle, framework="pt", device="cpu") as handle:
            if handle.metadata() != {
                "schema": "clark_qwen38_retained_tensors_v1",
                "model_revision": revision,
            }:
                raise ValueError("retained bundle metadata mismatch")
            if set(handle.keys()) != set(by_name):
                raise ValueError("retained bundle tensor-name set mismatch")
            verified_parameters = verified_bytes = 0
            for name in sorted(by_name):
                tensor = handle.get_tensor(name)
                entry = by_name[name]
                raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
                if (
                    list(tensor.shape) != entry.get("shape")
                    or str(tensor.dtype) != entry.get("dtype")
                    or tensor.numel() != entry.get("parameters")
                    or len(raw) != entry.get("bytes")
                    or hashlib.sha256(raw).hexdigest() != entry.get("sha256")
                    or str(tensor.dtype) != "torch.bfloat16"
                ):
                    raise ValueError(f"retained tensor verification failed for {name}")
                verified_parameters += tensor.numel()
                verified_bytes += len(raw)
        return {
            "valid": True,
            "tensors": len(by_name),
            "parameters": verified_parameters,
            "tensor_bytes": verified_bytes,
        }
    except Exception as error:
        return {"valid": False, "error": str(error)}


def verify_clark_source_provenance(path: Path, workspace: Path) -> dict[str, Any]:
    """Verify retained canonical Clark sources and their transport artifacts."""
    component_keys = (
        "clark_private_product",
        "clark_desktop_foundation",
        "clark_autoresearch",
        "clark_agent_compaction",
    )
    try:
        provenance = read_json(path)
        if provenance.get("schema") != "clark_qwen38_source_provenance_v1":
            raise ValueError("Clark source provenance schema mismatch")
        components = {}
        for key in component_keys:
            row = provenance[key]
            revision = row.get("revision")
            source_value = row.get("remote_path")
            archive_value = row.get("archive_path")
            source = Path(source_value) if isinstance(source_value, str) else Path("/")
            archive = Path(archive_value) if isinstance(archive_value, str) else Path("/")
            archive_hash = sha256_file(archive) if archive.is_file() else None
            valid = (
                isinstance(revision, str)
                and bool(re.fullmatch(r"[0-9a-f]{40}", revision))
                and is_local_workspace_path(source_value, workspace)
                and source.is_dir()
                and is_local_workspace_path(archive_value, workspace)
                and isinstance(row.get("archive_sha256"), str)
                and archive_hash == row.get("archive_sha256")
            )
            bundle_hash = None
            bundle_value = row.get("bundle_path")
            if "bundle_sha256" in row:
                bundle = Path(bundle_value) if isinstance(bundle_value, str) else Path("/")
                bundle_hash = sha256_file(bundle) if bundle.is_file() else None
                valid = (
                    valid
                    and is_local_workspace_path(bundle_value, workspace)
                    and bundle_hash == row.get("bundle_sha256")
                )
            live_revision = None
            if (source / ".git").exists():
                live_revision = git_head_revision(source)
                valid = valid and live_revision == revision
            components[key] = {
                "valid": valid,
                "revision": revision,
                "live_revision": live_revision,
                "source": source_value,
                "archive": archive_value,
                "archive_sha256": archive_hash,
                "bundle": bundle_value,
                "bundle_sha256": bundle_hash,
            }
        return {
            "valid": all(row["valid"] for row in components.values()),
            "version": provenance.get("clark_private_product", {}).get("version"),
            "components": components,
            "provenance": provenance,
        }
    except Exception as error:
        return {"valid": False, "error": str(error)}


def audit(
    *,
    state: Path,
    model_dir: Path,
    checkpoints: Path,
    workspace: Path,
    clark_binary: Path,
    source_provenance: Path,
    verify_large_hashes: bool = True,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> bool:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        return bool(passed)

    try:
        preflight = read_json(state / "qwen38-preflight.json")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        check("preflight", False, str(error))
        return {"schema": "clark_qwen38_completion_audit_v1", "complete": False, "checks": checks}

    revision = preflight.get("download_receipt", {}).get("revision")
    ledger = preflight.get("parameter_ledger", {})
    check("preflight_schema", preflight.get("schema") == "clark_qwen38_27b_preflight_v1", preflight.get("schema"))
    check("pinned_revision", isinstance(revision, str) and len(revision) == 40, revision)
    check("model_path_local", is_local_workspace_path(str(model_dir), workspace), str(model_dir))
    index = model_dir / "model.safetensors.index.json"
    expected_index_hash = preflight.get("download_receipt", {}).get("index_sha256")
    actual_index_hash = sha256_file(index) if index.is_file() else None
    check(
        "model_index_hash",
        isinstance(expected_index_hash, str) and len(expected_index_hash) == 64 and actual_index_hash == expected_index_hash,
        {"expected": expected_index_hash, "actual": actual_index_hash},
    )
    total = ledger.get("total")
    ledger_sum = ledger.get("ternary_payload", 0) + ledger.get("retained_high_precision", 0)
    check("parameter_ledger", isinstance(total, int) and total > 0 and ledger_sum == total, {"total": total, "parts": ledger_sum})

    receipt_paths = {
        "bf16_smoke": state / "qwen38-bf16-smoke.json",
        "reference": state / "qwen38-base3-reference.json",
        "resume": state / "qwen38-base3-resume.json",
        "full_pack": state / "qwen38-base3-full.json",
        "retained_export": state / "qwen38-retained-export.json",
        "quality": state / "qwen38-base3-quality.json",
        "radix5": state / "qwen38-radix5-performance.json",
        "sensitivity": state / "qwen38-layer-sensitivity.json",
        "teacher_cache": state / "qwen38-qat-teacher-cache.json",
        "qat": state / "qwen38-one-layer-qat.json",
        "target_158": state / "qwen38-target-158-reconstruction.json",
    }
    receipts: dict[str, dict[str, Any]] = {}
    for name, path in receipt_paths.items():
        try:
            receipt = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            check(f"{name}_receipt", False, str(error))
            continue
        receipts[name] = receipt
        check(f"{name}_schema", receipt.get("schema") == EXPECTED_SCHEMAS[name], receipt.get("schema"))
        if name not in ("bf16_smoke", "quality", "qat"):
            check(f"{name}_promotion", receipt.get("promotion", {}).get("passed") is True, receipt.get("promotion"))

    bf16 = receipts.get("bf16_smoke", {})
    check(
        "bf16_control",
        bf16.get("dtype") == "torch.bfloat16"
        and isinstance(bf16.get("completion"), str)
        and bool(bf16.get("completion", "").strip())
        and isinstance(bf16.get("new_tokens"), int)
        and bf16.get("new_tokens", 0) > 0,
        {key: bf16.get(key) for key in ("dtype", "new_tokens", "completion")},
    )

    revision_values: dict[str, Any] = {}
    for name, receipt in receipts.items():
        value = receipt.get("model_revision")
        if name in ("resume", "full_pack"):
            value = receipt.get("identity", {}).get("model_revision")
        if name != "bf16_smoke":
            revision_values[name] = value
    check(
        "receipt_revision_chain",
        len(revision_values) == len(receipt_paths) - 1 and all(value == revision for value in revision_values.values()),
        revision_values,
    )

    full = receipts.get("full_pack", {})
    artifact_value = full.get("artifact")
    artifact = Path(artifact_value) if isinstance(artifact_value, str) else Path("/")
    manifest = artifact / "codes.cq3d" / "manifest.json"
    manifest_hash = sha256_file(manifest) if manifest.is_file() else None
    check("full_artifact_local", is_local_workspace_path(artifact_value, workspace), artifact_value)
    expected_manifest_hash = full.get("code_manifest_sha256")
    check(
        "full_manifest_hash",
        isinstance(expected_manifest_hash, str) and len(expected_manifest_hash) == 64 and manifest_hash == expected_manifest_hash,
        {"expected": expected_manifest_hash, "actual": manifest_hash},
    )
    full_ledger = full.get("complete_model_ledger", {})
    effective_bpp = full_ledger.get("payload_effective_bits_per_total_parameter")
    full_artifact_evidence = verify_full_artifact_ledger(
        artifact,
        full,
        ledger,
        revision,
        expected_index_hash,
        verify_content=verify_large_hashes,
    )
    check(
        "full_exact_artifact_and_bit_ledger",
        full_artifact_evidence.get("valid") is True
        and (full_artifact_evidence.get("content_hashes_verified") is True or not verify_large_hashes),
        full_artifact_evidence,
    )
    check(
        "full_bit_ledger",
        isinstance(effective_bpp, (int, float))
        and math.isfinite(effective_bpp)
        and effective_bpp > math.log2(3)
        and full_artifact_evidence.get("effective_bits_per_total_parameter") == effective_bpp,
        {"receipt": effective_bpp, "recomputed": full_artifact_evidence.get("effective_bits_per_total_parameter")},
    )
    code_ledger = full.get("code_ledger", {})
    performance = full.get("performance", {})
    ternary_parameters = ledger.get("ternary_payload")
    check(
        "full_independent_decode_and_counts",
        isinstance(ternary_parameters, int)
        and code_ledger.get("trits") == ternary_parameters
        and performance.get("parallel_independently_verified_trits") == ternary_parameters
        and isinstance(full.get("code_source_sha256"), str)
        and len(full.get("code_source_sha256", "")) == 64
        and isinstance(full.get("scale_source_sha256"), str)
        and len(full.get("scale_source_sha256", "")) == 64,
        {
            "expected_trits": ternary_parameters,
            "packed_trits": code_ledger.get("trits"),
            "independently_decoded_trits": performance.get("parallel_independently_verified_trits"),
        },
    )

    retained = receipts.get("retained_export", {}).get("retained", {})
    retained_paths: dict[str, Path] = {}
    for label, path_key, hash_key in (
        ("retained_bundle", "path", "file_sha256"),
        ("retained_manifest", "manifest", "manifest_sha256"),
    ):
        value = retained.get(path_key)
        path = Path(value) if isinstance(value, str) else Path("/")
        retained_paths[label] = path
        expected = retained.get(hash_key)
        actual = sha256_file(path) if verify_large_hashes and path.is_file() else (expected if path.is_file() else None)
        check(f"{label}_local", is_local_workspace_path(value, workspace), value)
        check(
            f"{label}_hash",
            isinstance(expected, str) and len(expected) == 64 and actual == expected,
            {"expected": expected, "actual": actual},
        )
    retained_evidence = verify_retained_bundle(
        retained_paths.get("retained_bundle", Path("/")),
        retained_paths.get("retained_manifest", Path("/")),
        revision,
    )
    retained_receipt = receipts.get("retained_export", {})
    complete_bundle = retained_receipt.get("complete_parameter_bundle", {})
    check(
        "retained_tensor_content_and_ledger",
        retained_evidence.get("valid") is True
        and retained_evidence.get("tensors") == retained.get("tensor_count")
        and retained_evidence.get("parameters") == retained.get("parameters") == ledger.get("retained_high_precision")
        and retained_evidence.get("tensor_bytes") == retained.get("tensor_bytes") == ledger.get("retained_high_precision", 0) * 2
        and complete_bundle.get("ternary_parameters") == ternary_parameters
        and complete_bundle.get("retained_parameters") == ledger.get("retained_high_precision")
        and complete_bundle.get("total_parameters") == total
        and complete_bundle.get("ternary_code_frame_bytes") == full_ledger.get("ternary_code_frame_bytes")
        and complete_bundle.get("fp16_scale_bytes") == full_ledger.get("fp16_scale_bytes")
        and complete_bundle.get("retained_tensor_bytes") == full_ledger.get("retained_high_precision_bytes")
        and complete_bundle.get("complete_payload_bytes") == full_ledger.get("complete_payload_bytes")
        and complete_bundle.get("effective_bits_per_total_parameter") == effective_bpp,
        {"verification": retained_evidence, "receipt": retained, "complete_bundle": complete_bundle},
    )

    quality_cache = receipts.get("quality", {}).get("teacher_cache", {})
    teacher_cache_value = quality_cache.get("path")
    teacher_cache_path = Path(teacher_cache_value) if isinstance(teacher_cache_value, str) else Path("/")
    teacher_cache_hash = sha256_file(teacher_cache_path) if teacher_cache_path.is_file() else None
    check("heldout_cache_local", is_local_workspace_path(teacher_cache_value, workspace), teacher_cache_value)
    heldout_expected = quality_cache.get("sha256")
    check(
        "heldout_cache_hash",
        isinstance(heldout_expected, str) and len(heldout_expected) == 64 and teacher_cache_hash == heldout_expected,
        {"expected": heldout_expected, "actual": teacher_cache_hash},
    )
    quality = receipts.get("quality", {})
    quality_comparison = quality.get("comparison", {})
    quality_finite_values = (
        quality_comparison.get("perplexity_ratio"),
        quality_comparison.get("mean_teacher_kl"),
        quality.get("bf16", {}).get("likelihood", {}).get("mean_nll"),
        quality.get("bf16", {}).get("likelihood", {}).get("perplexity"),
        quality.get("ternary", {}).get("likelihood", {}).get("mean_nll"),
        quality.get("ternary", {}).get("likelihood", {}).get("perplexity"),
    )
    quality_terminal_valid = (
        isinstance(quality.get("promotion", {}).get("passed"), bool)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in quality_finite_values
        )
        and quality_comparison.get("all_bf16_and_ternary_completions_nonempty") is True
        and quality_comparison.get("bf16_generations_deterministic") is True
        and quality_comparison.get("ternary_generations_deterministic") is True
        and quality_comparison.get("bf16_generation_sha256")
        == quality_comparison.get("bf16_generation_repeat_sha256")
        and quality_comparison.get("ternary_generation_sha256")
        == quality_comparison.get("ternary_generation_repeat_sha256")
    )
    check(
        "quality_terminal_science",
        quality_terminal_valid,
        {
            "promotion": quality.get("promotion"),
            "perplexity_ratio": quality_comparison.get("perplexity_ratio"),
            "mean_teacher_kl": quality_comparison.get("mean_teacher_kl"),
            "deterministic": {
                "bf16": quality_comparison.get("bf16_generations_deterministic"),
                "ternary": quality_comparison.get("ternary_generations_deterministic"),
            },
        },
    )
    check(
        "quality_applied_full_artifact",
        receipts.get("quality", {}).get("applied", {}).get("parameters") == ternary_parameters,
        receipts.get("quality", {}).get("applied"),
    )
    quality_integrity = receipts.get("quality", {}).get("artifact_integrity", {})
    check(
        "quality_artifact_identity",
        quality_integrity.get("valid") is True
        and quality_integrity.get("artifact") == artifact_value
        and quality_integrity.get("manifest_sha256") == full.get("code_manifest_sha256")
        and quality_integrity.get("code_source_sha256") == full.get("code_source_sha256")
        and quality_integrity.get("scale_source_sha256") == full.get("scale_source_sha256")
        and quality_integrity.get("model_revision") == revision
        and quality_integrity.get("index_sha256") == expected_index_hash
        and quality_integrity.get("trits") == ternary_parameters,
        quality_integrity,
    )

    radix = receipts.get("radix5", {})
    radix_sample = radix.get("sample", {})
    radix_format = radix.get("format", {})
    radix_performance = radix.get("performance", {})
    radix_projection = radix.get("full_model_projection", {})
    radix_delta = radix_format.get("delta_bits_per_trit")
    radix_payload_bpt = radix_format.get("payload_bits_per_trit")
    radix_sources = radix_sample.get("sources", [])
    radix_sources_valid = (
        isinstance(radix_sources, list)
        and bool(radix_sources)
        and all(
            isinstance(row, dict)
            and isinstance(row.get("used_trits"), int)
            and row["used_trits"] >= 0
            for row in radix_sources
        )
    )
    radix_used_trits = (
        sum(row["used_trits"] for row in radix_sources)
        if radix_sources_valid else None
    )
    radix_projected_rate = radix_projection.get("effective_bits_per_total_parameter")
    expected_radix_extra = None
    expected_radix_bytes = None
    if isinstance(radix_delta, (int, float)) and isinstance(ternary_parameters, int):
        expected_radix_extra = math.ceil(radix_delta * ternary_parameters / 8)
        base3_bytes = ledger.get("base3_payload_bytes")
        if isinstance(base3_bytes, int):
            expected_radix_bytes = base3_bytes + expected_radix_extra
    check(
        "radix5_exact_performance_and_projection",
        radix_sample.get("trits") == 10_000_000
        and isinstance(radix_sample.get("source_sha256"), str)
        and len(radix_sample.get("source_sha256", "")) == 64
        and radix_sources_valid
        and radix_used_trits == 10_000_000
        and radix_format.get("exact_roundtrip") is True
        and isinstance(radix_payload_bpt, (int, float))
        and math.isfinite(radix_payload_bpt)
        and abs(radix_payload_bpt - 1.6) <= 1e-12
        and isinstance(radix_delta, (int, float))
        and math.isfinite(radix_delta)
        and abs(radix_delta - (1.6 - math.log2(3))) <= 1e-12
        and radix_delta <= 0.016
        and isinstance(radix_performance.get("encode_mtrits_per_second"), (int, float))
        and radix_performance.get("encode_mtrits_per_second") >= 50.0
        and isinstance(radix_performance.get("decode_mtrits_per_second"), (int, float))
        and radix_performance.get("decode_mtrits_per_second") >= 50.0
        and radix_projection.get("extra_bytes_vs_ideal_base3") == expected_radix_extra
        and radix_projection.get("complete_payload_bytes") == expected_radix_bytes
        and isinstance(total, int)
        and isinstance(expected_radix_bytes, int)
        and isinstance(radix_projected_rate, (int, float))
        and math.isfinite(radix_projected_rate)
        and abs(radix_projected_rate - expected_radix_bytes * 8 / total) <= 1e-12,
        {
            "sample": radix_sample,
            "format": radix_format,
            "performance": radix_performance,
            "projection": radix_projection,
            "expected_extra_bytes": expected_radix_extra,
            "expected_complete_bytes": expected_radix_bytes,
        },
    )

    calibration = receipts.get("teacher_cache", {})
    calibration_value = calibration.get("cache")
    calibration_path = Path(calibration_value) if isinstance(calibration_value, str) else Path("/")
    calibration_hash = sha256_file(calibration_path) if calibration_path.is_file() else None
    check("calibration_cache_local", is_local_workspace_path(calibration_value, workspace), calibration_value)
    calibration_expected = calibration.get("cache_sha256")
    check(
        "calibration_cache_hash",
        isinstance(calibration_expected, str) and len(calibration_expected) == 64 and calibration_hash == calibration_expected,
        {"expected": calibration_expected, "actual": calibration_hash},
    )

    complete_checkpoints: list[Path] = []
    incomplete_checkpoints: list[str] = []
    if checkpoints.is_dir():
        for path in checkpoints.iterdir():
            match = CHECKPOINT_NAME.fullmatch(path.name)
            if not match or not path.is_dir() or path.is_symlink():
                if path.name.startswith(".checkpoint-") or path.name.startswith("checkpoint-"):
                    incomplete_checkpoints.append(path.name)
                continue
            try:
                checkpoint = read_json(path / "checkpoint.json")
                state_file = path / checkpoint.get("state_file", "")
                valid = (
                    checkpoint.get("schema") == "clark_qwen38_qat_checkpoint_v1"
                    and checkpoint.get("complete") is True
                    and checkpoint.get("step") == int(match.group(1))
                    and state_file.is_file()
                    and sha256_file(state_file) == checkpoint.get("state_sha256")
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                valid = False
            (complete_checkpoints if valid else incomplete_checkpoints).append(path if valid else path.name)
    check("checkpoint_retention", len(complete_checkpoints) <= 2, [path.name for path in complete_checkpoints])
    check("no_incomplete_checkpoints", not incomplete_checkpoints, incomplete_checkpoints)

    qat = receipts.get("qat", {})
    qat_checkpoint_value = qat.get("checkpoint", {}).get("path")
    qat_checkpoint = Path(qat_checkpoint_value) if isinstance(qat_checkpoint_value, str) else Path("/")
    check("qat_checkpoint_local", is_local_workspace_path(qat_checkpoint_value, workspace), qat_checkpoint_value)
    qat_manifest = {}
    if qat_checkpoint in complete_checkpoints:
        qat_manifest = read_json(qat_checkpoint / "checkpoint.json")
    check(
        "qat_checkpoint_valid",
        qat_checkpoint in complete_checkpoints
        and qat.get("checkpoint", {}).get("state_sha256") == qat_manifest.get("state_sha256"),
        {"complete": [path.name for path in complete_checkpoints], "manifest_sha256": qat_manifest.get("state_sha256"), "receipt_sha256": qat.get("checkpoint", {}).get("state_sha256")},
    )
    check("qat_exact_resume", qat.get("checkpoint", {}).get("exact_resume_smoke") is True, qat.get("checkpoint", {}).get("exact_resume_smoke"))

    qat100 = {}
    qat100_required = qat.get("promotion", {}).get("passed") is True
    qat100_path = state / "qwen38-one-layer-qat-100.json"
    if qat100_required:
        try:
            qat100 = read_json(qat100_path)
            check("qat100_schema", qat100.get("schema") == "clark_qwen38_one_layer_qat_100_v1", qat100.get("schema"))
            check("qat100_revision", qat100.get("model_revision") == revision, qat100.get("model_revision"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            check("qat100_receipt", False, str(error))
    else:
        check("qat100_not_required", not qat100_path.exists(), {"first_rung_promoted": False, "receipt_exists": qat100_path.exists()})
    if qat100:
        terminal_value = qat100.get("checkpoint", {}).get("path")
        terminal = Path(terminal_value) if isinstance(terminal_value, str) else Path("/")
        terminal_manifest = read_json(terminal / "checkpoint.json") if terminal in complete_checkpoints else {}
        check("qat100_checkpoint_local", is_local_workspace_path(terminal_value, workspace), terminal_value)
        check(
            "qat100_executed_and_resumable",
            isinstance(qat100.get("completed_step"), int)
            and qat100.get("completed_step") >= 10
            and terminal in complete_checkpoints
            and qat100.get("checkpoint", {}).get("state_sha256") == terminal_manifest.get("state_sha256")
            and qat100.get("checkpoint", {}).get("exact_resume_smoke") is True,
            {
                "completed_step": qat100.get("completed_step"),
                "terminal": terminal_value,
                "stop_reason": qat100.get("stop_reason"),
            },
        )

    target = receipts.get("target_158", {})
    selected_target = target.get("selected_candidate")
    selected_ledger = target.get("ledgers", {}).get(selected_target, {}) if isinstance(selected_target, str) else {}
    target_rate = selected_ledger.get("complete_effective_bits_per_parameter")
    target_coverage = target.get("coverage", {})
    required_target_coverage = {
        "language",
        "vision",
        "mtp",
        "linear_attention",
        "convolution_retained_control",
        "all_rank_not_2_bfloat16",
    }
    check(
        "target_158_storage_and_coverage",
        isinstance(target_rate, (int, float))
        and math.isfinite(target_rate)
        and target_rate <= 1.6
        and target.get("promotion", {}).get("authorizes_full_pack") is False
        and required_target_coverage.issubset(target_coverage)
        and all(target_coverage.get(name) is True for name in required_target_coverage),
        {"selected_candidate": selected_target, "complete_bpp": target_rate, "coverage": target_coverage},
    )

    clark_source = verify_clark_source_provenance(source_provenance, workspace)
    check("clark_source_provenance", clark_source.get("valid") is True, clark_source)
    try:
        clark = read_json(state / "clark-scientist-update.json")
        binary_hash = sha256_file(clark_binary) if clark_binary.is_file() else None
        live_selftest = json.loads(subprocess.check_output([str(clark_binary), "--self-test"], text=True))
        provenance = clark_source.get("provenance", {})
        expected_revisions = {
            "clark_revision": provenance.get("clark_private_product", {}).get("revision"),
            "clark_desktop_revision": provenance.get("clark_desktop_foundation", {}).get("revision"),
            "autoresearch_revision": provenance.get("clark_autoresearch", {}).get("revision"),
            "compaction_revision": provenance.get("clark_agent_compaction", {}).get("revision"),
        }
        clark_valid = (
            clark.get("schema") == "clark_scientist_quant_update_v1"
            and clark.get("installed_path") == str(clark_binary)
            and all(clark.get(key) == value for key, value in expected_revisions.items())
            and clark.get("canonical_source")
            == str(Path(provenance.get("clark_private_product", {}).get("remote_path", "/")) / "crates/clark-desktop-product/scientist")
            and binary_hash == clark.get("worker_sha256")
            and live_selftest == clark.get("self_test")
            and live_selftest.get("status") == "passed"
            and live_selftest.get("worker") == "clark-code-headless"
            and live_selftest.get("protocol_version") == 1
            and live_selftest.get("specialist_count") == 2
            and live_selftest.get("worker_version") == clark_source.get("version")
            and isinstance(live_selftest.get("catalog_sha256"), str)
            and len(live_selftest.get("catalog_sha256", "")) == 64
        )
        check("clark_worker", clark_valid, {"sha256": binary_hash, "self_test": live_selftest, "revisions": expected_revisions})
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        check("clark_worker", False, str(error))

    transfer_matches: list[str] = []
    try:
        process_text = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True)
        transfer_matches.extend(active_transfer_processes(process_text))
    except (OSError, subprocess.CalledProcessError) as error:
        transfer_matches.append(f"process scan failed: {error}")
    check("no_off_node_transfer_process", not transfer_matches, transfer_matches)

    passed = all(row["passed"] for row in checks)
    qat_outcome = None
    if qat100:
        qat_outcome = "100_step_promoted" if qat100.get("promotion", {}).get("passed") is True else "100_step_valid_nonpromotion"
    elif qat:
        qat_outcome = "ten_step_eligible_for_100" if qat.get("promotion", {}).get("passed") is True else "ten_step_valid_nonpromotion"
    return {
        "schema": "clark_qwen38_completion_audit_v1",
        "complete": passed,
        "model_revision": revision,
        "quality_outcome": (
            "ptq_promoted"
            if quality.get("promotion", {}).get("passed") is True
            else "ptq_valid_nonpromotion"
        ) if quality_terminal_valid else None,
        "qat_outcome": qat_outcome,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=Path("/workspace/quant-state"))
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/Qwen3.8-27B"))
    parser.add_argument("--checkpoints", type=Path, default=Path("/workspace/quant-runs/qwen38-qat/checkpoints"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--clark-binary", type=Path, default=Path("/usr/local/bin/clark-code-headless"))
    parser.add_argument(
        "--source-provenance",
        type=Path,
        default=Path("/workspace/clark-quantize/experiments/qwen38_27b_spot/source_provenance.json"),
    )
    parser.add_argument("--skip-large-hashes", action="store_true", help="Only use during unit tests or diagnosis")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(
        state=args.state,
        model_dir=args.model_dir,
        checkpoints=args.checkpoints,
        workspace=args.workspace,
        clark_binary=args.clark_binary,
        source_provenance=args.source_provenance,
        verify_large_hashes=not args.skip_large_hashes,
    )
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(content)
        os.replace(temporary, args.out)
    print(content, end="")
    raise SystemExit(0 if report["complete"] else 3)


if __name__ == "__main__":
    main()
