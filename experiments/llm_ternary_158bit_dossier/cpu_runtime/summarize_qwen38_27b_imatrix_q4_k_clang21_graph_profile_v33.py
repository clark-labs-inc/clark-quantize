#!/usr/bin/env python3
"""Reduce the paired v10 quality-anchor cycle profile without changing samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


GROUP_PATTERNS = {
    "q4_k_verifier": (
        "ggml_gemm_q4_K_8x8_q8_K",
        "ggml_gemv_q4_K_8x8_q8_K",
        "ggml_gemv_q4_K_8x8_q8_K_pair_local_scaled_pair2_tile",
    ),
    "q8_x_q8_tinyblas": ("tinyBLAS_Q0_AVX<block_q8_0, block_q8_0",),
    "q6_k": ("ggml_vec_dot_q6_K_q8_K",),
    "openmp": ("libgomp.so", "GOMP_", "libomp.so", "__kmp"),
    "load_repack": ("repack_q4_K_to_q4_K_8_bl",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_report(path: Path) -> dict:
    text = path.read_text(errors="replace")
    lost_match = re.search(r"Total Lost Samples:\s*([0-9]+)", text)
    sample_match = re.search(
        r"^# Samples:\s*([0-9]+(?:\.[0-9]+)?)([KMG]?)",
        text,
        flags=re.MULTILINE,
    )
    if not lost_match or not sample_match:
        raise RuntimeError(f"missing perf headers in {path}")
    multiplier = {"": 1, "K": 1000, "M": 1_000_000, "G": 1_000_000_000}
    rows = []
    for line in text.splitlines():
        if line.lstrip().startswith("#") or ";" not in line:
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 4 or not parts[0].endswith("%"):
            continue
        try:
            overhead = float(parts[0][:-1]) / 100.0
            samples = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "overhead": overhead,
                "samples": samples,
                "dso": parts[2],
                "symbol": ";".join(parts[3:]),
            }
        )
    if not rows:
        raise RuntimeError(f"no parseable perf rows in {path}")
    groups = {name: 0.0 for name in GROUP_PATTERNS}
    for row in rows:
        haystack = f"{row['dso']} {row['symbol']}"
        for name, patterns in GROUP_PATTERNS.items():
            if any(pattern in haystack for pattern in patterns):
                groups[name] += row["overhead"]
                break
    return {
        "header_samples": int(
            float(sample_match.group(1)) * multiplier[sample_match.group(2)]
        ),
        "lost_samples": int(lost_match.group(1)),
        "parsed_overhead": sum(row["overhead"] for row in rows),
        "groups": groups,
        "top_symbols": sorted(rows, key=lambda row: row["overhead"], reverse=True)[:25],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prereg = json.loads(args.preregistration.read_text())
    validity = prereg["validity_gates"]
    expected = {
        Path(prereg["scope"]["artifact"]): prereg["scope"]["artifact_sha256"],
        Path(prereg["scope"]["parent_binary"]): prereg["scope"]["parent_binary_sha256"],
        Path(prereg["scope"]["candidate_binary"]): prereg["scope"]["candidate_binary_sha256"],
    }
    sealed_lines = (args.state / "sealed-inputs.sha256").read_text().splitlines()
    sealed = {Path(line.split(maxsplit=1)[1]): line.split(maxsplit=1)[0] for line in sealed_lines}
    identity_checks = {
        str(path): sealed.get(path) == digest for path, digest in expected.items()
    }

    frequencies = [
        float(line.split()[1])
        for line in (args.state / "frequency.log").read_text().splitlines()
        if len(line.split()) == 2
    ]
    profiles = {}
    for label in ("parent", "candidate"):
        report = args.state / f"{label}.perf-report.csv"
        bench = args.state / f"{label}.bench.json"
        profile = parse_report(report)
        samples_ns = [int(value) for value in json.loads(bench.read_text())[0]["samples_ns"]]
        profile.update(
            {
                "bench_samples": len(samples_ns),
                "bench_median_ns": statistics.median(samples_ns),
                "bench_mean_ns": statistics.mean(samples_ns),
                "bench_sha256": sha256(bench),
                "report_sha256": sha256(report),
            }
        )
        profiles[label] = profile

    frequency_range = max(frequencies) - min(frequencies)
    checks = {
        "artifact_and_binary_identities_exact": all(identity_checks.values()),
        "both_profiles_have_minimum_samples": all(
            profile["header_samples"] >= int(validity["samples_per_runtime_minimum"])
            for profile in profiles.values()
        ),
        "zero_lost_samples": all(profile["lost_samples"] == 0 for profile in profiles.values()),
        "bench_denominators_exact": all(
            profile["bench_samples"] == int(validity["llama_bench_samples_per_runtime"])
            for profile in profiles.values()
        ),
        "frequency_range_within_limit": frequency_range
        <= float(validity["frequency_range_mhz_maximum"]),
    }
    ratio = profiles["parent"]["bench_median_ns"] / profiles["candidate"]["bench_median_ns"]
    required = float(prereg["analysis"]["required_speedup_from_v23_anchor"])
    candidate_groups = profiles["candidate"]["groups"]
    material = {
        name: share for name, share in candidate_groups.items() if share >= 0.10
    }
    single_group_zero_cost_ceilings = {
        name: 1.0 / (1.0 - share) for name, share in material.items()
    }
    result = {
        "schema": "qwen38_27b_imatrix_q4_k_clang21_graph_profile_result_v33",
        "experiment_id": prereg["experiment_id"],
        "terminal": True,
        "infrastructure_valid": all(checks.values()),
        "passed": all(checks.values()),
        "checks": checks,
        "identity_checks": identity_checks,
        "frequency": {
            "minimum_mhz": min(frequencies),
            "maximum_mhz": max(frequencies),
            "range_mhz": frequency_range,
            "samples": len(frequencies),
        },
        "profiles": profiles,
        "paired_bench_parent_over_candidate": ratio,
        "target_analysis": {
            "v23_conventional_decode_tps": prereg["quality_anchor"]["conventional_decode_tps"],
            "target_tps": prereg["analysis"]["conventional_decode_target_tps"],
            "required_speedup_from_v23_anchor": required,
            "material_candidate_groups": material,
            "single_group_zero_cost_ceilings": single_group_zero_cost_ceilings,
            "single_profiled_group_can_close_gap": any(
                ceiling >= required for ceiling in single_group_zero_cost_ceilings.values()
            ),
        },
        "decision": "select_from_material_candidate_groups",
        "claim_boundary": prereg["claim_boundary"],
        "provenance": {
            "preregistration_sha256": sha256(args.preregistration),
            "runner_output_manifest_sha256": sha256(args.state / "sealed-inputs.sha256"),
            "frequency_sha256": sha256(args.state / "frequency.log"),
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["infrastructure_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
