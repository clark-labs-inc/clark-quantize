#!/usr/bin/env python3
"""Summarize frozen stages of the calibrated-Q4 plus native-x4 composition."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def mad_fraction(values: List[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values) / center


def load_bench(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"expected one llama-bench row in {path}")
    row = payload[0]
    if int(row["n_prompt"]) != 4 or int(row["n_gen"]) != 0:
        raise ValueError(f"{path} is not the frozen p4/n0 graph")
    samples = [int(value) for value in row["samples_ns"]]
    if len(samples) != 7:
        raise ValueError(f"expected seven samples in {path}, found {len(samples)}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "samples_ns": samples,
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
    }


def frequency_summary(path: Path) -> Dict[str, Any]:
    values = []
    for line in path.read_text().splitlines():
        try:
            values.append(float(line.rsplit(maxsplit=1)[-1]))
        except (ValueError, IndexError):
            continue
    if not values:
        raise ValueError("frequency monitor produced no parseable samples")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sample_count": len(values),
        "minimum_mhz": min(values),
        "maximum_mhz": max(values),
        "range_mhz": max(values) - min(values),
        "median_mhz": statistics.median(values),
    }


def task_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["task"]): row for row in rows}


def load_mtp_summary(root: Path) -> Dict[str, Any]:
    summary = load_json(root / "summary.json")
    if summary.get("terminal") is not True:
        raise ValueError(f"non-terminal MTP receipt at {root}")
    generation = summary["generation"]
    if generation["prompt_count"] != 4 or generation["tokens_per_prompt"] != 96:
        raise ValueError(f"non-frozen MTP denominator at {root}")
    return summary


def mtp_configuration(summary: Dict[str, Any], label: str) -> Dict[str, Any]:
    return next(row for row in summary["configurations"] if row["configuration"] == label)


def token_rows(root: Path, label: str) -> List[List[int]]:
    return [
        load_json(root / label / f"prompt-{index:02d}.json")["response"]["tokens"]
        for index in range(4)
    ]


def token_digests(rows: List[List[int]]) -> List[str]:
    return [
        hashlib.sha256(json.dumps(row, separators=(",", ":")).encode()).hexdigest()
        for row in rows
    ]


def summarize_artifact(args: argparse.Namespace) -> int:
    prereg = load_json(args.preregistration)
    amendment = load_json(args.amendment)
    verification = load_json(args.artifact_verification)
    v10 = load_json(args.v10_receipt)
    v21_graph = load_json(args.v21_graph_receipt)
    v21_system = load_json(args.v21_system_receipt)
    v22 = load_json(args.v22_quality_receipt)
    free_generation = load_mtp_summary(args.free_generation)
    target_rows = token_rows(args.free_generation, "target")
    response_metrics = []
    all_tokens = []
    for index, tokens in enumerate(target_rows):
        receipt = load_json(args.free_generation / "target" / f"prompt-{index:02d}.json")
        content = str(receipt["response"].get("content", ""))
        words = content.split()
        counts = collections.Counter(words)
        max_fraction = max(counts.values()) / len(words) if words else 1.0
        response_metrics.append(
            {
                "prompt_index": index,
                "content_characters": len(content),
                "distinct_whitespace_tokens": len(counts),
                "maximum_single_whitespace_token_fraction": max_fraction,
                "token_sha256": token_digests([tokens])[0],
            }
        )
        all_tokens.extend(tokens)
    unique_token_ids = len(set(all_tokens))
    zero_fraction = sum(token == 0 for token in all_tokens) / len(all_tokens)
    free_rules = amendment["free_generation_operational_definition"]
    free_checks = {
        "denominator_is_4x96": len(target_rows) == 4
        and all(len(row) == 96 for row in target_rows),
        "unique_token_ids_at_least_32": unique_token_ids
        >= int(free_rules["unique_token_ids_minimum"]),
        "zero_token_fraction_below_0_01": zero_fraction
        < float(free_rules["zero_token_fraction_maximum"]),
        "content_length_floor": all(
            row["content_characters"]
            >= int(free_rules["per_response_minimum_content_characters"])
            for row in response_metrics
        ),
        "distinct_whitespace_token_floor": all(
            row["distinct_whitespace_tokens"]
            >= int(free_rules["per_response_minimum_distinct_whitespace_tokens"])
            for row in response_metrics
        ),
        "single_whitespace_token_fraction_below_0_8": all(
            row["maximum_single_whitespace_token_fraction"]
            < float(free_rules["per_response_maximum_single_token_fraction"])
            for row in response_metrics
        ),
    }
    checks = {
        "preregistration_and_amendment_match": prereg.get("experiment_id")
        == amendment.get("experiment_id")
        == "qwen38_27b_imatrix_q4_k_native_x4_composition_v23",
        "v10_retained_rejection_exact": v10.get("terminal") is True
        and v10.get("decision")
        == "reject_graph_gate_but_retain_activation_calibration_quality_signal",
        "v21_graph_property_backend_and_speed_passed": v21_graph.get("passed") is True
        and v21_graph.get("decision") == "advance_to_system_gate"
        and "property_log_sha256" in v21_graph.get("provenance", {})
        and "backend_log_sha256" in v21_graph.get("provenance", {}),
        "v21_terminal_rejection_preserved": v21_system.get("terminal") is True
        and v21_system.get("decision") == "reject_system_gate",
        "v22_same_weights_quality_passed": v22.get("terminal") is True
        and v22.get("passed") is True
        and v22.get("decision") == "quality_qualified_same_weights_kernel",
        "v22_receipt_hash_exact": sha256_file(args.v22_quality_receipt)
        == amendment["receipt_resolution"]["v22_canonical_sha256"],
        "candidate_size_exact": args.candidate_model.stat().st_size
        == int(prereg["artifact_contract"]["candidate_bytes"]),
        "candidate_sha256_exact": sha256_file(args.candidate_model)
        == prereg["artifact_contract"]["candidate_sha256"],
        "stock_sha256_exact": sha256_file(args.stock_model)
        == prereg["scope"]["stock_model_sha256"],
        "imatrix_sha256_exact": sha256_file(args.imatrix)
        == prereg["artifact_contract"]["imatrix_sha256"],
        "artifact_inventory_and_payload_valid": verification.get("terminal") is True
        and verification.get("valid") is True
        and verification.get("selected_tensor_count")
        == prereg["artifact_contract"]["selected_tensor_count"]
        and verification.get("selected_weight_count")
        == prereg["artifact_contract"]["selected_weight_count"]
        and verification.get("candidate_type_counts", {}).get("Q8_0", 0) == 0,
        "free_generation_noncollapsed_and_coarse_coherence": all(free_checks.values()),
    }
    passed = all(checks.values())
    payload = {
        "schema": "qwen38_27b_imatrix_q4_k_native_x4_artifact_result_v23",
        "stage": "prerequisites_artifact_backend_and_free_generation",
        "terminal": not passed,
        "infrastructure_valid": True,
        "passed": passed,
        "checks": checks,
        "free_generation": {
            "unique_token_ids": unique_token_ids,
            "zero_token_fraction": zero_fraction,
            "response_metrics": response_metrics,
            "checks": free_checks,
            "summary_sha256": sha256_file(args.free_generation / "summary.json"),
        },
        "artifact_verification": verification,
        "provenance": {
            "preregistration_sha256": sha256_file(args.preregistration),
            "amendment_sha256": sha256_file(args.amendment),
            "v10_receipt_sha256": sha256_file(args.v10_receipt),
            "v21_graph_receipt_sha256": sha256_file(args.v21_graph_receipt),
            "v21_system_receipt_sha256": sha256_file(args.v21_system_receipt),
            "v22_quality_receipt_sha256": sha256_file(args.v22_quality_receipt),
            "verification_sha256": sha256_file(args.artifact_verification),
            "candidate_server_sha256": sha256_file(args.candidate_server),
            "quantizer_sha256": sha256_file(args.quantizer),
        },
        "decision": "advance_to_three_arm_graph" if passed else "reject_artifact_gate",
        "claim_boundary": "Artifact integrity and noncollapse only; no speed, capability, or breakthrough claim.",
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "artifact_gate", "passed": passed}, sort_keys=True))
    return 0 if passed else 2


def summarize_graph(args: argparse.Namespace) -> int:
    prereg = load_json(args.preregistration)
    artifact = load_json(args.artifact_decision)
    if artifact.get("passed") is not True:
        raise ValueError("graph stage requires a passed artifact gate")
    arms = {"A": [], "B": [], "C": []}
    for rotation in range(1, 7):
        for arm in arms:
            path = args.state / f"graph-rotation-{rotation:02d}-{arm}.json"
            arms[arm].append(load_bench(path))
    medians = {arm: [row["median_ns"] for row in rows] for arm, rows in arms.items()}
    b_over_a = [left / right for left, right in zip(medians["A"], medians["B"])]
    c_over_b = [left / right for left, right in zip(medians["B"], medians["C"])]
    c_over_a = [left / right for left, right in zip(medians["A"], medians["C"])]
    frequency = frequency_summary(args.frequency_log)
    gate = prereg["sequential_gates"]["gate_1_three_arm_four_row_graph"]
    checks = {
        "artifact_gate_passed": True,
        "b_over_a_at_least_1_08": statistics.median(b_over_a)
        >= float(gate["minimum_b_over_a_speedup"]),
        "c_over_b_at_least_1_12": statistics.median(c_over_b)
        >= float(gate["minimum_c_over_b_speedup"]),
        "c_over_a_at_least_1_25": statistics.median(c_over_a)
        >= float(gate["minimum_c_over_a_speedup"]),
        "arm_c_process_median_mad_at_most_0_03": mad_fraction(medians["C"])
        <= float(gate["maximum_arm_c_process_median_mad_fraction"]),
        "frequency_range_at_most_100_mhz": frequency["range_mhz"]
        <= float(gate["maximum_frequency_range_mhz"]),
    }
    passed = all(checks.values())
    payload = {
        "schema": "qwen38_27b_imatrix_q4_k_native_x4_graph_result_v23",
        "stage": "three_arm_four_row_graph",
        "terminal": False,
        "infrastructure_valid": True,
        "passed": passed,
        "denominator": {
            "arms": 3,
            "unscored_warmup_processes_per_arm": 1,
            "latin_square_rotations": 6,
            "samples_per_process": 7,
            "samples_per_arm": 42,
        },
        "arms": arms,
        "ratios": {
            "b_over_a_by_rotation": b_over_a,
            "b_over_a_median": statistics.median(b_over_a),
            "c_over_b_by_rotation": c_over_b,
            "c_over_b_median": statistics.median(c_over_b),
            "c_over_a_by_rotation": c_over_a,
            "c_over_a_median": statistics.median(c_over_a),
        },
        "arm_c_process_median_mad_fraction": mad_fraction(medians["C"]),
        "frequency": frequency,
        "checks": checks,
        "decision": "advance_to_paired_serving" if passed else "reject_graph_gate",
        "next_stage": "paired target and MTP3 serving" if passed else "always-run quick quality",
        "claim_boundary": "Three-arm p4 graph on four Milan cores only; not serving, quality, or a breakthrough result.",
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "graph_gate", "passed": passed, "ratios": payload["ratios"]}, sort_keys=True))
    return 0 if passed else 2


def summarize_system(args: argparse.Namespace) -> int:
    prereg = load_json(args.preregistration)
    graph = load_json(args.graph_decision)
    if graph.get("passed") is not True:
        raise ValueError("system stage requires a passed graph gate")
    control_summary = load_mtp_summary(args.control_mtp)
    candidate_summary = load_mtp_summary(args.candidate_mtp)
    control_target = mtp_configuration(control_summary, "target")
    candidate_target = mtp_configuration(candidate_summary, "target")
    control_mtp3 = mtp_configuration(control_summary, "mtp3")
    candidate_mtp3 = mtp_configuration(candidate_summary, "mtp3")
    candidate_target_rows = token_rows(args.candidate_mtp, "target")
    candidate_mtp3_rows = token_rows(args.candidate_mtp, "mtp3")
    target_ratio = (
        candidate_target["accepted_tokens_per_second_weighted"]
        / control_target["accepted_tokens_per_second_weighted"]
    )
    mtp3_ratio = (
        candidate_mtp3["accepted_tokens_per_second_weighted"]
        / control_mtp3["accepted_tokens_per_second_weighted"]
    )
    all_candidate_tokens = [token for row in candidate_mtp3_rows for token in row]
    zero_fraction = sum(token == 0 for token in all_candidate_tokens) / len(all_candidate_tokens)
    gate = prereg["sequential_gates"]["gate_2_paired_serving"]
    checks = {
        "graph_gate_passed": True,
        "target_candidate_over_control_at_least_1_12": target_ratio
        >= float(gate["minimum_candidate_over_control_target_tps"]),
        "mtp3_candidate_over_control_at_least_1_20": mtp3_ratio
        >= float(gate["minimum_candidate_over_control_mtp3_accepted_tps"]),
        "candidate_mtp3_at_least_3_75_tps": candidate_mtp3[
            "accepted_tokens_per_second_weighted"
        ]
        >= float(gate["minimum_candidate_mtp3_accepted_tps"]),
        "candidate_mtp3_matches_own_target_tokens": candidate_mtp3_rows
        == candidate_target_rows,
        "candidate_target_denominator_is_4x96": candidate_target[
            "output_tokens_including_prompt_eval_token"
        ]
        == 384,
        "candidate_mtp3_denominator_is_4x96": candidate_mtp3[
            "output_tokens_including_prompt_eval_token"
        ]
        == 384,
        "candidate_mtp3_zero_fraction_below_0_01": zero_fraction
        < float(gate["maximum_zero_token_fraction"]),
    }
    passed = all(checks.values())
    payload = {
        "schema": "qwen38_27b_imatrix_q4_k_native_x4_system_result_v23",
        "stage": "paired_target_and_mtp3_serving",
        "terminal": False,
        "infrastructure_valid": True,
        "passed": passed,
        "denominator": {"prompts_per_system": 4, "tokens_per_prompt": 96},
        "target": {
            "control_accepted_tps": control_target["accepted_tokens_per_second_weighted"],
            "candidate_accepted_tps": candidate_target["accepted_tokens_per_second_weighted"],
            "candidate_over_control": target_ratio,
        },
        "mtp3": {
            "control_accepted_tps": control_mtp3["accepted_tokens_per_second_weighted"],
            "candidate_accepted_tps": candidate_mtp3["accepted_tokens_per_second_weighted"],
            "candidate_over_control": mtp3_ratio,
            "candidate_exact_parity_with_own_target": candidate_mtp3_rows
            == candidate_target_rows,
            "candidate_zero_token_fraction": zero_fraction,
            "control_draft_n": control_mtp3["draft_n"],
            "control_draft_n_accepted": control_mtp3["draft_n_accepted"],
            "candidate_draft_n": candidate_mtp3["draft_n"],
            "candidate_draft_n_accepted": candidate_mtp3["draft_n_accepted"],
        },
        "checks": checks,
        "decision": "advance_to_quality_adjudication" if passed else "reject_serving_gate",
        "provenance": {
            "graph_decision_sha256": sha256_file(args.graph_decision),
            "control_summary_sha256": sha256_file(args.control_mtp / "summary.json"),
            "candidate_summary_sha256": sha256_file(args.candidate_mtp / "summary.json"),
            "candidate_target_token_sha256": token_digests(candidate_target_rows),
            "candidate_mtp3_token_sha256": token_digests(candidate_mtp3_rows),
            "control_server_sha256": sha256_file(args.control_server),
            "candidate_server_sha256": sha256_file(args.candidate_server),
        },
        "claim_boundary": "Four-prompt paired serving result only; not broad capability, final-node, or a breakthrough claim.",
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "system_gate", "passed": passed, "target": payload["target"], "mtp3": payload["mtp3"]}, sort_keys=True))
    return 0 if passed else 2


def summarize_quick_quality(args: argparse.Namespace) -> int:
    prereg = load_json(args.preregistration)
    baseline = load_json(args.baseline)
    cute = load_json(args.cute)
    math_code = load_json(args.math_code)
    wiki = load_json(args.wikitext)
    model_sha = prereg["artifact_contract"]["candidate_sha256"]
    cute_rows = task_rows(cute["summaries"])
    math_rows = task_rows(math_code["summaries"])
    observed = {
        "cute_correct": int(cute_rows["cute_quick_20"]["correct"]),
        "cute_n": int(cute_rows["cute_quick_20"]["n"]),
        "math500_correct": int(math_rows["math500"]["correct"]),
        "math500_n": int(math_rows["math500"]["n"]),
        "code_trace_correct": int(math_rows["code_trace"]["correct"]),
        "code_trace_n": int(math_rows["code_trace"]["n"]),
        "wikitext_ppl": float(wiki["final_ppl"]),
    }
    observed["wikitext_ppl_ratio_vs_stock"] = observed["wikitext_ppl"] / float(
        baseline["wikitext_ppl"]
    )
    gate = prereg["sequential_gates"]["gate_3_always_run_quick_quality_after_valid_artifact"]
    receipt_checks = {
        "cute_receipt_valid_20": cute.get("schema") == "cpu_cute_quick_result_v1"
        and cute.get("model_sha256") == model_sha
        and len(cute.get("records", [])) == 20
        and observed["cute_n"] == 20,
        "math_code_receipt_valid_8_plus_6": math_code.get("schema")
        == "cpu_math_code_canary_result_v1"
        and math_code.get("model_sha256") == model_sha
        and len(math_code.get("records", [])) == 14
        and observed["math500_n"] == 8
        and observed["code_trace_n"] == 6,
        "wikitext_receipt_valid_4_chunks": wiki.get("schema")
        == "cpu_wikitext_quick_result_v1"
        and wiki.get("valid") is True
        and wiki.get("status") == "completed"
        and wiki.get("model_sha256") == model_sha
        and len(wiki.get("chunks", [])) == 4
        and wiki.get("parameters", {}).get("context") == 512,
    }
    quality_checks = {
        "cute_at_least_9_of_20": observed["cute_correct"]
        >= int(gate["cute_quick_minimum_correct"]),
        "math500_at_least_5_of_8": observed["math500_correct"]
        >= int(gate["math500_minimum_correct"]),
        "code_trace_at_least_5_of_6": observed["code_trace_correct"]
        >= int(gate["code_trace_minimum_correct"]),
        "wikitext_ratio_at_most_1_01": observed["wikitext_ppl_ratio_vs_stock"]
        <= float(gate["maximum_wikitext_ppl_ratio_vs_stock"]),
    }
    passed = all(receipt_checks.values()) and all(quality_checks.values())
    payload = {
        "schema": "qwen38_27b_imatrix_q4_k_native_x4_quick_quality_result_v23",
        "stage": "always_run_quick_quality",
        "terminal": False,
        "infrastructure_valid": all(receipt_checks.values()),
        "passed": passed,
        "denominators": {"cute": 20, "math500": 8, "code_trace": 6, "wikitext_chunks": 4},
        "observed": observed,
        "receipt_checks": receipt_checks,
        "quality_checks": quality_checks,
        "decision": "quick_quality_passed" if passed else "reject_quick_quality_gate",
        "provenance": {
            "baseline_sha256": sha256_file(args.baseline),
            "cute_sha256": sha256_file(args.cute),
            "math_code_sha256": sha256_file(args.math_code),
            "wikitext_sha256": sha256_file(args.wikitext),
        },
        "claim_boundary": "Frozen quick capability tripwire only; not broad near-lossless quality.",
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "quick_quality", "passed": passed, "observed": observed}, sort_keys=True))
    return 0 if passed else 2


def summarize_final(args: argparse.Namespace) -> int:
    prereg = load_json(args.preregistration)
    artifact = load_json(args.artifact_decision)
    graph = load_json(args.graph_decision)
    quick = load_json(args.quick_quality_decision)
    system = load_json(args.system_decision) if args.system_decision else None
    prior_checks = {
        "artifact_gate_passed": artifact.get("passed") is True,
        "graph_gate_passed": graph.get("passed") is True,
        "serving_gate_passed": system is not None and system.get("passed") is True,
        "quick_quality_gate_passed": quick.get("passed") is True,
    }
    prior_passed = all(prior_checks.values())
    mmlu_payload = None
    mmlu_checks = None
    if args.mmlu:
        mmlu = load_json(args.mmlu)
        gate = prereg["sequential_gates"]["gate_4_conditional_mmlu"]
        correct = int(mmlu["summary"]["correct"])
        n = int(mmlu["summary"]["n"])
        baseline = load_json(args.baseline)
        loss = int(baseline["mmlu"]["correct"]) - correct
        mmlu_payload = {"correct": correct, "n": n, "accuracy": float(mmlu["summary"]["accuracy"]), "loss_vs_stock_correct": loss}
        mmlu_checks = {
            "receipt_valid_64": mmlu.get("valid") is True
            and mmlu.get("status") == "completed"
            and mmlu.get("contract", {}).get("model_sha256")
            == prereg["artifact_contract"]["candidate_sha256"]
            and n == int(gate["denominator"]),
            "correct_at_least_51": correct >= int(gate["minimum_correct"]),
            "loss_vs_stock_at_most_1": loss <= int(gate["maximum_loss_vs_stock_correct"]),
        }
    if not prior_passed:
        terminal = True
        passed = False
        decision = "reject_composition_before_mmlu"
        exit_code = 2
    elif args.mmlu is None:
        terminal = False
        passed = False
        decision = "advance_to_mmlu_64"
        exit_code = 0
    else:
        terminal = True
        passed = all(mmlu_checks.values())
        decision = "promote_quality_qualified_cpu_composition" if passed else "reject_mmlu_gate"
        exit_code = 0 if passed else 2
    provenance = {
        "artifact_decision_sha256": sha256_file(args.artifact_decision),
        "graph_decision_sha256": sha256_file(args.graph_decision),
        "quick_quality_decision_sha256": sha256_file(args.quick_quality_decision),
    }
    if args.system_decision:
        provenance["system_decision_sha256"] = sha256_file(args.system_decision)
    if args.mmlu:
        provenance["mmlu_sha256"] = sha256_file(args.mmlu)
    payload = {
        "schema": "qwen38_27b_imatrix_q4_k_native_x4_composition_result_v23",
        "experiment_id": prereg["experiment_id"],
        "terminal": terminal,
        "infrastructure_valid": artifact.get("infrastructure_valid") is True
        and graph.get("infrastructure_valid") is True
        and quick.get("infrastructure_valid") is True
        and (system is None or system.get("infrastructure_valid") is True),
        "passed": passed,
        "decision": decision,
        "prior_checks": prior_checks,
        "graph_ratios": graph.get("ratios"),
        "serving": ({"target": system["target"], "mtp3": system["mtp3"]} if system else {"ran": False, "reason": "graph gate failed"}),
        "quick_quality": quick.get("observed"),
        "mmlu": mmlu_payload,
        "mmlu_checks": mmlu_checks,
        "provenance": provenance,
        "claim_boundary": (
            "A pass establishes the preregistered four-core Milan composition only. "
            "It does not meet the 2.75 tok/s conventional-decode breakthrough gate, "
            "establish final-node scaling, reach 100 tok/s, or support novelty claims."
        ),
    }
    atomic_json(args.out, payload)
    print(json.dumps({"event": "composition_final" if terminal else "mmlu_admitted", "decision": decision, "passed": passed}, sort_keys=True))
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("artifact", "graph", "system", "quick-quality", "final"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--artifact-verification", type=Path)
    parser.add_argument("--artifact-decision", type=Path)
    parser.add_argument("--v10-receipt", type=Path)
    parser.add_argument("--v21-graph-receipt", type=Path)
    parser.add_argument("--v21-system-receipt", type=Path)
    parser.add_argument("--v22-quality-receipt", type=Path)
    parser.add_argument("--free-generation", type=Path)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--stock-model", type=Path)
    parser.add_argument("--imatrix", type=Path)
    parser.add_argument("--candidate-server", type=Path)
    parser.add_argument("--control-server", type=Path)
    parser.add_argument("--quantizer", type=Path)
    parser.add_argument("--frequency-log", type=Path)
    parser.add_argument("--graph-decision", type=Path)
    parser.add_argument("--control-mtp", type=Path)
    parser.add_argument("--candidate-mtp", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cute", type=Path)
    parser.add_argument("--math-code", type=Path)
    parser.add_argument("--wikitext", type=Path)
    parser.add_argument("--quick-quality-decision", type=Path)
    parser.add_argument("--system-decision", type=Path)
    parser.add_argument("--mmlu", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "artifact":
        return summarize_artifact(args)
    if args.stage == "graph":
        return summarize_graph(args)
    if args.stage == "system":
        return summarize_system(args)
    if args.stage == "quick-quality":
        return summarize_quick_quality(args)
    return summarize_final(args)


if __name__ == "__main__":
    raise SystemExit(main())
