"""Run a VLM judge bakeoff over panel-level pairwise jobs.

This script deliberately defaults to dry-run. Add --execute to spend API calls.
It performs normal and A/B-swapped judgments, normalizes all answers back to the
original candidate orientation, and accepts only stable high-confidence consensus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm_pairwise_judge import judge_pair


DEFAULT_MODELS = ["google/gemini-3.1-flash-lite", "qwen/qwen3-vl-8b-instruct"]
DEFAULT_ADJUDICATOR = "google/gemini-3.5-flash"


def load_jobs(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    jobs = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
            if limit is not None and len(jobs) >= limit:
                break
    return jobs


def parse_models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def flip_winner(winner: str) -> str:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return winner


def normalize_swapped(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a result from swapped A/B images back to original A/B."""
    r = json.loads(json.dumps(result))
    r["winner"] = flip_winner(r["winner"])
    r["candidate_a"], r["candidate_b"] = r["candidate_b"], r["candidate_a"]
    for row in r.get("requirement_scores", []):
        row["candidate_a"], row["candidate_b"] = row["candidate_b"], row["candidate_a"]
        row["evidence_a"], row["evidence_b"] = row["evidence_b"], row["evidence_a"]
    return r


def has_critical_contradiction(result: dict[str, Any]) -> bool:
    winner = result.get("winner")
    if winner == "tie":
        return False
    winner_key = "candidate_a" if winner == "A" else "candidate_b"
    loser_key = "candidate_b" if winner == "A" else "candidate_a"
    winner_failures = result.get(winner_key, {}).get("critical_failures", [])
    loser_failures = result.get(loser_key, {}).get("critical_failures", [])
    return bool(winner_failures and not loser_failures)


def summarize_consensus(
    *,
    normalized_results: list[dict[str, Any]],
    min_confidence: float,
    expected_winner: str | None,
) -> dict[str, Any]:
    if not normalized_results:
        return {"accepted": False, "reason": "no_results"}
    winners = [r.get("winner") for r in normalized_results]
    confidences = [float(r.get("confidence", 0.0)) for r in normalized_results]
    contradictions = [has_critical_contradiction(r) for r in normalized_results]
    stable = len(set(winners)) == 1
    confident = min(confidences) >= min_confidence
    contradiction = any(contradictions)
    expected_ok = expected_winner is None or winners[0] == expected_winner
    accepted = bool(stable and confident and not contradiction and expected_ok)
    reason = "accepted"
    if not stable:
        reason = "winner_disagreement"
    elif not confident:
        reason = "low_confidence"
    elif contradiction:
        reason = "critical_contradiction"
    elif not expected_ok:
        reason = "expected_winner_mismatch"
    return {
        "accepted": accepted,
        "winner": winners[0] if stable else "conflict",
        "confidence": sum(confidences) / len(confidences),
        "min_confidence": min(confidences),
        "reason": reason,
        "votes": winners,
        "critical_contradictions": contradictions,
    }


def run_one_model(
    *,
    job: dict[str, Any],
    model: str,
    cache_dir: str | Path,
    execute: bool,
    no_swaps: bool,
    zdr: bool,
    require_parameters: bool,
) -> list[dict[str, Any]]:
    calls = [
        {
            "swap": False,
            "candidate_a": job["candidate_a"],
            "candidate_b": job["candidate_b"],
        }
    ]
    if not no_swaps:
        calls.append(
            {
                "swap": True,
                "candidate_a": job["candidate_b"],
                "candidate_b": job["candidate_a"],
            }
        )
    out = []
    for call in calls:
        if not execute:
            out.append({"model": model, "swap": call["swap"], "dry_run": True})
            continue
        try:
            wrapped = judge_pair(
                model=model,
                prompt=job["prompt"],
                rubric=job["rubric"],
                reference_path=job["reference"],
                candidate_a_path=call["candidate_a"],
                candidate_b_path=call["candidate_b"],
                cache_dir=cache_dir,
                zdr=zdr,
                require_parameters=require_parameters,
            )
            raw = wrapped["result"]
            normalized = normalize_swapped(raw) if call["swap"] else raw
            out.append(
                {
                    "model": model,
                    "swap": call["swap"],
                    "raw": raw,
                    "normalized": normalized,
                    "usage": wrapped.get("usage"),
                    "cache_id": wrapped.get("id"),
                }
            )
        except Exception as exc:
            out.append(
                {
                    "model": model,
                    "swap": call["swap"],
                    "error": str(exc)[:1200],
                }
            )
    return out


def maybe_adjudicate(
    *,
    job: dict[str, Any],
    current: dict[str, Any],
    adjudicator: str,
    cache_dir: str | Path,
    execute: bool,
    no_swaps: bool,
    zdr: bool,
    require_parameters: bool,
    min_confidence: float,
) -> list[dict[str, Any]]:
    if current.get("accepted") or not adjudicator:
        return []
    return run_one_model(
        job=job,
        model=adjudicator,
        cache_dir=cache_dir,
        execute=execute,
        no_swaps=no_swaps,
        zdr=zdr,
        require_parameters=require_parameters,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--adjudicator", default=DEFAULT_ADJUDICATOR)
    ap.add_argument("--cache-dir", default=".vlm_judge_cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.70)
    ap.add_argument("--execute", action="store_true", help="Actually call OpenRouter. Without this, only print planned calls.")
    ap.add_argument("--no-swaps", action="store_true")
    ap.add_argument("--no-zdr", action="store_true")
    ap.add_argument("--no-require-parameters", action="store_true")
    args = ap.parse_args()

    models = parse_models(args.models)
    jobs = load_jobs(args.jobs, limit=args.limit or None)
    calls_per_job = len(models) * (1 if args.no_swaps else 2)
    max_calls = len(jobs) * calls_per_job
    if args.adjudicator:
        max_calls += len(jobs) * (1 if args.no_swaps else 2)
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "jobs": len(jobs),
                    "models": models,
                    "adjudicator": args.adjudicator,
                    "planned_primary_calls": len(jobs) * calls_per_job,
                    "planned_max_calls_with_adjudication": max_calls,
                    "note": "add --execute to call OpenRouter",
                },
                indent=2,
                sort_keys=True,
            )
        )

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with Path(args.out_jsonl).open("w") as f:
        for job in jobs:
            judgments = []
            for model in models:
                judgments.extend(
                    run_one_model(
                        job=job,
                        model=model,
                        cache_dir=args.cache_dir,
                        execute=args.execute,
                        no_swaps=args.no_swaps,
                        zdr=not args.no_zdr,
                        require_parameters=not args.no_require_parameters,
                    )
                )
            normalized = [j["normalized"] for j in judgments if "normalized" in j]
            consensus = summarize_consensus(
                normalized_results=normalized,
                min_confidence=args.min_confidence,
                expected_winner=job.get("expected_winner"),
            )
            adjudications = maybe_adjudicate(
                job=job,
                current=consensus,
                adjudicator=args.adjudicator,
                cache_dir=args.cache_dir,
                execute=args.execute,
                no_swaps=args.no_swaps,
                zdr=not args.no_zdr,
                require_parameters=not args.no_require_parameters,
                min_confidence=args.min_confidence,
            )
            if adjudications:
                normalized_all = normalized + [j["normalized"] for j in adjudications if "normalized" in j]
                consensus = summarize_consensus(
                    normalized_results=normalized_all,
                    min_confidence=args.min_confidence,
                    expected_winner=job.get("expected_winner"),
                )
            record = {
                "job": job,
                "judgments": judgments,
                "adjudications": adjudications,
                "consensus": consensus,
            }
            summaries.append(record)
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()

    accepted = sum(1 for r in summaries if r["consensus"].get("accepted"))
    print(f"jobs={len(summaries)} accepted={accepted} output={args.out_jsonl}")

    if args.out_md:
        lines = [
            f"jobs={len(summaries)} accepted={accepted} acceptance={accepted / max(len(summaries), 1):.3f}",
            "",
            "| status | winner | confidence | goal | prompt | A | B | reason |",
            "|---|---|---:|---|---|---|---|---|",
        ]
        for record in summaries:
            job = record["job"]
            con = record["consensus"]
            lines.append(
                "| {status} | {winner} | {confidence:.3f} | {goal} | {prompt_id} | `{a}` | `{b}` | {reason} |".format(
                    status="accepted" if con.get("accepted") else "review",
                    winner=con.get("winner", ""),
                    confidence=float(con.get("confidence", 0.0)),
                    goal=job.get("goal", ""),
                    prompt_id=job.get("prompt_id", ""),
                    a=job.get("candidate_a_label", "A"),
                    b=job.get("candidate_b_label", "B"),
                    reason=con.get("reason", ""),
                )
            )
        Path(args.out_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
