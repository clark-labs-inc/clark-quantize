"""Streaming/resumable AgentWorldBench judge scoring.

The upstream AgentWorld judge command writes judged rows only after all samples
finish. This wrapper imports the same prompt construction and parser, but
appends one judged JSONL row at a time so large paid judge runs can resume.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from openai import OpenAI


INVALID_VALUE = 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_agentworld_eval(eval_dir: Path) -> Any:
    eval_path = eval_dir / "eval.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"AgentWorld eval.py not found: {eval_path}")
    sys.path.insert(0, str(eval_dir))
    spec = importlib.util.spec_from_file_location("agentworld_eval", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load import spec for {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_key(job: dict[str, Any]) -> str:
    return json.dumps(
        [
            job.get("task", ""),
            job.get("id", ""),
            job.get("turn_idx", ""),
            job.get("total_turns", ""),
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(sample_key(json.loads(line)))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed existing judged row")
    return completed


def judge_one(
    *,
    index: int,
    total: int,
    job: dict[str, Any],
    agentworld_eval: Any,
    judge_system_prompts: dict[str, str],
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
    request_timeout: float,
) -> dict[str, Any]:
    row = dict(job)
    gen = row.get("gen", "")
    if not gen:
        row.update(
            {
                "failed": 1.0,
                "total_score": INVALID_VALUE,
                "error_message": "No model generation",
            }
        )
        return row

    client = OpenAI(
        base_url=judge_base_url,
        api_key=judge_api_key,
        timeout=request_timeout,
    )
    subtask = agentworld_eval.get_subtask(row)
    config = agentworld_eval.TASK_CONFIGS.get(
        subtask,
        agentworld_eval.TASK_CONFIGS["mcp"],
    )
    response_tag = config.get("response_tag", "predicted_observation")
    model_output = agentworld_eval.parse_model_output(gen, response_tag)
    judge_messages = agentworld_eval.build_judge_messages(
        row,
        model_output,
        judge_system_prompts,
    )

    parsed = None
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=judge_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            raw_output = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - match upstream robustness.
            last_error = str(exc)
            logger.warning(
                "[%d/%d] Judge call failed for %s attempt %d/%d: %s",
                index,
                total,
                row.get("id"),
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
            continue

        parsed = agentworld_eval.parse_judge_output(
            raw_output,
            response_tag=config.get("judge_response_tag", "final_evaluation"),
        )
        if parsed["success"]:
            break
        last_error = "Judge parse failed"
        logger.warning(
            "[%d/%d] Judge parse failed for %s attempt %d/%d",
            index,
            total,
            row.get("id"),
            attempt,
            max_retries,
        )
        if attempt < max_retries:
            time.sleep(retry_sleep * attempt)

    if parsed and parsed["success"]:
        scores = parsed.get("scores", {})
        row.update(
            {
                "total_score": parsed.get("total_score", 0.0),
                "format": scores.get("format", 0),
                "factuality": scores.get("factuality", 0),
                "consistency": scores.get("consistency", 0),
                "realism": scores.get("realism", 0),
                "quality": scores.get("quality", 0),
                "failed": 0.0,
                "strengths": parsed.get("strengths", []),
                "weaknesses": parsed.get("weaknesses", []),
                "extracted_output": model_output,
                "judge_raw_output": parsed.get("raw_output", ""),
            }
        )
    else:
        row.update(
            {
                "total_score": INVALID_VALUE,
                "failed": 1.0,
                "error_message": last_error or "Judge scoring failed",
                "extracted_output": model_output,
            }
        )
    return row


def write_progress(
    *,
    progress_path: Path,
    completed_count: int,
    total: int,
    done_this_run: int,
    failures_this_run: int,
    start: float,
) -> None:
    elapsed = max(time.time() - start, 1e-6)
    rate = done_this_run / elapsed
    remaining = total - completed_count
    eta = remaining / rate if rate > 0 else None
    logger.info(
        "[%d/%d] Streaming judge progress; this_run=%d failures=%d eta_sec=%s",
        completed_count,
        total,
        done_this_run,
        failures_this_run,
        f"{eta:.0f}" if eta is not None else "unknown",
    )
    progress_path.write_text(
        json.dumps(
            {
                "completed": completed_count,
                "total": total,
                "done_this_run": done_this_run,
                "failures_this_run": failures_this_run,
                "eta_sec": eta,
                "updated_at": time.time(),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="/workspace/Qwen-AgentWorld/eval", type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--judge-base-url",
        default="https://openrouter.ai/api/v1",
    )
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")

    judge_api_key = (
        args.judge_api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    if judge_api_key == "EMPTY":
        raise RuntimeError("missing judge API key")

    agentworld_eval = load_agentworld_eval(args.eval_dir)
    judge_system_prompts = agentworld_eval.load_judge_system_prompts()
    jobs = load_rows(args.predictions)
    total = len(jobs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "judged.jsonl"
    progress_path = args.output_dir / "judge_progress.json"
    completed = load_completed(out_path)
    logger.info("Loaded %d completed judged rows from %s", len(completed), out_path)

    pending_jobs = [
        (index, job)
        for index, job in enumerate(jobs, start=1)
        if sample_key(job) not in completed
    ]
    done_this_run = 0
    failures_this_run = 0
    start = time.time()

    with out_path.open("a", buffering=1) as out:
        logger.info(
            "Starting streaming judge for %d pending rows with concurrency=%d",
            len(pending_jobs),
            args.concurrency,
        )
        next_job = 0
        in_flight: set[Future[dict[str, Any]]] = set()

        def submit_one(executor: ThreadPoolExecutor) -> None:
            nonlocal next_job
            index, job = pending_jobs[next_job]
            next_job += 1
            in_flight.add(
                executor.submit(
                    judge_one,
                    index=index,
                    total=total,
                    job=job,
                    agentworld_eval=agentworld_eval,
                    judge_system_prompts=judge_system_prompts,
                    judge_base_url=args.judge_base_url,
                    judge_api_key=judge_api_key,
                    judge_model=args.judge_model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    request_timeout=args.request_timeout,
                )
            )

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            while next_job < len(pending_jobs) and len(in_flight) < args.concurrency:
                submit_one(executor)

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed.add(sample_key(row))
                    done_this_run += 1
                    if row.get("failed", 1.0) != 0.0:
                        failures_this_run += 1

                    if len(completed) % 10 == 0 or len(completed) == total:
                        write_progress(
                            progress_path=progress_path,
                            completed_count=len(completed),
                            total=total,
                            done_this_run=done_this_run,
                            failures_this_run=failures_this_run,
                            start=start,
                        )

                    while next_job < len(pending_jobs) and len(in_flight) < args.concurrency:
                        submit_one(executor)

    write_progress(
        progress_path=progress_path,
        completed_count=len(completed),
        total=total,
        done_this_run=done_this_run,
        failures_this_run=failures_this_run,
        start=start,
    )


if __name__ == "__main__":
    main()
