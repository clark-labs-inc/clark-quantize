"""Streaming/resumable AgentWorldBench inference.

The upstream AgentWorld eval script writes predictions only after all samples
finish. This wrapper preserves the same prompt construction but appends one
JSONL row per completed sample so long Qwen3.5 full-context runs can resume.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from openai import OpenAI


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
                logger.warning("Skipping malformed existing prediction row")
    return completed


def build_messages(job: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = job.get("system_str", "")
    current_prompt = job.get("current_prompt", "")
    if not current_prompt:
        prompts = job.get("prompt", [])
        turn_idx = job.get("turn_idx", 1) - 1
        if isinstance(prompts, list) and turn_idx < len(prompts):
            current_prompt = prompts[turn_idx]

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": current_prompt})
    return messages


def infer_one(
    *,
    index: int,
    total: int,
    job: dict[str, Any],
    model_base_url: str,
    model_api_key: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
    request_timeout: float,
) -> dict[str, Any]:
    client = OpenAI(
        base_url=model_base_url,
        api_key=model_api_key,
        timeout=request_timeout,
    )
    gen = ""
    error_message = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=build_messages(job),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            gen = response.choices[0].message.content or ""
            break
        except Exception as exc:  # noqa: BLE001 - keep parity with upstream eval robustness.
            error_message = str(exc)
            logger.warning(
                "[%d/%d] Inference failed for %s attempt %d/%d: %s",
                index,
                total,
                job.get("id"),
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)

    row = dict(job)
    row["gen"] = gen
    if error_message and not gen:
        row["inference_error"] = error_message
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
        "[%d/%d] Streaming inference progress; this_run=%d failures=%d eta_sec=%s",
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
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-api-key", default="EMPTY")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")

    agentworld_eval = load_agentworld_eval(args.eval_dir)
    jobs = agentworld_eval.load_data(str(args.data_dir))
    total = len(jobs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "predictions.jsonl"
    progress_path = args.output_dir / "stream_progress.json"
    completed = load_completed(out_path)
    logger.info("Loaded %d completed rows from %s", len(completed), out_path)

    done_this_run = 0
    failures_this_run = 0
    start = time.time()

    with out_path.open("a", buffering=1) as out:
        pending_jobs = [
            (index, job)
            for index, job in enumerate(jobs, start=1)
            if sample_key(job) not in completed
        ]
        logger.info(
            "Starting streaming inference for %d pending rows with concurrency=%d",
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
                    infer_one,
                    index=index,
                    total=total,
                    job=job,
                    model_base_url=args.model_base_url,
                    model_api_key=args.model_api_key,
                    model_name=args.model_name,
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
                    if row.get("inference_error") and not row.get("gen"):
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

    logger.info("Predictions saved incrementally to %s", out_path)


if __name__ == "__main__":
    main()
