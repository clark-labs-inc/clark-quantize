#!/usr/bin/env python3
"""Run a quant-only OmniVoiceGraph evaluation against Lexar-backed manifests.

The audio and references stay local on Lexar.  Only the request bytes and JSON
response cross the SSH tunnel to the isolated quant service.  Raw responses
are preserved so the evaluation can be independently re-scored.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import sys
import time
import unicodedata
from pathlib import Path

import aiohttp


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint_hash(payload: dict) -> str | None:
    value = payload.get("checkpoint")
    return str(value) if value else None


def reference_scripts(text: str) -> set[str]:
    scripts = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            scripts.add("Latin")
        elif any(token in name for token in ("CJK", "IDEOGRAPH", "HIRAGANA", "KATAKANA")):
            scripts.add("CJK")
        elif "HANGUL" in name:
            scripts.add("Hangul")
        elif "CYRILLIC" in name:
            scripts.add("Cyrillic")
        elif "THAI" in name:
            scripts.add("Thai")
        else:
            scripts.add(name.split()[0] if name else "Other")
    return scripts


def to_moss_text(payload: dict) -> str:
    speaker_ids: dict[str, str] = {}
    chunks: list[str] = []
    for segment in payload.get("segments", []):
        text = str(segment.get("text", "") or "").strip()
        if not text:
            continue
        speaker = str(segment.get("speaker_hash", "S00"))
        speaker_ids.setdefault(speaker, f"S{len(speaker_ids) + 1:02d}")
        start = float(segment.get("start_s", 0.0))
        end = float(segment.get("end_s", start))
        chunks.append(f"[{start:.3f}][{speaker_ids[speaker]}]{text}[{end:.3f}]")
    return "".join(chunks)


async def evaluate(args: argparse.Namespace) -> None:
    rows = load_rows(args.manifest)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if args.english_only:
        non_english = [
            (str(row["id"]), sorted(reference_scripts(str(row.get("reference", "")))))
            for row in rows
            if reference_scripts(str(row.get("reference", ""))) != {"Latin"}
        ]
        if non_english:
            raise ValueError(
                "--english-only rejected non-Latin reference rows: "
                + json.dumps(non_english[:10], ensure_ascii=False)
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=max(1, args.concurrency))
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results: list[dict] = []
    started = time.time()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def run_one(row: dict) -> None:
            async with semaphore:
                rid = str(row["id"])
                out_path = raw_dir / f"{rid}.json"
                if out_path.exists() and not args.force:
                    payload = json.loads(out_path.read_text())
                    results.append({"id": rid, "status": "reused", "payload": payload})
                    return
                audio_path = Path(row["audio"])
                raw = audio_path.read_bytes()
                form = aiohttp.FormData()
                form.add_field("audio", raw, filename=audio_path.name, content_type="application/octet-stream")
                t0 = time.perf_counter()
                result = {"id": rid, "audio": str(audio_path), "sha256": hashlib.sha256(raw).hexdigest()}
                try:
                    async with session.post(args.api_url, params={"format": "json", "variant": args.variant}, data=form) as response:
                        result["http_status"] = response.status
                        body = await response.text()
                        if response.status != 200:
                            result["error"] = body
                            result["status"] = "failed"
                        else:
                            payload = json.loads(body)
                            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                            result["status"] = "ok"
                            result["payload"] = payload
                except Exception as exc:  # preserve per-row failures and continue
                    result["status"] = "failed"
                    result["error"] = repr(exc)
                result["elapsed_s"] = time.perf_counter() - t0
                results.append(result)

        await asyncio.gather(*(run_one(row) for row in rows))

    results.sort(key=lambda item: item["id"])
    by_id = {row["id"]: row for row in rows}
    metric_rows = []
    checkpoint_values = set()
    for result in results:
        rid = result["id"]
        payload = result.get("payload") or {}
        checkpoint = checkpoint_hash(payload)
        if checkpoint:
            checkpoint_values.add(checkpoint)
        metric_rows.append(
            {
                "id": rid,
                "audio_path": by_id[rid]["audio"],
                "reference_text": by_id[rid]["reference"],
                "prediction_text": to_moss_text(payload) if result["status"] in {"ok", "reused"} else "",
                "status": result["status"],
            }
        )
    (args.out_dir / "request_results.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n"
    )
    (args.out_dir / "metric_rows.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in metric_rows) + "\n"
    )

    sys.path.insert(0, str(args.official_repo))
    metrics_module = importlib.import_module("benchmarks.metrics.transcribe_diarize_metrics")
    metric_input = [
        metrics_module.DiarizationRow(
            sample_id=item["id"],
            audio_path=item["audio_path"],
            reference_text=item["reference_text"],
            prediction_text=item["prediction_text"],
        )
        for item in metric_rows
        if item["status"] in {"ok", "reused"}
    ]
    computed = metrics_module.compute_diarization_metrics(metric_input)
    metric_payload = {
        "schema": "omnivoicegraph.moss_official_metrics_receipt.v1",
        "official_repo": str(args.official_repo),
        "manifest": str(args.manifest),
        "total_rows": len(rows),
        "completed_rows": sum(item["status"] in {"ok", "reused"} for item in results),
        "failed_rows": sum(item["status"] == "failed" for item in results),
        "checkpoint_values": sorted(checkpoint_values),
        "metrics": computed.metrics_percent,
        "sample_metrics": [item.__dict__ if hasattr(item, "__dict__") else {field: getattr(item, field) for field in item.__dataclass_fields__} for item in computed.samples],
        "started_unix": started,
        "finished_unix": time.time(),
        "concurrency": args.concurrency,
        "api_url": args.api_url,
        "variant": args.variant,
        "english_only": args.english_only,
        "language_policy": "all alphabetic reference characters must be Unicode Latin"
        if args.english_only
        else None,
    }
    (args.out_dir / "metrics_official.json").write_text(json.dumps(metric_payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: metric_payload[k] for k in ("total_rows", "completed_rows", "failed_rows", "checkpoint_values", "metrics")}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:18077/transcribe")
    parser.add_argument("--variant", default="best")
    parser.add_argument("--official-repo", type=Path, default=Path("/Users/stan/Documents/git/sglang-omni"))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Fail closed unless every manifest reference is Latin-script English-only.",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
