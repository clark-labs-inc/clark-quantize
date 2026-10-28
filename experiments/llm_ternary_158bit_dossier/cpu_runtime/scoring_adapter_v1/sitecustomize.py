"""Conservatively score one llama.cpp model-originated format failure.

This module is loaded through ``PYTHONPATH`` only by the post-observation
Qwen3.8-27B TQ control continuation.  A precisely matched llama.cpp HTTP 500
raised *after generation* because the output does not satisfy the model's
PEG-native chat grammar is converted into a deterministic assistant sentinel.
The frozen evaluators therefore count the attempted item as incorrect instead
of aborting and silently shrinking the denominator.

Every conversion is appended to a JSONL audit log.  All other HTTP errors are
re-raised with their response bodies preserved.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_STATUS = 500
EXPECTED_CODE = 500
EXPECTED_TYPE = "server_error"
EXPECTED_MESSAGE = (
    "The model produced output that does not match the expected peg-native format"
)
SENTINEL = "[CLARK_SCORER_UPSTREAM_PEG_NATIVE_ERROR_HTTP_500]"
_original_urlopen = urllib.request.urlopen


def _request_descriptor(target: Any) -> dict[str, Any]:
    if isinstance(target, urllib.request.Request):
        data = target.data or b""
        return {
            "url": target.full_url,
            "method": target.get_method(),
            "request_body_bytes": len(data),
            "request_body_sha256": hashlib.sha256(data).hexdigest(),
        }
    return {
        "url": str(target),
        "method": "GET",
        "request_body_bytes": 0,
        "request_body_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _matched_peg_native_error(status: int, body: bytes) -> tuple[bool, dict[str, Any] | None]:
    try:
        payload = json.loads(body)
        detail = payload["error"]
    except (KeyError, TypeError, ValueError, UnicodeDecodeError):
        return False, None
    matched = bool(
        status == EXPECTED_STATUS
        and detail.get("code") == EXPECTED_CODE
        and detail.get("type") == EXPECTED_TYPE
        and detail.get("message") == EXPECTED_MESSAGE
    )
    return matched, payload if matched else None


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _synthetic_response(error_payload: dict[str, Any]) -> io.BytesIO:
    payload = {
        "id": "clark-scored-model-format-error",
        "object": "chat.completion",
        "created": 0,
        "model": "local",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": SENTINEL},
                "finish_reason": "server_format_error",
            }
        ],
        "usage": None,
        "clark_scoring_adapter": {
            "rule": "exact_peg_native_http_500_counts_as_incorrect",
            "upstream_error": error_payload["error"],
        },
    }
    return io.BytesIO(json.dumps(payload, separators=(",", ":")).encode())


def resilient_urlopen(target: Any, *args: Any, **kwargs: Any) -> Any:
    started = time.perf_counter()
    try:
        return _original_urlopen(target, *args, **kwargs)
    except urllib.error.HTTPError as error:
        body = error.read()
        matched, payload = _matched_peg_native_error(error.code, body)
        if not matched or payload is None:
            raise urllib.error.HTTPError(
                error.url,
                error.code,
                error.msg,
                error.headers,
                io.BytesIO(body),
            ) from error

        event_path = os.environ.get("CLARK_PEG_ERROR_EVENTS")
        if not event_path:
            raise RuntimeError(
                "CLARK_PEG_ERROR_EVENTS is required when scoring PEG-native failures"
            ) from error
        event = {
            "schema": "qwen38_27b_peg_native_scoring_event_v1",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            **_request_descriptor(target),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "upstream_http_status": error.code,
            "upstream_error": payload["error"],
            "upstream_error_body_bytes": len(body),
            "upstream_error_body_sha256": hashlib.sha256(body).hexdigest(),
            "scored_as_incorrect": True,
            "replacement_content": SENTINEL,
            "replacement_finish_reason": "server_format_error",
        }
        _append_event(Path(event_path), event)
        return _synthetic_response(payload)


if os.environ.get("CLARK_SCORE_PEG_NATIVE_HTTP_500_AS_INCORRECT") == "1":
    urllib.request.urlopen = resilient_urlopen
