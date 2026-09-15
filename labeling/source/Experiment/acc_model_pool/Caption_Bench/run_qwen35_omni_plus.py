"""Run qwen3.5-omni-plus on a Caption_Bench-compatible manifest.

The transport follows ``lzy/api/qwen35_omni_audio.py``: audio is sent through
an OpenAI-compatible streaming chat-completions endpoint and text is
assembled from SSE deltas.  The module also exposes ``request_text`` for the
PVQD/RAVDESS diagnostic adapter.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

from caption_bench_utils import CaptionBenchSample, build_prompt, build_common_parser, run_airbench_experiment

# Reuse the validated audio conversion and payload helpers from the supplied
# standalone example.  Importing them does not run its CLI entry point.
PROJECT_ROOT = Path(os.environ.get("AUDIO_CAPTION_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from lzy.api.qwen35_omni_audio import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_payload,
    chat_completions_url,
    sse_events,
)


MODEL_NAME = DEFAULT_MODEL


class QwenAPIError(RuntimeError):
    """An API response or stream that cannot produce a text answer."""


def resolve_api_key(cli_value: str | None) -> str:
    value = cli_value or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("SHELL_API_KEY")
    if not value or not value.strip():
        raise ValueError("API key is empty. Set QWEN_API_KEY or pass --api-key.")
    return value.strip()


def load_model(args: Any) -> dict[str, Any]:
    api_key = resolve_api_key(args.api_key)
    base_url = str(args.base_url or DEFAULT_BASE_URL)
    return {
        "url": chat_completions_url(base_url),
        "model": str(args.api_model or MODEL_NAME),
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        "timeout": int(args.timeout),
    }


def retryable_error(exc: Exception) -> bool:
    if isinstance(exc, requests.RequestException):
        return True
    text = str(exc).lower()
    return any(token in text for token in ("http 408", "http 409", "http 429", "http 5", "timeout", "connection", "temporar"))


def _delta_text(delta: Any) -> str:
    if isinstance(delta, str):
        return delta
    if isinstance(delta, list):
        parts = []
        for item in delta:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _stream_text(response: requests.Response, metadata: dict[str, Any] | None = None) -> str:
    pieces: list[str] = []
    if metadata is None:
        metadata = {}
    metadata.update(http_status=response.status_code, finish_reasons=[], stream_done=False)
    for event_data in sse_events(response):
        if event_data == "[DONE]":
            metadata["stream_done"] = True
            break
        try:
            event = json.loads(event_data)
        except json.JSONDecodeError as exc:
            raise QwenAPIError(f"invalid SSE JSON: {event_data[:200]!r}") from exc
        error = event.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else error
            raise QwenAPIError(str(message))
        for name in ("model", "id", "usage"):
            if event.get(name) is not None:
                metadata[name] = event[name]
        for choice in event.get("choices") or []:
            if choice.get("finish_reason"):
                metadata["finish_reasons"].append(choice["finish_reason"])
            delta = choice.get("delta") or {}
            text = _delta_text(delta.get("content"))
            if text:
                pieces.append(text)
    result = "".join(pieces).strip()
    if not result:
        raise QwenAPIError("stream contained no text content")
    return result


def request_text(bundle: dict[str, Any], audio_path: Path | str, prompt: str, args: Any) -> str:
    """Send one audio prompt and return the complete text response."""
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Build through the supplied example's exact payload contract, including
    # WAV conversion and the input-audio MIME/format fields.
    payload = build_payload(path, prompt, bundle["model"])
    last_error: Exception | None = None
    for attempt in range(1, int(args.request_retries) + 1):
        response: requests.Response | None = None
        try:
            response = requests.post(
                bundle["url"],
                headers=bundle["headers"],
                json=payload,
                stream=True,
                timeout=bundle["timeout"],
            )
            response.encoding = "utf-8"
            if not response.ok:
                message = f"HTTP {response.status_code}: {response.text[:1000]}"
                raise QwenAPIError(message)
            bundle["response_metadata"] = {}
            return _stream_text(response, bundle["response_metadata"])
        except Exception as exc:
            last_error = exc
            if attempt >= int(args.request_retries) or not retryable_error(exc):
                raise
            time.sleep(min(float(args.retry_backoff_sec) * (2 ** (attempt - 1)), 30.0))
        finally:
            if response is not None:
                response.close()
    raise QwenAPIError(str(last_error or "request failed"))


def infer_one(bundle: dict[str, Any], sample: CaptionBenchSample, args: Any) -> str:
    return request_text(bundle, sample.audio_path, build_prompt(sample, args.prompt_prefix), args)


def main() -> None:
    parser = build_common_parser(MODEL_NAME, description="Run qwen3.5-omni-plus on Caption_Bench.")
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=os.getenv("QWEN_MODEL", MODEL_NAME))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-sec", type=float, default=3.0)
    args = parser.parse_args()
    if args.request_retries < 1 or args.retry_backoff_sec < 0:
        parser.error("--request-retries must be >= 1 and --retry-backoff-sec must be >= 0")
    run_airbench_experiment(
        args=args,
        model_name=MODEL_NAME,
        load_model=load_model,
        infer_one=infer_one,
    )


if __name__ == "__main__":
    main()
