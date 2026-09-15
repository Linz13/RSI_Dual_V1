"""Run Gemini 3.1 Pro Preview on a Caption_Bench-compatible manifest.

The request shape intentionally matches ``lzy/api/gemini_audio.py``, which is
the locally validated transport for this relay.  In particular, the default
payload does not add a ``generationConfig`` block.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from airbench_utils import (
    AirBenchSample,
    build_common_parser,
    build_prompt,
    run_airbench_experiment,
)


MODEL_NAME = "gemini-3.1-pro-preview"
DEFAULT_BASE_URL = "https://api.videoimagent.com/v1beta"
DEFAULT_LARGE_AUDIO_THRESHOLD_MB = 20.0


class GeminiHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"API request failed: HTTP {status_code}\n{body}")


def get_mime_type(file_path: Path | str) -> str:
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/m4a",
        ".aac": "audio/aac",
    }.get(Path(file_path).suffix.lower(), "audio/wav")


def _validated_script_key() -> str:
    """Migration copy: credentials come only from CLI/environment."""
    return ""


def resolve_api_key(cli_value: str | None) -> str:
    key = (
        cli_value
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or _validated_script_key()
    )
    if not str(key or "").strip():
        raise ValueError(
            "Gemini API key is empty. Set GEMINI_API_KEY, pass --api-key, "
            "or set OPENAI_API_KEY."
        )
    return str(key).strip()


def retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, requests.RequestException):
        return True
    if isinstance(exc, GeminiHTTPError):
        return exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500
    text = str(exc).casefold()
    return any(
        token in text
        for token in (
            "timeout",
            "timed out",
            "connection",
            "temporar",
            "ssl",
            "eof",
            "no available channel",
        )
    )


def load_model(args: Any) -> dict[str, Any]:
    api_key = resolve_api_key(args.api_key)
    base_url = str(args.base_url).rstrip("/")
    return {
        "headers": {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        "url": f"{base_url}/models/{args.api_model}:generateContent",
        "timeout": args.timeout,
    }


def audio_for_gemini(
    audio_path: Path, threshold_mb: float
) -> tuple[Path, Path | None]:
    threshold_bytes = int(threshold_mb * 1024 * 1024)
    if audio_path.stat().st_size <= threshold_bytes:
        return audio_path, None
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            f"Audio exceeds the {threshold_mb:g} MiB inline threshold and "
            "soundfile is unavailable for temporary FLAC conversion."
        ) from exc
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    handle = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
    converted = Path(handle.name)
    handle.close()
    sf.write(converted, audio, sample_rate, format="FLAC")
    return converted, converted


def extract_text(resp_json: dict[str, Any]) -> str:
    candidates = resp_json.get("candidates") or []
    if not candidates:
        raise ValueError(f"No candidates in response: {resp_json}")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    texts = [
        part["text"].strip()
        for part in parts
        if isinstance(part, dict)
        and isinstance(part.get("text"), str)
        and part["text"].strip()
    ]
    if not texts:
        raise ValueError(f"No text found in response: {resp_json}")
    return "\n".join(texts)


def request_text(
    bundle: dict[str, Any],
    audio_path: Path,
    prompt: str,
    args: Any,
) -> str:
    upload_path, cleanup_path = audio_for_gemini(
        audio_path, args.large_audio_threshold_mb
    )
    try:
        audio_b64 = base64.b64encode(upload_path.read_bytes()).decode("utf-8")
        # Keep this identical to the payload in lzy/api/gemini_audio.py except
        # for the per-sample MIME type and prompt.
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": get_mime_type(upload_path),
                                "data": audio_b64,
                            }
                        },
                    ]
                }
            ]
        }
        errors: list[str] = []
        for attempt in range(1, args.request_retries + 1):
            try:
                response = requests.post(
                    bundle["url"],
                    headers=bundle["headers"],
                    data=json.dumps(payload),
                    timeout=bundle["timeout"],
                )
                if not response.ok:
                    raise GeminiHTTPError(response.status_code, response.text)
                data = response.json()
                bundle["response_metadata"] = {
                    "http_status": response.status_code,
                    "model_version": data.get("modelVersion"),
                    "response_id": data.get("responseId"),
                    "finish_reasons": [c.get("finishReason") for c in data.get("candidates", [])],
                    "usage": data.get("usageMetadata"),
                }
                return extract_text(data).strip()
            except Exception as exc:
                errors.append(repr(exc))
                if attempt >= args.request_retries or not retryable_api_error(exc):
                    if len(errors) == 1:
                        raise
                    raise RuntimeError(
                        f"Gemini request failed after {attempt} attempts: "
                        + " | ".join(errors)
                    ) from exc
                delay = min(
                    args.retry_backoff_sec * (2 ** (attempt - 1)), 30.0
                )
                time.sleep(delay + random.uniform(0.0, min(1.0, delay / 4)))
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)


def infer_one(bundle: dict[str, Any], sample: AirBenchSample, args: Any) -> str:
    return request_text(
        bundle,
        Path(sample.audio_path),
        build_prompt(sample, args.prompt_prefix),
        args,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_common_parser(
        MODEL_NAME,
        description="Run Gemini 3.1 Pro Preview on a Caption-Bench manifest.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GEMINI_API_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=MODEL_NAME)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-sec", type=float, default=3.0)
    parser.add_argument(
        "--large-audio-threshold-mb",
        type=float,
        default=DEFAULT_LARGE_AUDIO_THRESHOLD_MB,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_airbench_experiment(
        args=args,
        model_name=args.api_model,
        load_model=load_model,
        infer_one=infer_one,
    )


if __name__ == "__main__":
    main()
