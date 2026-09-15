"""Check the actual GPT judge on this host before resuming an existing run.

No training implementation or configuration changes. Only synthetic text is sent
during the check; the normal launcher is executed after a successful check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
from unittest.mock import patch
from urllib.parse import urlsplit

import requests

from dual_isl_train.io import atomic_json, load_yaml
from dual_isl_train.labeling import LabelService, demo_credentials


ROOT = Path(__file__).resolve().parents[1]


def safe_text(value, secrets):
    text = str(value)
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, "<REDACTED>")
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "<REDACTED>", text)
    text = re.sub(r"(?i)Bearer\s+[^\s\"'<>]+", "Bearer <REDACTED>", text)
    text = re.sub(r"https?://[^\s\"'<>]+", "<URL>", text)
    return text[:1500]


def http_diagnostic(response, secrets):
    result = {"http_status": response.status_code}
    try:
        body = response.json()
    except ValueError:
        # HTML proxy/WAF pages can contain credentials; do not persist them.
        result["response_format"] = "not_json"
        return result
    if response.status_code == 200:
        if isinstance(body, dict):
            result["returned_model"] = safe_text(body.get("model"), secrets)
            choices = body.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                result["finish_reason"] = safe_text(choices[0].get("finish_reason"), secrets)
        return result
    error = body.get("error", body) if isinstance(body, dict) else {}
    if isinstance(error, dict):
        for key in ("type", "code", "message"):
            if key in error:
                result["error_" + key] = safe_text(error[key], secrets)
    elif isinstance(error, str):
        result["error_message"] = safe_text(error, secrets)
    return result


def check(run_dir):
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    if not (run_dir / "run_state.json").is_file():
        raise ValueError("An existing run_state.json is required for resume")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = run_dir / "diagnostics" / ("gpt_judge_check_" + stamp)
    directory.mkdir(parents=True, exist_ok=False)
    service = LabelService.__new__(LabelService)
    service.cfg = dict(cfg["labeling"])
    service.cache = directory
    service.identity = "synthetic_judge_diagnostic_only"
    service.lock = threading.Lock()
    service.events = directory / "events.jsonl"
    url, key = os.getenv("V5_JUDGE_URL"), os.getenv("V5_JUDGE_KEY")
    if not url or not key:
        demo_url, demo_key = demo_credentials(service.cfg["judge_demo_path"])
        url, key = url or demo_url, key or demo_key
    report = {
        "ok": False, "run_dir": str(run_dir), "endpoint_host": urlsplit(url).hostname,
        "requested_model": "gpt-5.5", "scope": "synthetic text API check; no GPU work",
        "environment_present": {name: bool(os.getenv(name)) for name in (
            "OPENAI_BASE_URL", "V5_JUDGE_URL", "V5_JUDGE_KEY", "HTTP_PROXY", "HTTPS_PROXY",
            "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")},
        "requests": [],
    }
    print(json.dumps({"judge_check_started": True, "report": str(directory / "summary.json"),
                      "endpoint_host": report["endpoint_host"],
                      "environment_present": report["environment_present"]}), flush=True)
    original_post = requests.post

    def traced_post(*args, **kwargs):
        started = time.monotonic()
        try:
            response = original_post(*args, **kwargs)
        except requests.RequestException as exc:
            record = {"error_type": type(exc).__name__}
            raise
        else:
            record = http_diagnostic(response, [key])
            return response
        finally:
            if "record" in locals():
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                report["requests"].append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)

    example = {"paralinguistic": {"prosody": "Steady intonation with an even rhythm.",
                                  "pause": "A brief pause separates the two clauses."}}
    try:
        with patch.object(requests, "post", traced_post):
            scores = service.judge(example, example)
        report["scores"] = scores
        if len(scores) != 2 or set(scores.values()) != {1}:
            raise ValueError("Unexpected self-comparison scores")
        if not report["requests"] or report["requests"][-1].get("returned_model") != "gpt-5.5":
            raise ValueError("Unexpected returned judge model")
        report["ok"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    atomic_json(directory / "summary.json", report)
    print(json.dumps({"judge_check_passed": report["ok"], "report": str(directory / "summary.json")}), flush=True)
    return report["ok"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Do not launch training after the check")
    args = parser.parse_args()
    os.umask(0)
    os.environ["DUALISL_SHARED_WRITABLE"] = "1"
    run_dir = args.run_dir.resolve()
    if not check(run_dir):
        print("Judge check failed; training was not started. See the HTTP diagnostic above.", flush=True)
        return 1
    if not args.check_only:
        print("Judge check passed; resuming the existing run with its saved configuration.", flush=True)
        os.execvp("bash", ["bash", str(ROOT / "scripts/run_v5.sh"), "resume", str(run_dir)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
