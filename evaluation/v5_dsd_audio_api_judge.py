#!/usr/bin/env python3
"""Judge existing V5/base DSD WAVs using api/gemini_audio.py REST transport."""
import argparse
import base64
import fcntl
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import requests

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "InstructTTSEval-public/qwen3_voice_design"
sys.path.insert(0, str(PIPELINE))
import judge
import score
from common import atomic_write_json, sha256_json, sha256_file

DEFAULT_RUN = PIPELINE / "runs/v5_midasheng_round000_dsd_seed42"
BASE_RUN = PIPELINE / "runs/base/full_bilingual_seed42"
API_SCRIPT = ROOT.parents[1] / "api/gemini_audio.py"
MODEL = "models/gemini-2.5-pro"
TAG = "evaluation_gemini_2_5_pro_audio_api"


def load_config():
    config = runpy.run_path(str(API_SCRIPT), run_name="dsd_audio_api_config")
    key, endpoint = config["API_KEY"], config["BASE_URL"].rstrip("/")
    if not isinstance(key, str) or not key.strip() or not endpoint.startswith("https://"):
        raise ValueError("Invalid Gemini audio API credentials/endpoint")
    return key.strip(), endpoint


class RestClient:
    def __init__(self, key, endpoint, timeout):
        self.key, self.endpoint, self.timeout = key, endpoint, timeout
        self.session = requests.Session()

    def close(self):
        self.session.close()


def call_rest(client, backend, model, prompt, audio_path):
    if backend != "inline" or model != MODEL:
        raise ValueError("This DSD adapter requires inline audio and the original Gemini-2.5-Pro")
    payload = {
        "contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "audio/wav",
                             "data": base64.b64encode(audio_path.read_bytes()).decode("ascii")}},
        ]}],
        # Preserve the original benchmark judge settings, rather than the
        # demo's generation defaults. In particular, do not change thinking.
        "generationConfig": {"temperature": 0, "responseMimeType": "text/plain"},
        "safetySettings": [{"category": name, "threshold": "BLOCK_NONE"} for name in (
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")],
    }
    response = client.session.post(
        f"{client.endpoint}/models/{model.removeprefix('models/')}:generateContent",
        headers={"x-goog-api-key": client.key, "Content-Type": "application/json"},
        json=payload, timeout=client.timeout)
    if not response.ok:
        # Do not retain untrusted response dumps or authentication data in logs.
        raise RuntimeError(f"Gemini audio API HTTP {response.status_code}")
    data = response.json()
    parts = [p for c in data.get("candidates", []) for p in (c.get("content") or {}).get("parts", [])]
    raw = "\n".join(p["text"].strip() for p in parts
                    if not p.get("thought") and isinstance(p.get("text"), str) and p["text"].strip())
    if not raw:
        raise ValueError("Gemini audio API returned no answer text")
    usage = data.get("usageMetadata") or {}
    names = {"promptTokenCount": "prompt_token_count", "candidatesTokenCount": "candidates_token_count",
             "thoughtsTokenCount": "thoughts_token_count", "totalTokenCount": "total_token_count",
             "cachedContentTokenCount": "cached_content_token_count"}
    return raw, SimpleNamespace(**{v: usage[k] for k, v in names.items() if k in usage})


def invoke(function, arguments):
    previous = sys.argv
    try:
        sys.argv = ["dsd_audio_api", *map(str, arguments)]
        return function()
    finally:
        sys.argv = previous


def base_dsd_manifest(root, destination):
    source = root / "generation_manifest.json"
    manifest = json.loads(source.read_text())
    if manifest.get("generation_identity", {}).get("adapter") is not None:
        raise ValueError("Base evaluation must use a generation manifest without an adapter")
    jobs = [j for j in manifest["items"] if j.get("task") == "DSD"]
    keys = {(j["language"], j["id"], j["task"]) for j in jobs}
    if (len(jobs) != 2000 or len(keys) != 2000
            or any(sum(j["language"] == lang for j in jobs) != 1000 for lang in ("en", "zh"))):
        raise ValueError("Base must contain exactly 1,000 unique DSD items per language")
    for job in jobs:
        if (job.get("status") not in ("generated", "existing")
                or sha256_file(Path(job["audio_path"])) != job.get("audio_sha256")):
            raise ValueError("Missing, failed or changed base DSD audio: " + job["id"])
    identity = {"parent": manifest["generation_identity_sha256"], "tasks": ["DSD"],
                "items_sha256": sha256_json(jobs), "candidate": "base"}
    selected = {"complete": True, "expected_audios": len(jobs), "items": jobs,
                "generation_identity": identity, "generation_identity_sha256": sha256_json(identity)}
    target = destination / "dsd_generation_manifest.json"
    if target.exists() and json.loads(target.read_text()) != selected:
        raise ValueError("Base DSD manifest changed; refusing to reuse judge results")
    atomic_write_json(target, selected)
    return target


def selected_round(root, manifest, candidate, requested):
    if candidate == "base":
        if requested is not None:
            raise ValueError("Base has no training round")
        return None
    plan = json.loads((root / "dsd_plan.json").read_text())
    number = plan["round"]
    if not isinstance(number, int) or number < 0 or (requested is not None and number != requested):
        raise ValueError("Requested round differs from the DSD generation plan")
    if manifest["generation_identity"].get("adapter") != plan["adapter"]:
        raise ValueError("DSD generation adapter differs from the selected round plan")
    return number


def same_api_base_score():
    path = BASE_RUN / TAG / "summary.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if not report.get("complete") or report.get("expected") != 2000:
        return None
    return report["bilingual_macro_average"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "judge", "score"))
    parser.add_argument("--candidate", choices=("v5", "base"), default="v5")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--round-index", type=int, help="Zero based V5 round; defaults to the generation plan")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    os.umask(0)
    default_run = PIPELINE / f"runs/v5_midasheng_round{(args.round_index or 0):03d}_dsd_seed42"
    root = (args.run_dir or (BASE_RUN if args.candidate == "base" else default_run)).resolve()
    destination = root / TAG
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "judge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / "generation_manifest.json"
        if args.candidate == "base":
            manifest_path = base_dsd_manifest(root, destination)
        manifest, jobs = judge.generation_jobs(manifest_path)
        if len(jobs) != 2000 or any(j["task"] != "DSD" for j in jobs):
            raise ValueError("Expected the complete 2,000-item DSD generation manifest")
        round_index = selected_round(root, manifest, args.candidate, args.round_index)
        if args.mode == "check":
            load_config()
            print(f"Config loaded from {API_SCRIPT}; candidate={args.candidate}; model={MODEL}; "
                  f"DSD items={len(jobs)}; workers={args.workers}; no API request sent")
            return 0
        if args.mode == "smoke":
            destination /= "smoke"
            manifest_path = destination / "generation_manifest.json"
            identity = {"parent": manifest["generation_identity_sha256"], "ids": [jobs[0]["id"]]}
            atomic_write_json(manifest_path, {"complete": True, "expected_audios": 1,
                              "generation_identity_sha256": sha256_json(identity), "items": jobs[:1]})
        if args.mode != "score":
            key, endpoint = load_config()
            # Reuse the existing prompt, parser, checkpointing and retry logic;
            # replace only the authentication/HTTP transport in this process.
            judge.configure_judger = lambda: (key, endpoint, str(API_SCRIPT))
            judge.create_client = RestClient
            judge.call_gemini = call_rest
            result = invoke(judge.main, ["--generation-manifest", manifest_path,
                "--output-dir", destination, "--model", MODEL, "--backend", "inline",
                "--workers", 1 if args.mode == "smoke" else args.workers,
                "--attempts", 1 if args.mode == "smoke" else 5,
                "--confirm-paid", "--retry-failed"])
            if result:
                return result
        result = invoke(score.main, ["--generation-manifest", manifest_path,
            "--judge-results", destination / "judge_results.jsonl",
            "--output", destination / "summary.json"])
        report = json.loads((destination / "summary.json").read_text())
        atomic_write_json(destination / "dsd_summary.json", {
            "benchmark": "InstructTTSEval DSD", "candidate": args.candidate,
            "round": round_index, "complete": report["complete"],
            "smoke": args.mode == "smoke", "expected": report["expected"], "scored": report["scored"],
            "metrics": {lang: report["metrics"][lang]["DSD"] for lang in ("en", "zh")},
            "bilingual_percentage": report["bilingual_macro_average"] if args.mode != "smoke" else None,
            "base_bilingual_percentage": same_api_base_score(),
            "base_summary_path": str(BASE_RUN / TAG / "summary.json"),
            "judge_model": MODEL, "usage": report["usage"],
            "note": "API provider changed; original Gemini-2.5-Pro model and judging prompt/settings retained.",
        })
        print(f"[REPORT] {destination / 'dsd_summary.json'}")
        return result


if __name__ == "__main__":
    raise SystemExit(main())
