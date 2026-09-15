#!/usr/bin/env python3
"""Compute InstructTTSEval statistics with explicit scored-item coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import LANGUAGES, TASKS, atomic_write_json, load_latest_records, record_key


OFFICIAL_JUDGE_MODEL = "models/gemini-2.5-pro-preview-05-06"
OFFICIAL_PROMPT_SHA256 = "3d4b2a49f86606eb18fab9bc862e4e417d3b23f4a19150ebfd5ffe85208dcd70"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--judge-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dry-run", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--input-price-per-million", type=float, default=1.25)
    parser.add_argument("--output-price-per-million", type=float, default=10.0)
    return parser.parse_args()


def token_count(usage: dict[str, Any] | None, names: tuple[str, ...]) -> int:
    if not usage:
        return 0
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
    return 0


def main() -> int:
    args = parse_args()
    generation = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
    expected_items = [
        item
        for item in generation.get("items", [])
        if item.get("status") in {"generated", "existing"}
    ]
    expected = {record_key(item) for item in expected_items}
    if len(expected) != generation.get("expected_audios"):
        raise RuntimeError("Generation manifest is incomplete or contains duplicate keys")
    latest = load_latest_records(args.judge_results)
    success: dict[tuple[str, str, str], dict[str, Any]] = {}
    failed: set[tuple[str, str, str]] = set()
    for key in expected:
        row = latest.get(key)
        if row is None:
            continue
        if row.get("status") != "success":
            failed.add(key)
            continue
        if row.get("dry_run") and not args.allow_dry_run:
            continue
        if not isinstance(row.get("gemini_score"), bool):
            failed.add(key)
            continue
        success[key] = row
    missing = sorted(expected - success.keys() - failed)
    complete = not missing and not failed and len(success) == len(expected)

    failed_items = []
    for language, sample_id, task in sorted(failed):
        row = latest.get((language, sample_id, task), {})
        detail = {
            "language": language,
            "id": sample_id,
            "task": task,
            "status": row.get("status", "failed"),
        }
        for field in (
            "timestamp_utc",
            "error_type",
            "error",
            "elapsed_seconds",
        ):
            if field in row:
                detail[field] = row[field]
        failed_items.append(detail)

    cells: dict[str, dict[str, Any]] = {}
    for language in LANGUAGES:
        cells[language] = {}
        for task in TASKS:
            keys = sorted(key for key in expected if key[0] == language and key[2] == task)
            valid = [success[key]["gemini_score"] for key in keys if key in success]
            true_count = sum(value is True for value in valid)
            cells[language][task] = {
                "expected": len(keys),
                "valid": len(valid),
                "scored": len(valid),
                "true": true_count,
                "percentage": round(100 * true_count / len(valid), 4) if valid else None,
                "coverage_percentage": (
                    round(100 * len(valid) / len(keys), 4) if keys else None
                ),
            }
        values = [
            cells[language][task]["percentage"]
            for task in TASKS
            if cells[language][task]["percentage"] is not None
        ]
        cells[language]["macro_average"] = (
            round(sum(values) / len(values), 4) if values else None
        )
    all_cell_values = [
        cells[language][task]["percentage"]
        for language in LANGUAGES
        for task in TASKS
        if cells[language][task]["percentage"] is not None
    ]
    input_tokens = sum(
        token_count(
            row.get("usage"),
            ("prompt_token_count", "prompt_tokens", "input_token_count"),
        )
        for row in success.values()
    )
    output_tokens = sum(
        token_count(
            row.get("usage"),
            ("candidates_token_count", "completion_tokens", "output_token_count"),
        )
        for row in success.values()
    )
    # Gemini candidates_token_count excludes thinking; completion_tokens in
    # OpenAI-compatible responses already includes it and must not be doubled.
    thinking_tokens = sum(
        token_count(row.get("usage"), ("thoughts_token_count", "thoughtsTokenCount"))
        for row in success.values()
        if isinstance((row.get("usage") or {}).get("candidates_token_count"), int)
    )
    billable_output_tokens = output_tokens + thinking_tokens
    estimated_cost = (
        input_tokens * args.input_price_per_million
        + billable_output_tokens * args.output_price_per_million
    ) / 1_000_000
    judge_metadata_path = args.judge_results.resolve().parent / "judge_metadata.json"
    judge_metadata: dict[str, Any] = {}
    if judge_metadata_path.is_file():
        judge_metadata = json.loads(judge_metadata_path.read_text(encoding="utf-8"))
    official_protocol = (
        complete
        and bool(success)
        and not any(row.get("dry_run") for row in success.values())
        and {row.get("judge_model") for row in success.values()} == {OFFICIAL_JUDGE_MODEL}
        and {row.get("backend") for row in success.values()} == {"files"}
        and judge_metadata.get("judge_model") == OFFICIAL_JUDGE_MODEL
        and judge_metadata.get("prompt_sha256") == OFFICIAL_PROMPT_SHA256
        and judge_metadata.get("backend") == "files"
        and judge_metadata.get("endpoint") is None
    )
    expected_count = len(expected)
    scored_count = len(success)
    coverage_percentage = (
        round(100 * scored_count / expected_count, 4) if expected_count else None
    )
    summary = {
        "benchmark": "InstructTTSEval",
        "official": official_protocol,
        "official_protocol_requirements": {
            "judge_model": OFFICIAL_JUDGE_MODEL,
            "prompt_sha256": OFFICIAL_PROMPT_SHA256,
            "backend": "files",
            "endpoint": "Google default",
        },
        "dry_run": any(row.get("dry_run") for row in success.values()),
        "complete": complete,
        "expected": expected_count,
        "success": scored_count,
        "scored": scored_count,
        "coverage_percentage": coverage_percentage,
        "score_basis": {
            "policy": "successful_judge_results_only",
            "expected_items": expected_count,
            "scored_items": scored_count,
            "failed_items": len(failed),
            "missing_items": len(missing),
            "coverage_percentage": coverage_percentage,
            "macro_cells_expected": sum(
                cells[language][task]["expected"] > 0
                for language in LANGUAGES for task in TASKS
            ),
            "macro_cells_scored": len(all_cell_values),
        },
        "missing": [list(key) for key in missing],
        "missing_items": [
            {"language": language, "id": sample_id, "task": task}
            for language, sample_id, task in missing
        ],
        "failed": [list(key) for key in sorted(failed)],
        "failed_items": failed_items,
        "metrics": cells,
        "bilingual_macro_average": (
            round(sum(all_cell_values) / len(all_cell_values), 4)
            if all_cell_values
            else None
        ),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "billable_output_tokens": billable_output_tokens,
            "cost_basis": "recorded_successful_responses_including_thinking; excludes unrecorded retries",
            "input_price_per_million_usd": args.input_price_per_million,
            "output_price_per_million_usd": args.output_price_per_million,
            "estimated_cost_usd": round(estimated_cost, 6),
        },
    }
    atomic_write_json(args.output.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not complete and not args.allow_incomplete:
        print("[ERROR] Incomplete results require --allow-incomplete for scoring.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
