from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from labeling2.parsing import extract_json, normalize_payload, validate_subset
    from labeling2.pipeline import MODEL_FIELDS
    from labeling2.target_schema import get_path, set_path
else:
    from .parsing import extract_json, normalize_payload, validate_subset
    from .pipeline import MODEL_FIELDS
    from .target_schema import get_path, set_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def distinct_first_five(value: Any) -> list[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    result: list[str] = []
    for item in value:
        item = item.strip()
        if item and item not in result:
            result.append(item)
    return result[:5]


def repair_row(model: str, row: dict[str, Any]) -> dict[str, Any] | None:
    fields = MODEL_FIELDS[model]
    payload, _ = extract_json(str(row.get("response_text", "")))
    if payload is None:
        return None
    repaired_fields: dict[str, str] = {}
    emphasis_text = "paralinguistic.emphasis.emphasized_text"
    value = get_path(payload, emphasis_text)
    cleaned = distinct_first_five(value)
    if cleaned is not None and cleaned != value:
        set_path(payload, emphasis_text, cleaned)
        repaired_fields[emphasis_text] = "deduplicated_and_truncated_to_five"

    if model == "kimi_audio":
        level_field = "paralinguistic.emphasis.level"
        level = get_path(payload, level_field)
        if level in {"sobbing", "crying", "screaming", "laughter", "cough", "sigh"}:
            set_path(payload, level_field, "unknown")
            repaired_fields[level_field] = f"invalid_event_level_replaced_with_unknown:{level}"

    if not repaired_fields:
        return None
    current_errors = validate_subset(payload, fields)
    normalized = normalize_payload(payload, fields)
    accumulated = copy.deepcopy(row.get("parsed") or {})
    for field in fields:
        if field not in current_errors:
            value = get_path(normalized, field)
            if value is not None:
                set_path(accumulated, field, value)
    unresolved = {
        field: message
        for field, message in current_errors.items()
        if get_path(accumulated, field) is None
    }
    if unresolved:
        return None
    repaired = copy.deepcopy(row)
    repaired.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "parsed": accumulated,
        "field_errors": {},
        "deterministic_repair": repaired_fields,
        "repaired_from_attempt": row.get("attempt"),
    })
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("qwen3_captioner", "kimi_audio"), required=True)
    args = parser.parse_args()
    summary: dict[str, int] = {}
    for model in args.models:
        path = args.run_dir / "raw_predictions" / f"{model}.jsonl"
        latest = {str(row["sample_id"]): row for row in read_jsonl(path) if row.get("sample_id")}
        count = 0
        for row in latest.values():
            if row.get("status") == "success" and not row.get("field_errors"):
                continue
            repaired = repair_row(model, row)
            if repaired is not None:
                append_jsonl(path, repaired)
                count += 1
        summary[model] = count
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
