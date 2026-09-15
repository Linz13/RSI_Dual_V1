from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from labeling2.target_schema import validate_target
else:
    from .target_schema import validate_target


GENERAL_MODELS = ("gemini", "qwen35", "qwen3_captioner", "kimi_audio", "step_audio_r1_1")
EXPERTS = ("volume", "emotion", "accent_en", "accent_zh", "rate_en", "rate_zh")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in read_jsonl(path) if row.get("sample_id")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed labeling2 run without modifying it.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--section", choices=("all", "experts"), default="all")
    args = parser.parse_args()
    root = args.run_dir.expanduser().resolve()
    expected = args.expected_samples
    failures: list[str] = []
    report: dict[str, Any] = {"run_dir": str(root), "expected_samples": expected, "models": {}, "experts": {}}

    manifest = read_jsonl(root / "state/manifest_snapshot.jsonl")
    gemini = latest_rows(root / "raw_predictions/gemini.jsonl")
    languages = Counter()
    for row in manifest:
        sample_id = str(row.get("sample_id", ""))
        parsed = gemini.get(sample_id, {}).get("parsed") or {}
        language = (parsed.get("semantic_content") or {}).get("language")
        if language not in {"English", "Chinese", "other", "unknown"}:
            hint = str(row.get("language_hint") or "").casefold()
            language = "English" if hint in {"english", "en"} else "Chinese" if hint in {"chinese", "zh", "mandarin", "cantonese"} else "unknown"
        languages[str(language)] += 1
    if len(manifest) != expected:
        failures.append(f"manifest:{len(manifest)}!={expected}")

    for name in GENERAL_MODELS if args.section == "all" else ():
        latest = latest_rows(root / "raw_predictions" / f"{name}.jsonl")
        statuses = Counter(str(row.get("status")) for row in latest.values())
        field_error_rows = sum(bool(row.get("field_errors")) for row in latest.values())
        report["models"][name] = {"latest": len(latest), "statuses": dict(statuses), "field_error_rows": field_error_rows}
        if len(latest) != expected or statuses != Counter({"success": expected}) or field_error_rows:
            failures.append(f"model:{name}")

    language_for_expert = {
        "accent_en": "English", "rate_en": "English",
        "accent_zh": "Chinese", "rate_zh": "Chinese",
    }
    for name in EXPERTS:
        latest = latest_rows(root / "expert_predictions" / f"{name}.jsonl")
        statuses = Counter(str(row.get("status")) for row in latest.values())
        report["experts"][name] = {"latest": len(latest), "statuses": dict(statuses)}
        if name in {"volume", "emotion"}:
            wanted = Counter({"success": expected})
        else:
            matched = languages[language_for_expert[name]]
            wanted = Counter({"success": matched, "skipped": expected - matched})
        if len(latest) != expected or statuses != wanted:
            failures.append(f"expert:{name}")

    if args.section == "all":
        labels = read_jsonl(root / "final/labels.jsonl")
        provenance = read_jsonl(root / "final/provenance.jsonl")
        invalid = [row.get("sample_id") for row in labels if validate_target(row.get("Target_JSON_Schema", {}))]
        report["final"] = {
            "labels": len(labels), "provenance": len(provenance),
            "invalid_schema": len(invalid),
            "review_queue": len(read_jsonl(root / "final/review_queue.jsonl")),
        }
        if len(labels) != expected or len(provenance) != expected or invalid:
            failures.append("final")

    report["ok"] = not failures
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
