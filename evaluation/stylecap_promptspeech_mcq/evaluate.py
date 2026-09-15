#!/usr/bin/env python3
"""Score complete JSONL predictions for the StyleCap/PromptSpeech MCQ benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import BENCHMARK_JSONL, TASK_ORDER, read_jsonl, write_json


TASK_DISPLAY = {
    "gender": "Gender",
    "pitch": "Pitch",
    "speaking_speed": "Speaking Speed",
    "volume": "Volume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_JSONL)
    parser.add_argument("--output", type=Path, help="Optional JSON summary path")
    return parser.parse_args()


def index_unique(rows: list[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            raise ValueError(f"{source}: row is missing non-empty {key}: {row}")
        if value in indexed:
            duplicates.append(value)
        indexed[value] = row
    if duplicates:
        raise ValueError(f"{source}: duplicate {key} values: {sorted(set(duplicates))[:5]}")
    return indexed


def normalize_prediction(prediction: Any, question: dict[str, Any]) -> str | None:
    if not isinstance(prediction, str):
        return None
    value = prediction.strip()
    if not value:
        return None
    answer = value.upper()
    if answer in question["choices"]:
        return answer
    normalized = value.casefold()
    for option, choice in question["choices"].items():
        if normalized == choice.casefold():
            return option
    return None


def score_predictions(
    benchmark_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    benchmark = index_unique(benchmark_rows, "question_id", Path("benchmark"))
    predictions = index_unique(prediction_rows, "question_id", Path("predictions"))
    expected_ids = set(benchmark)
    predicted_ids = set(predictions)
    missing = sorted(expected_ids - predicted_ids)
    extra = sorted(predicted_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            "prediction/benchmark question_id mismatch: "
            f"missing={len(missing)} {missing[:3]}, extra={len(extra)} {extra[:3]}"
        )

    totals = Counter()
    correct = Counter()
    invalid = Counter()
    for question_id, question in benchmark.items():
        task = question["task"]
        if task not in TASK_ORDER:
            raise ValueError(f"unsupported benchmark task: {task}")
        prediction = predictions[question_id]
        if "prediction" not in prediction:
            raise ValueError(f"prediction row lacks 'prediction': {question_id}")
        normalized = normalize_prediction(prediction["prediction"], question)
        totals[task] += 1
        invalid[task] += int(normalized is None)
        correct[task] += int(normalized == question["answer"])

    task_results = {
        task: {
            "correct": correct[task],
            "total": totals[task],
            "invalid": invalid[task],
            "accuracy": correct[task] / totals[task] if totals[task] else 0.0,
        }
        for task in TASK_ORDER
    }
    macro = sum(task_results[task]["accuracy"] for task in TASK_ORDER) / len(TASK_ORDER)
    return {
        "benchmark_questions": len(benchmark_rows),
        "prediction_rows": len(prediction_rows),
        "tasks": task_results,
        "macro_average_accuracy": macro,
    }


def main() -> None:
    args = parse_args()
    benchmark_path = args.benchmark.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    summary = score_predictions(read_jsonl(benchmark_path), read_jsonl(predictions_path))
    for task in TASK_ORDER:
        values = summary["tasks"][task]
        print(
            f"{TASK_DISPLAY[task]} Accuracy: {values['accuracy'] * 100:.2f}% "
            f"({values['correct']}/{values['total']}, invalid={values['invalid']})"
        )
    print(f"Macro Average Accuracy: {summary['macro_average_accuracy'] * 100:.2f}%")
    if args.output:
        write_json(args.output.expanduser().resolve(), summary)
        print(f"Wrote JSON summary: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
