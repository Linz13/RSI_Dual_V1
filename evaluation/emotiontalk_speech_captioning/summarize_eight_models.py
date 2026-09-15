#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import TASKS, write_json


MODEL_NAMES = (
    "qwen_base", "qwen8_round0", "qwen8_round1", "qwen8_round2",
    "midasheng_base", "midasheng4_round0", "midasheng4_round1",
    "midasheng4_round2",
)
METRIC_VALUE_KEYS = {
    "bleu_4": "bleu_4",
    "rouge_l": "rouge_l",
    "meteor": "meteor",
    "spider": "spider",
    "fense": "fense",
    "bertscore": "bert_score.f1",
    "clapscore": "clap_sim",
}


def metric_value(metric_name: str, metric: dict[str, Any]) -> float:
    if metric.get("status") != "ok":
        raise ValueError(
            f"metric {metric_name} is unavailable: "
            f"{metric.get('error_type')}: {metric.get('error')}"
        )
    corpus = metric.get("corpus")
    key = METRIC_VALUE_KEYS[metric_name]
    if not isinstance(corpus, dict) or not isinstance(corpus.get(key), (int, float)):
        raise ValueError(f"metric {metric_name} has no numeric corpus.{key}")
    return float(corpus[key])


def load_comparison(run_root: Path, expected_count: int) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        score_path = run_root / model_name / "metrics_standard_public" / "scores.json"
        if not score_path.is_file():
            raise FileNotFoundError(score_path)
        report = json.loads(score_path.read_text(encoding="utf-8"))
        task_scores: dict[str, Any] = {}
        for task in TASKS:
            task_report = report.get("tasks", {}).get(task)
            if not isinstance(task_report, dict):
                raise ValueError(f"{score_path}: missing task {task}")
            if task_report.get("count") != expected_count:
                raise ValueError(
                    f"{score_path}: {task} count={task_report.get('count')}, "
                    f"expected={expected_count}"
                )
            metrics = task_report.get("metrics", {})
            task_scores[task] = {
                name: metric_value(name, metrics[name]) for name in METRIC_VALUE_KEYS
            }
        models[model_name] = task_scores
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "expected_count_per_task": expected_count,
        "profile": "standard_public",
        "paper_exact_reproduction": False,
        "models": models,
    }


def markdown(summary: dict[str, Any]) -> str:
    metric_names = list(METRIC_VALUE_KEYS)
    lines = [
        "# EmotionTalk eight-model comparison",
        "",
        "> Audio-only four-task captioning. `standard_public` is not an exact "
        "reproduction of the paper's unpublished scoring implementation.",
        "",
    ]
    for task in TASKS:
        lines.extend([
            f"## {task.title()}",
            "",
            "| Model | " + " | ".join(metric_names) + " |",
            "|---|" + "---:|" * len(metric_names),
        ])
        for model_name in MODEL_NAMES:
            values = summary["models"][model_name][task]
            rendered = " | ".join(f"{values[name]:.6f}" for name in metric_names)
            lines.append(f"| {model_name} | {rendered} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate all eight EmotionTalk reports.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-prefix", default="comparison")
    args = parser.parse_args()
    if args.expected_count < 1:
        raise ValueError("--expected-count must be positive")
    run_root = args.run_root.resolve()
    summary = load_comparison(run_root, args.expected_count)
    json_path = run_root / f"{args.output_prefix}.json"
    markdown_path = run_root / f"{args.output_prefix}.md"
    write_json(json_path, summary)
    markdown_path.write_text(markdown(summary), encoding="utf-8")
    print(f"Wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
