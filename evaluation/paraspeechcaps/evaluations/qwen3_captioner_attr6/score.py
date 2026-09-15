#!/usr/bin/env python3
"""Score PSC Attr6 predictions and create deterministic reports."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from common import (
    ALL_FIELDS,
    DEFAULT_RUN_ROOT,
    EVAL_ROOT,
    MANIFEST_META_PATH,
    MANIFEST_PATH,
    MULTI_FIELDS,
    SCHEMA_PATH,
    SINGLE_FIELDS,
    load_schema,
    read_jsonl,
    sha256_file,
)


DEFAULT_PREDICTIONS = DEFAULT_RUN_ROOT / "outputs" / "predictions.jsonl"
DEFAULT_REPORT_DIR = DEFAULT_RUN_ROOT / "reports"
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_ROUNDS = 10_000


def set_f1(gold: Iterable[str], prediction: Iterable[str]) -> float:
    gold_set, prediction_set = set(gold), set(prediction)
    if not gold_set and not prediction_set:
        return 1.0
    tp = len(gold_set & prediction_set)
    fp = len(prediction_set - gold_set)
    fn = len(gold_set - prediction_set)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def positive_recall(gold: Iterable[str], prediction: Iterable[str]) -> float | None:
    gold_set, prediction_set = set(gold), set(prediction)
    if not gold_set:
        return None
    return len(gold_set & prediction_set) / len(gold_set)


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def bootstrap_ci(
    values: list[float], rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = sorted(
        mean(values[rng.randrange(len(values))] for _ in values) for _ in range(rounds)
    )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def latest_predictions(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            previous = latest.get(sample_id)
            if previous is None or row.get("status") == "success" or previous.get("status") != "success":
                latest[sample_id] = row
    return latest


def score_samples(
    manifest: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in manifest:
        record = predictions.get(sample["sample_id"], {})
        valid = record.get("status") == "success" and isinstance(record.get("prediction"), dict)
        prediction = record.get("prediction", {}) if valid else {}
        field_scores: dict[str, float] = {}
        output: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "benchmark_indices": "|".join(str(x) for x in sample["benchmark_indices"]),
            "source": sample["source"],
            "audio_path": sample["audio_path"],
            "status": record.get("status", "missing"),
            "error": record.get("error", "missing prediction"),
            "schema_violations": "|".join(record.get("schema_violations", [])),
        }
        for field in SINGLE_FIELDS:
            predicted_value = prediction.get(field, "")
            value = float(valid and predicted_value == sample["gt"][field])
            field_scores[field] = value
            output[f"gt_{field}"] = sample["gt"][field]
            output[f"pred_{field}"] = predicted_value
            output[f"score_{field}"] = value
            output[f"unknown_{field}"] = int(valid and predicted_value == "unknown")
        for field in MULTI_FIELDS:
            gold_values = sample["gt"][field]
            predicted_values = prediction.get(field, []) if valid else []
            value = set_f1(gold_values, predicted_values) if valid else 0.0
            recall = positive_recall(gold_values, predicted_values) if valid else (
                0.0 if gold_values else None
            )
            field_scores[field] = value
            output[f"gt_{field}"] = "|".join(gold_values)
            output[f"pred_{field}"] = "|".join(predicted_values)
            output[f"score_{field}"] = value
            output[f"positive_recall_{field}"] = "" if recall is None else recall
            output[f"exact_{field}"] = int(valid and set(gold_values) == set(predicted_values))
        output["sample_score"] = mean(field_scores.values())
        rows.append(output)
    return rows


def multilabel_summary(
    field: str,
    manifest: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, float]:
    tp = fp = fn = exact = 0
    sample_scores: list[float] = []
    recalls: list[float] = []
    for sample in manifest:
        record = predictions.get(sample["sample_id"], {})
        valid = record.get("status") == "success" and isinstance(record.get("prediction"), dict)
        gold = set(sample["gt"][field])
        predicted = set(record["prediction"].get(field, [])) if valid else set()
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        sample_scores.append(set_f1(gold, predicted) if valid else 0.0)
        exact += int(valid and gold == predicted)
        recall = positive_recall(gold, predicted)
        if recall is not None:
            recalls.append(recall if valid else 0.0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "mean_sample_f1": mean(sample_scores),
        "exact_set_accuracy": exact / len(manifest) if manifest else 0.0,
        "positive_recall": mean(recalls),
    }


def per_label_rows(
    manifest: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in MULTI_FIELDS:
        for label in schema["multi_choice"][field]:
            tp = fp = fn = 0
            for sample in manifest:
                record = predictions.get(sample["sample_id"], {})
                valid = record.get("status") == "success" and isinstance(record.get("prediction"), dict)
                gold = label in sample["gt"][field]
                predicted = valid and label in record["prediction"].get(field, [])
                tp += int(gold and predicted)
                fp += int(not gold and predicted)
                fn += int(gold and not predicted)
            support = tp + fn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            rows.append(
                {
                    "field": field,
                    "label": label,
                    "support": support,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "macro_eligible_support_ge_5": int(support >= 5),
                }
            )
    return rows


def summarize_subset(
    scored: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_ids = {row["sample_id"] for row in scored}
    subset_manifest = [row for row in manifest if row["sample_id"] in sample_ids]
    return {
        "samples": len(scored),
        "final_score": mean(row["sample_score"] for row in scored),
        "single_choice_accuracy": {
            field: mean(row[f"score_{field}"] for row in scored) for field in SINGLE_FIELDS
        },
        "unknown_rate": {
            field: mean(row[f"unknown_{field}"] for row in scored) for field in SINGLE_FIELDS
        },
        "multi_choice": {
            field: multilabel_summary(field, subset_manifest, predictions) for field in MULTI_FIELDS
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_report(summary: dict[str, Any]) -> str:
    singles = summary["single_choice_accuracy"]
    multis = summary["multi_choice"]
    sources = summary["per_source"]
    lines = [
        "# Qwen3-Captioner ParaSpeechCaps Attr6 Results",
        "",
        f"- Main sample-weighted Attr6 score: **{format_pct(summary['final_score'])}**",
        f"- 95% bootstrap CI: **[{format_pct(summary['bootstrap_95_ci'][0])}, "
        f"{format_pct(summary['bootstrap_95_ci'][1])}]**",
        f"- Coverage: **{summary['successful_samples']}/{summary['samples']}** "
        f"({format_pct(summary['coverage'])})",
        f"- Samples with cross-field schema violations: **{summary['samples_with_schema_violations']}**",
        f"- Unique audio: **{summary['samples']}** (from 142 annotation rows)",
        "",
        "## Field scores",
        "",
        "| Field | Score | Auxiliary |",
        "| --- | ---: | --- |",
    ]
    for field in SINGLE_FIELDS:
        lines.append(
            f"| {field} | {format_pct(singles[field])} | "
            f"unknown={format_pct(summary['unknown_rate'][field])} |"
        )
    for field in MULTI_FIELDS:
        value = multis[field]
        lines.append(
            f"| {field} | {format_pct(value['mean_sample_f1'])} | "
            f"micro-F1={format_pct(value['micro_f1'])}; "
            f"positive recall={format_pct(value['positive_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Per source",
            "",
            "| Source | Samples | Attr6 score |",
            "| --- | ---: | ---: |",
        ]
    )
    for source, values in sorted(sources.items()):
        lines.append(f"| {source} | {values['samples']} | {format_pct(values['final_score'])} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is a prompted closed-set speech-attribute evaluation, not unrestricted audio captioning.",
            "- Strict precision/F1 measures agreement with ParaSpeechCaps annotations; an unannotated trait is not necessarily inaudible or false.",
            "- The result covers the locally available 140 unique audios, not the complete 246-row official test set.",
            "- Public-mirror audio identity is matched, but some files do not reproduce all paper preprocessing such as loudness normalization and VoiceFixer.",
            "- Noise is intentionally excluded because the local audit found metadata/reference inconsistencies.",
            "",
            "See `summary.json`, `per_sample.csv`, `per_source.csv`, and `per_label.csv` for auditable details.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    manifest = read_jsonl(args.manifest.resolve())
    if len(manifest) != 140:
        raise ValueError(f"expected 140 manifest rows, found {len(manifest)}")
    predictions = latest_predictions(args.predictions.resolve())
    scored = score_samples(manifest, predictions)
    schema = load_schema()
    label_rows = per_label_rows(manifest, predictions, schema)
    overall = summarize_subset(scored, manifest, predictions)
    ci_low, ci_high = bootstrap_ci([row["sample_score"] for row in scored])
    per_source = {
        source: summarize_subset(
            [row for row in scored if row["source"] == source], manifest, predictions
        )
        for source in sorted({row["source"] for row in scored})
    }
    eligible_f1 = [row["f1"] for row in label_rows if row["macro_eligible_support_ge_5"]]
    statuses = Counter(row["status"] for row in scored)
    schema_violation_counter = Counter(
        violation
        for record in predictions.values()
        for violation in record.get("schema_violations", [])
    )
    run_meta_path = args.predictions.resolve().parent / "run_metadata.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.is_file() else {}
    manifest_meta = (
        json.loads(MANIFEST_META_PATH.read_text(encoding="utf-8"))
        if MANIFEST_META_PATH.is_file()
        else {}
    )
    summary = {
        "protocol": schema["protocol"],
        "model_name": "qwen3_captioner",
        **overall,
        "successful_samples": statuses.get("success", 0),
        "coverage": statuses.get("success", 0) / len(scored),
        "status_counts": dict(statuses),
        "samples_with_schema_violations": sum(bool(row["schema_violations"]) for row in scored),
        "schema_violation_counts": dict(schema_violation_counter),
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_95_ci": [ci_low, ci_high],
        "support_ge_5_label_macro_f1": mean(eligible_f1),
        "support_ge_5_label_count": len(eligible_f1),
        "per_source": per_source,
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "predictions_sha256": sha256_file(args.predictions.resolve()),
        "schema_sha256": sha256_file(SCHEMA_PATH),
        "manifest_metadata": manifest_meta,
        "run_metadata": run_meta,
        "warnings": [
            "Strict multi-label precision/F1 is annotation agreement, not verified perceptual truth.",
            "This is the 140-unique-audio locally available subset, not the complete 246-row test set.",
            "Noise is excluded by protocol.",
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "per_sample.csv", scored)
    write_csv(
        output_dir / "per_source.csv",
        [
            {
                "source": source,
                "samples": values["samples"],
                "final_score": values["final_score"],
                **{f"accuracy_{key}": value for key, value in values["single_choice_accuracy"].items()},
                **{
                    f"sample_f1_{key}": value["mean_sample_f1"]
                    for key, value in values["multi_choice"].items()
                },
            }
            for source, values in sorted(per_source.items())
        ],
    )
    write_csv(output_dir / "per_label.csv", label_rows)
    (output_dir / "final_report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
