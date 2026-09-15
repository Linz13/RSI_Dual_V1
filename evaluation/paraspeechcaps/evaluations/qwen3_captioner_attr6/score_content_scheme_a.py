#!/usr/bin/env python3
"""Score independent-field Attr6 Scheme A predictions over a fixed sample denominator."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    ALL_FIELDS,
    MANIFEST_META_PATH,
    MULTI_FIELDS,
    SCHEMA_PATH,
    SINGLE_FIELDS,
    load_schema,
    read_jsonl,
    sha256_file,
)
from content_scheme_a import PROTOCOL
from score import bootstrap_ci, mean, positive_recall, set_f1, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["sample_id"]): row
        for row in read_jsonl(path)
        if row.get("sample_id")
    }


def field_record(
    predictions: dict[str, dict[str, Any]], sample_id: str, field: str
) -> dict[str, Any]:
    return predictions.get(sample_id, {}).get("fields", {}).get(
        field,
        {"status": "missing", "prediction": None, "parse_mode": "missing"},
    )


def score_samples(
    manifest: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in manifest:
        scores: dict[str, float] = {}
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "benchmark_indices": "|".join(str(x) for x in sample["benchmark_indices"]),
            "source": sample["source"],
            "audio_path": sample["audio_path"],
        }
        parsed_fields = 0
        for field in SINGLE_FIELDS:
            record = field_record(predictions, sample["sample_id"], field)
            valid = record.get("status") == "success"
            prediction = record.get("prediction") if valid else None
            value = float(valid and prediction == sample["gt"][field])
            scores[field] = value
            parsed_fields += int(valid)
            row[f"status_{field}"] = record.get("status", "missing")
            row[f"parse_mode_{field}"] = record.get("parse_mode", "missing")
            row[f"gt_{field}"] = sample["gt"][field]
            row[f"pred_{field}"] = "" if prediction is None else prediction
            row[f"score_{field}"] = value
        for field in MULTI_FIELDS:
            record = field_record(predictions, sample["sample_id"], field)
            valid = record.get("status") == "success" and isinstance(
                record.get("prediction"), list
            )
            prediction = record.get("prediction", []) if valid else []
            value = set_f1(sample["gt"][field], prediction) if valid else 0.0
            scores[field] = value
            parsed_fields += int(valid)
            row[f"status_{field}"] = record.get("status", "missing")
            row[f"parse_mode_{field}"] = record.get("parse_mode", "missing")
            row[f"gt_{field}"] = "|".join(sample["gt"][field])
            row[f"pred_{field}"] = "|".join(prediction)
            row[f"score_{field}"] = value
        row["parsed_fields"] = parsed_fields
        row["all_six_fields_parsed"] = int(parsed_fields == len(ALL_FIELDS))
        row["sample_score"] = mean(scores.values())
        rows.append(row)
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
        record = field_record(predictions, sample["sample_id"], field)
        valid = record.get("status") == "success" and isinstance(
            record.get("prediction"), list
        )
        gold = set(sample["gt"][field])
        predicted = set(record.get("prediction", [])) if valid else set()
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
                record = field_record(predictions, sample["sample_id"], field)
                valid = record.get("status") == "success" and isinstance(
                    record.get("prediction"), list
                )
                gold = label in sample["gt"][field]
                predicted = valid and label in record.get("prediction", [])
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
    ids = {row["sample_id"] for row in scored}
    subset_manifest = [row for row in manifest if row["sample_id"] in ids]
    return {
        "samples": len(scored),
        "final_score": mean(row["sample_score"] for row in scored),
        "single_choice_accuracy": {
            field: mean(row[f"score_{field}"] for row in scored)
            for field in SINGLE_FIELDS
        },
        "multi_choice": {
            field: multilabel_summary(field, subset_manifest, predictions)
            for field in MULTI_FIELDS
        },
        "field_parse_rate": {
            field: mean(
                float(row[f"status_{field}"] == "success") for row in scored
            )
            for field in ALL_FIELDS
        },
        "all_six_fields_parse_rate": mean(
            float(row["all_six_fields_parsed"]) for row in scored
        ),
    }


def format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# ParaSpeechCaps Attr6 Content-only Scheme A Results",
        "",
        f"- Candidate: **{summary['candidate_name']}**",
        f"- Main content-only Attr6 score: **{format_pct(summary['final_score'])}**",
        f"- 95% bootstrap CI: **[{format_pct(summary['bootstrap_95_ci'][0])}, "
        f"{format_pct(summary['bootstrap_95_ci'][1])}]**",
        f"- Samples: **{summary['samples']}**",
        f"- All-six-field parse rate (diagnostic only): "
        f"**{format_pct(summary['all_six_fields_parse_rate'])}**",
        "",
        "## Field scores",
        "",
        "| Field | Score | Field parse rate | Auxiliary |",
        "| --- | ---: | ---: | --- |",
    ]
    for field in SINGLE_FIELDS:
        lines.append(
            f"| {field} | {format_pct(summary['single_choice_accuracy'][field])} | "
            f"{format_pct(summary['field_parse_rate'][field])} | accuracy over all samples |"
        )
    for field in MULTI_FIELDS:
        values = summary["multi_choice"][field]
        lines.append(
            f"| {field} | {format_pct(values['mean_sample_f1'])} | "
            f"{format_pct(summary['field_parse_rate'][field])} | "
            f"micro-F1={format_pct(values['micro_f1'])}; "
            f"positive recall={format_pct(values['positive_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Each audio is evaluated with six independent field-specific questions.",
            "- A parse failure affects only that field; it never zeros the other five fields.",
            "- Every field score uses the complete selected sample set as its denominator.",
            "- Option-ID parsing is primary; exact canonical-label matching is a deterministic fallback.",
            "- This content-only protocol is separate from, and does not replace, Strict Attr6.",
            "- Multi-label scores measure annotation agreement, not verified perceptual truth.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    manifest = read_jsonl(manifest_path)
    predictions = load_predictions(predictions_path)
    if not manifest:
        raise ValueError(f"manifest is empty: {manifest_path}")
    if set(predictions) != {row["sample_id"] for row in manifest}:
        missing = sorted({row["sample_id"] for row in manifest} - set(predictions))
        extra = sorted(set(predictions) - {row["sample_id"] for row in manifest})
        raise ValueError(f"prediction/manifest mismatch; missing={missing[:3]} extra={extra[:3]}")
    protocols = {row.get("protocol") for row in predictions.values()}
    if protocols != {PROTOCOL}:
        raise ValueError(f"unexpected prediction protocols: {protocols}")

    scored = score_samples(manifest, predictions)
    schema = load_schema()
    overall = summarize_subset(scored, manifest, predictions)
    ci_low, ci_high = bootstrap_ci([row["sample_score"] for row in scored])
    label_rows = per_label_rows(manifest, predictions, schema)
    per_source = {
        source: summarize_subset(
            [row for row in scored if row["source"] == source], manifest, predictions
        )
        for source in sorted({row["source"] for row in scored})
    }
    field_status_counts = {
        field: dict(
            Counter(
                field_record(predictions, sample["sample_id"], field).get(
                    "status", "missing"
                )
                for sample in manifest
            )
        )
        for field in ALL_FIELDS
    }
    run_meta_path = predictions_path.parent / "run_metadata.json"
    run_meta = (
        json.loads(run_meta_path.read_text(encoding="utf-8"))
        if run_meta_path.is_file()
        else {}
    )
    manifest_meta = (
        json.loads(MANIFEST_META_PATH.read_text(encoding="utf-8"))
        if MANIFEST_META_PATH.is_file()
        else {}
    )
    summary = {
        "protocol": PROTOCOL,
        "candidate_name": run_meta.get("candidate_name", "unknown"),
        **overall,
        "field_status_counts": field_status_counts,
        "bootstrap_rounds": 10_000,
        "bootstrap_seed": 20260826,
        "bootstrap_95_ci": [ci_low, ci_high],
        "per_source": per_source,
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "schema_sha256": sha256_file(SCHEMA_PATH),
        "manifest_metadata": manifest_meta,
        "run_metadata": run_meta,
    }
    output_dir = args.output_dir.expanduser().resolve()
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
                **{
                    f"accuracy_{field}": value
                    for field, value in values["single_choice_accuracy"].items()
                },
                **{
                    f"sample_f1_{field}": value["mean_sample_f1"]
                    for field, value in values["multi_choice"].items()
                },
            }
            for source, values in sorted(per_source.items())
        ],
    )
    write_csv(output_dir / "per_label.csv", label_rows)
    (output_dir / "final_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
