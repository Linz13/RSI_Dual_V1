#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from common import ROOT, TASKS, TASK_REFERENCE_FIELDS, read_jsonl, write_json


def to_plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel") and value.numel() == 1:
        return float(value.item())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def guarded(metric_name: str, function: Callable[[], Any]) -> dict[str, Any]:
    try:
        corpus, sentence = function()
        return {"status": "ok", "corpus": to_plain(corpus), "sentence": to_plain(sentence)}
    except Exception as error:
        return {
            "status": "unavailable", "metric": metric_name,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        }


def normalize_metric_text(value: str) -> str:
    """Replace line breaks for text metrics only.

    Predictions and references on disk remain byte-for-byte unchanged. AAC's
    PTB tokenizer rejects newlines, while a line break has no semantic role in
    a caption, so replace line breaks only in the evaluator inputs.
    """
    return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def prepare_singleton_inputs(
    candidates: list[str], references: list[list[str]]
) -> tuple[list[str], list[list[str]], bool]:
    """Work around AAC metrics' singleton input limitations.

    With one candidate/reference item, some aac-metrics 0.6.0 paths reject a
    singleton or return scalar tensors that later get sliced. Duplicating the
    identical item preserves the corpus mean and gives the library a batched
    shape. The caller trims sentence-level output back to one item.
    """
    if len(candidates) == 1:
        return candidates * 2, references * 2, True
    return candidates, references, False


def trim_bertscore_sentences(sentence_scores: Any, duplicated: bool) -> Any:
    if not duplicated or not isinstance(sentence_scores, dict):
        return sentence_scores
    return {key: value[:1] for key, value in sentence_scores.items()}


def evaluate_aac_metric(
    name: str, candidates: list[str], references: list[list[str]], device: str
) -> Any:
    """Run an AAC metric, batching singleton SPIDEr inputs when needed."""
    from aac_metrics import Evaluate

    score_candidates, score_references, duplicated = candidates, references, False
    if name == "spider":
        score_candidates, score_references, duplicated = prepare_singleton_inputs(candidates, references)
    output = Evaluate(metrics=[name], preprocess=True, device=device, verbose=0)(
        score_candidates, score_references
    )
    if duplicated and isinstance(output, tuple) and len(output) == 2:
        corpus, sentence = output
        return corpus, trim_bertscore_sentences(sentence, duplicated)
    return output


def ensure_unique_predictions(rows: list[dict]) -> None:
    keys = [(row.get("id"), row.get("task")) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate predictions: {duplicates[:5]}")


def evaluate_task(candidates: list[str], references: list[list[str]], audio_paths: list[str], device: str) -> dict[str, Any]:
    candidates = [normalize_metric_text(candidate) for candidate in candidates]
    references = [[normalize_metric_text(reference) for reference in group] for group in references]
    results: dict[str, Any] = {}
    try:
        import aac_metrics
        from aac_metrics import Evaluate
        version = getattr(aac_metrics, "__version__", "unknown")
    except Exception as error:
        message = {"status": "unavailable", "error_type": type(error).__name__, "error": str(error)}
        return {name: dict(message) for name in ("bleu_4", "rouge_l", "meteor", "spider", "fense", "bertscore", "clapscore")}

    for name in ("bleu_4", "rouge_l", "meteor", "spider"):
        results[name] = guarded(name, lambda name=name: evaluate_aac_metric(
            name, candidates, references, device
        ))

    def run_fense() -> Any:
        from aac_metrics.classes import FENSE
        return FENSE(device=device, verbose=0)(candidates, references)

    def run_bertscore() -> Any:
        from aac_metrics.classes import BERTScoreMRefs
        scorer = BERTScoreMRefs(
            model="google-bert/bert-base-chinese", device=device,
            batch_size=8, max_length=128, reduction="max", verbose=0,
        )
        score_candidates, score_references, duplicated = prepare_singleton_inputs(candidates, references)
        corpus, sentence = scorer(score_candidates, score_references)
        return corpus, trim_bertscore_sentences(sentence, duplicated)

    def run_clapscore() -> Any:
        from aac_metrics.functional.clap_sim import clap_sim
        return clap_sim(
            candidates=candidates, audio_paths=audio_paths,
            clap_method="audio", clap_model="MS-CLAP-2023",
            device=device, batch_size=8, seed=42, verbose=0,
        )

    results["fense"] = guarded("fense", run_fense)
    results["bertscore"] = guarded("bertscore", run_bertscore)
    results["clapscore"] = guarded("clapscore", run_clapscore)
    results["aac_metrics_version"] = version
    return results


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# EmotionTalk caption metric report", "", "> Pipeline smoke unless all four task counts are 1,929. This is the `standard_public` profile, not an exact reproduction of Table 5.", ""]
    for task in TASKS:
        task_report = report["tasks"].get(task)
        if task_report is None:
            continue
        lines += [f"## {task.title()} Caption", "", f"Predictions: {task_report['count']}", "", "| Metric | Status | Corpus score / error |", "|---|---|---|"]
        for metric, value in task_report["metrics"].items():
            if metric == "aac_metrics_version":
                continue
            if value["status"] == "ok":
                detail = json.dumps(value["corpus"], ensure_ascii=False)
            else:
                detail = f"{value.get('error_type')}: {value.get('error')}".replace("\n", " ")
            lines.append(f"| {metric} | {value['status']} | {detail} |")
        lines.append("")
    lines += ["## Comparability note", "", "EmotionTalk does not publish the Table 5 scoring code, Chinese tokenization setup, or exact FENSE/CLAP checkpoints. FENSE/SPICE-derived SPIDEr/MS-CLAP defaults are not designed specifically for Chinese emotional speech; these scores must not be presented as paper-exact.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EmotionTalk caption predictions per task.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=ROOT / "data/test_references.jsonl")
    parser.add_argument("--profile", choices=["standard_public"], default="standard_public")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--require-all-metrics",
        action="store_true",
        help="Fail after writing the report if any configured metric is unavailable.",
    )
    parser.add_argument("--device", default="cuda_if_available")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    refs_rows = read_jsonl(args.references.resolve())
    refs_by_id = {row["id"]: row for row in refs_rows}
    predictions = read_jsonl(args.predictions.resolve())
    ensure_unique_predictions(predictions)
    keys = [(row.get("id"), row.get("task")) for row in predictions]
    invalid_tasks = sorted({task for _, task in keys if task not in TASKS})
    if invalid_tasks:
        raise ValueError(f"Invalid tasks: {invalid_tasks}")
    unknown = sorted({uid for uid, _ in keys if uid not in refs_by_id})
    if unknown:
        raise ValueError(f"Unknown prediction IDs: {unknown[:5]}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        if not isinstance(row.get("prediction"), str) or not row["prediction"].strip():
            raise ValueError(f"Empty prediction: {row.get('id')}/{row.get('task')}")
        grouped[row["task"]].append(row)
    if not args.allow_partial:
        counts = {task: len(grouped[task]) for task in TASKS}
        if any(count != 1929 for count in counts.values()):
            raise ValueError(f"Full evaluation requires 1929 predictions per task; got {counts}")

    report: dict[str, Any] = {
        "profile": args.profile, "paper_exact_reproduction": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "allow_partial": args.allow_partial,
        "require_all_metrics": args.require_all_metrics, "tasks": {},
    }
    for task in TASKS:
        rows = grouped.get(task, [])
        if not rows:
            continue
        candidates = [row["prediction"] for row in rows]
        references: list[list[str]] = []
        audio_paths: list[str] = []
        for row in rows:
            reference = refs_by_id[row["id"]]
            field = TASK_REFERENCE_FIELDS[task]
            value = reference[field]
            references.append(value if isinstance(value, list) else [value])
            audio_paths.append(str((ROOT / reference["audio_path"]).resolve()))
        metrics = evaluate_task(candidates, references, audio_paths, args.device)
        report["tasks"][task] = {"count": len(rows), "metrics": metrics}

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "scores.json", report)
    (output_dir / "scores.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Wrote {output_dir / 'scores.json'} and {output_dir / 'scores.md'}")
    if args.require_all_metrics:
        unavailable = []
        for task, task_report in report["tasks"].items():
            for metric, value in task_report["metrics"].items():
                if metric != "aac_metrics_version" and value.get("status") != "ok":
                    unavailable.append(f"{task}/{metric}")
        if unavailable:
            raise SystemExit("Required metrics unavailable: " + ", ".join(unavailable))


if __name__ == "__main__":
    main()
