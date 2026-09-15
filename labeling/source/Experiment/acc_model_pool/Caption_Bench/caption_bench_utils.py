"""Execution helpers for Caption_Bench model experiments.

The model backends were originally validated on AIR-Bench.  This module uses
the Caption_Bench manifest directly while preserving their ``load_model`` and
``infer_one`` interfaces.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(os.environ.get("AUDIO_CAPTION_ROOT", Path(__file__).resolve().parents[3])).resolve()
EXPERIMENT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark" / "Caption_Bench"
DEFAULT_MANIFEST = BENCHMARK_DIR / "builds" / "v1" / "manifest.jsonl"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


@dataclass(frozen=True)
class CaptionBenchSample:
    sample_id: str
    audio_path: str
    question: str
    choices: tuple[tuple[str, str], ...]
    answer_gt: str
    attribute: str
    attribute_name_zh: str
    source: str
    source_task: str
    coverage: str

    @property
    def task_id(self) -> str:
        return f"{self.attribute}::{self.source}::{self.source_task}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_from_row(row: dict[str, Any]) -> CaptionBenchSample:
    choices = tuple((str(choice["id"]), str(choice["text"])) for choice in row["choices"])
    if len(choices) < 2:
        raise ValueError(f"{row.get('sample_id')}: fewer than two choices")
    if not any(str(row["answer_gt"]).casefold() == text.casefold() for _, text in choices):
        raise ValueError(f"{row.get('sample_id')}: gold answer is not an option")
    return CaptionBenchSample(
        sample_id=str(row["sample_id"]),
        audio_path=str(row["audio_path"]),
        question=str(row.get("question", "")),
        choices=choices,
        answer_gt=str(row["answer_gt"]),
        attribute=str(row["attribute"]),
        attribute_name_zh=str(row.get("attribute_name_zh", "")),
        source=str(row["source"]),
        source_task=str(row["source_task"]),
        coverage=str(row["coverage"]),
    )


def load_manifest(path: Path) -> list[CaptionBenchSample]:
    if not path.is_file():
        raise FileNotFoundError(f"Caption_Bench manifest not found: {path}")
    samples = [sample_from_row(row) for row in read_jsonl(path)]
    if not samples:
        raise ValueError(f"Caption_Bench manifest is empty: {path}")
    return samples


def build_prompt(sample: CaptionBenchSample, prompt_prefix: str = "") -> str:
    choices = "\n".join(f"{choice_id}. {text}" for choice_id, text in sample.choices)
    instruction = (
        "Listen to the audio and answer the multiple-choice question. "
        "Reply with only one option ID or the exact option text."
    )
    parts = [part.strip() for part in (prompt_prefix, instruction, sample.question, choices) if part and part.strip()]
    return "\n".join(parts)


def parse_prediction(response_text: Any, sample: CaptionBenchSample) -> str:
    """Return canonical option text; accepts option IDs or exact option text."""
    import re

    text = str(response_text or "").strip()
    if not text:
        return ""
    folded = text.casefold()
    for choice_id, choice_text in sample.choices:
        if folded == choice_id.casefold() or folded == choice_text.strip().casefold():
            return choice_text

    # Rate/quality pair prompts explicitly allow a tie.  Audio-capable models
    # often explain that both clips are equally appropriate instead of emitting
    # the literal option ID; normalize those unambiguous answers to Tie.
    tie_choice = next((choice_text for choice_id, choice_text in sample.choices
                       if choice_id.casefold() == "tie" or choice_text.strip().casefold() == "tie"), None)
    if tie_choice is not None and re.search(
        r"(?:equally\s+appropriate|(?:both|two).{0,80}(?:appropriate|similar|same|too\s+short)|"
        r"either\s+(?:could|is)\s+(?:be\s+)?considered|neither\s+clip|no\s+difference)",
        folded,
    ):
        return tie_choice

    # Handles common forms such as "Answer: C", "Option 5." and "(B)".
    for choice_id, choice_text in sample.choices:
        pattern = rf"(?<![\w]){re.escape(choice_id)}(?![\w])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return choice_text

    # Some audio-capable models answer a binary intonation question by
    # transcribing the utterance rather than repeating its label.  A terminal
    # question mark is still an unambiguous realization of the ``疑问`` / question
    # option, so normalize that response instead of discarding a usable result.
    if re.search(r"[?？]\s*$", text):
        for _, choice_text in sample.choices:
            if choice_text.strip().casefold() in {"疑问", "question"}:
                return choice_text
    return ""


def build_common_parser(model_name: str, description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or f"Run {model_name} on Caption_Bench.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / f"{model_name}.jsonl")
    parser.add_argument("--max-samples", type=int, default=2, help="0 means all selected samples.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Skip samples with an existing successful prediction.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1, help="Use 1 for local GPU models; APIs may use 2--4.")
    parser.add_argument("--prompt-prefix", default="")
    parser.add_argument("--attribute-filter", action="append", default=[], help="Attribute ID to include; repeatable.")
    parser.add_argument("--source-filter", action="append", default=[], help="Source ID to include; repeatable.")
    parser.add_argument("--task-filter", action="append", default=[], help="Exact source_task to include; repeatable.")
    return parser


def select_samples(samples: list[CaptionBenchSample], args: argparse.Namespace) -> list[CaptionBenchSample]:
    if args.start_index < 0 or args.max_samples < 0:
        raise ValueError("--start-index and --max-samples must be >= 0")
    if args.attribute_filter:
        allowed = set(args.attribute_filter)
        samples = [sample for sample in samples if sample.attribute in allowed]
    if args.source_filter:
        allowed = set(args.source_filter)
        samples = [sample for sample in samples if sample.source in allowed]
    if args.task_filter:
        allowed = set(args.task_filter)
        samples = [sample for sample in samples if sample.source_task in allowed]
    selected = samples[args.start_index :]
    return selected[: args.max_samples] if args.max_samples else selected


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def successful_ids(path: Path, model_name: str) -> set[str]:
    if not path.exists():
        return set()
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("model_name") == model_name and row.get("sample_id"):
            latest[str(row["sample_id"])] = row
    return {sample_id for sample_id, row in latest.items() if row.get("status") == "success" and row.get("prediction")}


def result_record(model_name: str, sample: CaptionBenchSample, args: argparse.Namespace, *, status: str, response: str = "", error: str = "", duration_sec: float = 0.0) -> dict[str, Any]:
    prediction = parse_prediction(response, sample) if status == "success" else ""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "sample_id": sample.sample_id,
        "audio_path": sample.audio_path,
        "attribute": sample.attribute,
        "attribute_name_zh": sample.attribute_name_zh,
        "source": sample.source,
        "source_task": sample.source_task,
        "coverage": sample.coverage,
        "question": sample.question,
        "choices": [{"id": choice_id, "text": text} for choice_id, text in sample.choices],
        "prompt": build_prompt(sample, args.prompt_prefix),
        "answer_gt": sample.answer_gt,
        "response_text": response,
        "prediction": prediction,
        "is_correct": prediction == sample.answer_gt if prediction else False,
        "status": status,
        "error": error,
        "duration_sec": duration_sec,
    }


def run_airbench_experiment(*, args: argparse.Namespace, model_name: str, load_model: Callable[[argparse.Namespace], Any], infer_one: Callable[[Any, CaptionBenchSample, argparse.Namespace], str], startup_extra: dict[str, Any] | None = None) -> None:
    """Compatibility name used by the original AIR-Bench model entrypoints."""
    del startup_extra
    args.manifest = args.manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    samples = select_samples(load_manifest(args.manifest), args)
    missing = [sample.audio_path for sample in samples if not Path(sample.audio_path).is_file()]
    if missing:
        raise FileNotFoundError(f"Selected audio is missing, first: {missing[0]}")
    skipped = successful_ids(args.output, model_name) if args.resume else set()
    samples = [sample for sample in samples if sample.sample_id not in skipped]
    print(f"model_name: {model_name}\nmanifest: {args.manifest}\nselected_samples: {len(samples)}\noutput: {args.output}")
    if args.resume:
        print(f"resume_skipped_successes: {len(skipped)}")
    if not samples:
        print("No samples selected; nothing to run.")
        return
    bundle = load_model(args)

    def run_one(sample: CaptionBenchSample) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = str(infer_one(bundle, sample, args) or "")
            return result_record(model_name, sample, args, status="success", response=response, duration_sec=time.perf_counter() - started)
        except Exception as exc:  # keep long-running experiments resumable
            return result_record(model_name, sample, args, status="error", error=repr(exc), duration_sec=time.perf_counter() - started)

    if args.num_workers <= 1:
        for index, sample in enumerate(samples, 1):
            print(f"[{index}/{len(samples)}] {sample.task_id}", flush=True)
            append_jsonl(args.output, run_one(sample))
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(run_one, sample): (index, sample) for index, sample in enumerate(samples, 1)}
            for complete, future in enumerate(as_completed(futures), 1):
                index, sample = futures[future]
                record = future.result()
                append_jsonl(args.output, record)
                print(f"[{complete}/{len(samples)} done; input_index={index}] {sample.task_id} status={record['status']}", flush=True)
    print("Done.")
