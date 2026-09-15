from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("AUDIO_CAPTION_ROOT", Path(__file__).resolve().parents[3])).resolve()
DEFAULT_FOUNDATION_ROOT = PROJECT_ROOT / "benchmark" / "AIR-Bench" / "Foundation"
DEFAULT_MANIFEST = DEFAULT_FOUNDATION_ROOT / "Foundation_meta_selected.json"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_REPORT_DIR = EXPERIMENT_DIR / "reports"
EXPECTED_TASKS = (
    "Acoustic_Scene_Classification_CochlScene",
    "Acoustic_Scene_Classification_TUT2017",
    "Sound_AQA_avqa",
    "Sound_AQA_clothoaqa",
    "Speaker_Age_Prediction_common_voice_13.0_en",
    "Speaker_Emotion_Recontion_iemocap",
    "Speaker_Emotion_Recontion_meld",
    "Speaker_Gender_Recognition_common_voice_13_en",
    "Speaker_Gender_Recognition_meld",
    "Speaker_Intent_Classification_slurp",
    "Speech_Entity_Reconition_slurp",
    "Spoken_Language_Identification_covost2",
)
LETTER_CHOICES = ("A", "B", "C", "D")
DEFAULT_PROMPT_PREFIX = (
    "Choose the most suitable answer from options A, B, C, and D to respond "
    "the question in next line. Reply with only one letter: A, B, C, or D."
)


@dataclass(frozen=True)
class AirBenchSample:
    sample_id: str
    audio_path: str
    path: str
    question: str
    choice_a: str
    choice_b: str
    choice_c: str | None
    choice_d: str | None
    answer_gt: str
    task_name: str
    dataset_name: str
    uniq_id: Any

    @property
    def task_id(self) -> str:
        return f"{self.task_name}_{self.dataset_name}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(val) for val in value]
        return str(value)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON list in {path}")
            return data
        return [json.loads(line) for line in f if line.strip()]


def build_sample_id(item: dict[str, Any]) -> str:
    task_id = f"{item['task_name']}_{item['dataset_name']}"
    return f"{task_id}::{item.get('uniq_id')}::{item['path']}"


def resolve_audio_path(item: dict[str, Any], foundation_root: Path) -> Path:
    task_id = f"{item['task_name']}_{item['dataset_name']}"
    rel_path = str(item["path"])
    if item["task_name"] == "Audio_Grounding":
        rel_path = rel_path[:-3] + "flac"
    return foundation_root / task_id / rel_path


def sample_from_item(item: dict[str, Any], foundation_root: Path) -> AirBenchSample:
    return AirBenchSample(
        sample_id=build_sample_id(item),
        audio_path=str(resolve_audio_path(item, foundation_root).resolve()),
        path=str(item["path"]),
        question=str(item["question"]),
        choice_a=str(item["choice_a"]),
        choice_b=str(item["choice_b"]),
        choice_c=item.get("choice_c"),
        choice_d=item.get("choice_d"),
        answer_gt=str(item["answer_gt"]),
        task_name=str(item["task_name"]),
        dataset_name=str(item["dataset_name"]),
        uniq_id=item.get("uniq_id"),
    )


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST, foundation_root: Path = DEFAULT_FOUNDATION_ROOT) -> list[AirBenchSample]:
    rows = read_manifest_rows(manifest_path)
    return [sample_from_item(row, foundation_root.expanduser().resolve()) for row in rows]


def write_manifest(samples: Iterable[AirBenchSample], manifest_path: Path) -> None:
    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def validate_manifest(samples: list[AirBenchSample], foundation_root: Path = DEFAULT_FOUNDATION_ROOT) -> None:
    if not samples:
        raise ValueError("AIR-Bench manifest is empty.")
    foundation_root = foundation_root.expanduser().resolve()
    missing_dirs = [task for task in EXPECTED_TASKS if not (foundation_root / task).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing AIR-Bench task folders: {missing_dirs}")
    missing_audio = [sample.audio_path for sample in samples if not Path(sample.audio_path).is_file()]
    if missing_audio:
        raise FileNotFoundError(f"Manifest references missing audio files, first: {missing_audio[0]}")


def choice_pairs(sample: AirBenchSample) -> list[tuple[str, str]]:
    pairs = [("A", sample.choice_a), ("B", sample.choice_b)]
    if sample.choice_c is not None:
        pairs.append(("C", str(sample.choice_c)))
    if sample.choice_d is not None:
        pairs.append(("D", str(sample.choice_d)))
    return pairs


def build_prompt(sample: AirBenchSample, prompt_prefix: str = DEFAULT_PROMPT_PREFIX) -> str:
    choices = "\n".join(f"{letter}. {choice}" for letter, choice in choice_pairs(sample))
    return f"{prompt_prefix}\n{sample.question}\n{choices}"


def gt_letter(sample: AirBenchSample) -> str:
    for letter, choice in choice_pairs(sample):
        if sample.answer_gt == choice:
            return letter
    raise ValueError(f"answer_gt does not match choices for {sample.sample_id}: {sample.answer_gt!r}")


def parse_prediction(response_text: str, sample: AirBenchSample | None = None) -> str:
    text = str(response_text or "").strip()
    if not text or text == "None":
        return ""
    compact = text.replace("\n", "").strip()
    if compact and compact[0].upper() in LETTER_CHOICES:
        return compact[0].upper()
    if len(compact) > 1 and compact[-2].upper() in LETTER_CHOICES:
        return compact[-2].upper()
    match = re.search(r"\b([ABCD])\b", compact, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if sample is not None:
        lowered = compact.lower()
        matches = [letter for letter, choice in choice_pairs(sample) if lowered == str(choice).strip().lower()]
        if len(matches) == 1:
            return matches[0]
    return ""


def build_common_parser(model_name: str, description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or f"Run {model_name} on AIR-Bench Foundation.")
    parser.add_argument("--foundation-root", type=Path, default=DEFAULT_FOUNDATION_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / f"{model_name}.jsonl")
    parser.add_argument("--max-samples", type=int, default=2, help="0 means all selected samples.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already present in output.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1, help="Concurrent workers. Keep 1 for local GPU models.")
    parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)
    parser.add_argument(
        "--task-filter",
        action="append",
        default=[],
        help="Task id to include, e.g. Speaker_Emotion_Recontion_meld. May be repeated.",
    )
    return parser


def select_samples(samples: list[AirBenchSample], start_index: int, max_samples: int, task_filter: list[str]) -> list[AirBenchSample]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if task_filter:
        allowed = set(task_filter)
        samples = [sample for sample in samples if sample.task_id in allowed]
    selected = samples[start_index:]
    if max_samples:
        selected = selected[:max_samples]
    return selected


def load_processed_ids(path: Path, model_name: str, *, success_only: bool = True) -> set[str]:
    processed: set[str] = set()
    for item in read_jsonl(path):
        if item.get("model_name") != model_name or not item.get("sample_id"):
            continue
        if success_only and item.get("status") != "success":
            continue
        if success_only and not item.get("prediction"):
            continue
        processed.add(str(item["sample_id"]))
    return processed


def public_args(args: argparse.Namespace) -> dict[str, Any]:
    hidden = {"api_key"}
    return {key: value for key, value in vars(args).items() if key not in hidden}


def make_result_record(
    *,
    model_name: str,
    sample: AirBenchSample,
    args: argparse.Namespace,
    status: str,
    response_text: str = "",
    error: str = "",
    duration_sec: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = parse_prediction(response_text, sample) if status == "success" else ""
    gt = gt_letter(sample)
    prompt = build_prompt(sample, args.prompt_prefix)
    record: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "model_name": model_name,
        "sample_id": sample.sample_id,
        "audio_path": sample.audio_path,
        "prompt": prompt,
        "prediction": prediction,
        "gt_letter": gt,
        "is_correct": prediction == gt if prediction else False,
        "status": status,
        "error": error,
        "duration_sec": duration_sec,
        "path": sample.path,
        "question": sample.question,
        "choice_a": sample.choice_a,
        "choice_b": sample.choice_b,
        "choice_c": sample.choice_c,
        "choice_d": sample.choice_d,
        "answer_gt": sample.answer_gt,
        "task_name": sample.task_name,
        "dataset_name": sample.dataset_name,
        "response": response_text,
        "uniq_id": sample.uniq_id,
    }
    if hasattr(args, "model_dir"):
        record["model_path"] = str(args.model_dir.expanduser().resolve())
    if hasattr(args, "api_url"):
        record["api_url"] = args.api_url
    if extra:
        record.update(extra)
    return record


def run_airbench_experiment(
    *,
    args: argparse.Namespace,
    model_name: str,
    load_model: Callable[[argparse.Namespace], Any],
    infer_one: Callable[[Any, AirBenchSample, argparse.Namespace], str],
    startup_extra: dict[str, Any] | None = None,
) -> None:
    args.foundation_root = args.foundation_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    manifest_samples = load_manifest(args.manifest, args.foundation_root)
    validate_manifest(manifest_samples, args.foundation_root)
    samples = select_samples(manifest_samples, args.start_index, args.max_samples, args.task_filter)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        processed = load_processed_ids(args.output, model_name, success_only=True)
        samples = [sample for sample in samples if sample.sample_id not in processed]
    else:
        processed = set()

    print(f"model_name: {model_name}")
    print(f"manifest: {args.manifest}")
    print(f"foundation_root: {args.foundation_root}")
    print(f"manifest_samples: {len(manifest_samples)}")
    print(f"selected_samples: {len(samples)}")
    if args.resume:
        print(f"resume_skipped_samples: {len(processed)}")
    if args.task_filter:
        print(f"task_filter: {', '.join(args.task_filter)}")
    print(f"output: {args.output}")
    print(f"num_workers: {args.num_workers}")

    if not samples:
        print("No samples selected; nothing to run.")
        return

    model_bundle = load_model(args)

    def run_one(index: int, sample: AirBenchSample) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response_text = infer_one(model_bundle, sample, args)
            duration_sec = time.perf_counter() - started
            return make_result_record(
                model_name=model_name,
                sample=sample,
                args=args,
                status="success",
                response_text=response_text,
                duration_sec=duration_sec,
                extra=startup_extra,
            )
        except Exception as exc:
            duration_sec = time.perf_counter() - started
            return make_result_record(
                model_name=model_name,
                sample=sample,
                args=args,
                status="error",
                error=repr(exc),
                duration_sec=duration_sec,
                extra=startup_extra,
            )

    if args.num_workers <= 1:
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] {sample.task_id} uniq_id={sample.uniq_id}", flush=True)
            record = run_one(index, sample)
            append_jsonl(args.output, record)
    else:
        workers = max(1, args.num_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_sample = {
                pool.submit(run_one, index, sample): (index, sample)
                for index, sample in enumerate(samples, start=1)
            }
            completed = 0
            for future in as_completed(future_to_sample):
                index, sample = future_to_sample[future]
                record = future.result()
                append_jsonl(args.output, record)
                completed += 1
                print(
                    f"[{completed}/{len(samples)} done; input_index={index}] "
                    f"{sample.task_id} uniq_id={sample.uniq_id} status={record['status']}",
                    flush=True,
                )

    print("Done.")


def latest_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id:
            latest[str(sample_id)] = row
    return latest


def summarize_output(path: Path) -> dict[str, Any]:
    rows = list(latest_by_sample(read_jsonl(path)).values())
    model_name = path.stem.replace(".repaired", "")
    if rows:
        model_name = str(rows[0].get("model_name") or model_name)
    per_task: dict[str, dict[str, Any]] = {}
    totals = {"total": 0, "success": 0, "valid_prediction": 0, "correct": 0, "error": 0, "invalid_prediction": 0}
    for row in rows:
        task_id = f"{row.get('task_name')}_{row.get('dataset_name')}"
        stats = per_task.setdefault(
            task_id,
            {"task_id": task_id, "total": 0, "success": 0, "valid_prediction": 0, "correct": 0, "error": 0, "invalid_prediction": 0},
        )
        stats["total"] += 1
        totals["total"] += 1
        if row.get("status") == "success":
            stats["success"] += 1
            totals["success"] += 1
            prediction = str(row.get("prediction") or "").strip().upper()
            if not prediction:
                prediction = parse_prediction(str(row.get("response") or ""))
            if prediction:
                stats["valid_prediction"] += 1
                totals["valid_prediction"] += 1
                if prediction == str(row.get("gt_letter") or "").strip().upper():
                    stats["correct"] += 1
                    totals["correct"] += 1
            else:
                stats["invalid_prediction"] += 1
                totals["invalid_prediction"] += 1
        else:
            stats["error"] += 1
            totals["error"] += 1
    for stats in per_task.values():
        valid = stats["valid_prediction"]
        stats["accuracy"] = stats["correct"] / valid if valid else 0.0
    totals["accuracy"] = totals["correct"] / totals["valid_prediction"] if totals["valid_prediction"] else 0.0
    return {
        "model_name": model_name,
        "output_file": str(path),
        **totals,
        "per_task": [per_task[key] for key in sorted(per_task)],
    }


def write_summary_reports(summaries: list[dict[str, Any]], report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    report_dir = report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "accuracy_summary.json"
    csv_path = report_dir / "accuracy_summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "model_name",
        "task_id",
        "total",
        "success",
        "valid_prediction",
        "correct",
        "accuracy",
        "error",
        "invalid_prediction",
        "output_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in fieldnames if key != "task_id"} | {"task_id": "__overall__"})
            for task_summary in summary["per_task"]:
                row = {key: task_summary.get(key, "") for key in fieldnames}
                row["model_name"] = summary["model_name"]
                row["output_file"] = summary["output_file"]
                writer.writerow(row)
