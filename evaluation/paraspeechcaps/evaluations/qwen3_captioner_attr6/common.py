#!/usr/bin/env python3
"""Shared schema, prompt, parsing, and file helpers for PSC Attr6."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


EVAL_ROOT = Path(__file__).resolve().parent
PSC_ROOT = EVAL_ROOT.parents[1]
WORKSPACE = EVAL_ROOT.parents[3]
CLUSTER_ROOT = WORKSPACE.parent
SCHEMA_PATH = EVAL_ROOT / "schema.json"
DEFAULT_RUN_ROOT = EVAL_ROOT / "runs" / "new_cluster_default"
MANIFEST_PATH = DEFAULT_RUN_ROOT / "manifest.jsonl"
MANIFEST_META_PATH = DEFAULT_RUN_ROOT / "manifest_meta.json"
TEST_AVAILABLE_CSV = PSC_ROOT / "data" / "test_available.csv"
TEST_FULL_CSV = PSC_ROOT / "data" / "test.csv"
TEST_AUDIO_DIR = PSC_ROOT / "audio" / "test"
DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "CAPTIONER_MODEL_PATH",
        WORKSPACE / "models" / "Qwen3-Omni-30B-A3B-Captioner",
    )
)
QWEN_ENV_PYTHON = Path(
    os.environ.get(
        "CAPTIONER_PYTHON",
        CLUSTER_ROOT / "miniconda3" / "envs" / "qwen3-captioner" / "bin" / "python",
    )
)
DEFAULT_MIDASHENG_MODEL_DIR = Path(
    os.environ.get(
        "MIDASHENG_MODEL_PATH",
        WORKSPACE / "models" / "MiDashengLM-7B-1021-BF16",
    )
)
MIDASHENG_ENV_PYTHON = Path(
    os.environ.get(
        "MIDASHENG_PYTHON",
        CLUSTER_ROOT / "miniconda3" / "envs" / "midasheng-captioner" / "bin" / "python",
    )
)

SINGLE_FIELDS = ("gender", "pitch", "speaking_rate", "accent")
MULTI_FIELDS = ("intrinsic_traits", "situational_traits")
ALL_FIELDS = SINGLE_FIELDS + MULTI_FIELDS


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def normalize_label(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value).strip().casefold().replace("_", "-"))
    aliases = {
        "vocal fry": "vocal-fry",
        "vocal--fry": "vocal-fry",
        "low pitched": "low-pitched",
        "medium pitched": "medium-pitched",
        "high pitched": "high-pitched",
        "sing-song": "singsong",
        "monotone": "monotonous",
    }
    return aliases.get(value, value)


def build_prompt(schema: dict[str, Any] | None = None) -> str:
    schema = schema or load_schema()
    singles = schema["single_choice"]
    multis = schema["multi_choice"]
    return (
        "Listen carefully to the speech audio and identify its observable speech attributes. "
        "Use only the canonical candidate values below. Do not infer the speaker's identity or "
        "use the spoken content as evidence for an attribute.\n\n"
        "Single-choice fields (choose exactly one candidate, or \"unknown\" only when the audio "
        "does not provide enough evidence):\n"
        f"- gender: {', '.join(singles['gender'])}\n"
        f"- pitch: {', '.join(singles['pitch'])}\n"
        f"- speaking_rate: {', '.join(singles['speaking_rate'])}\n"
        f"- accent: {', '.join(singles['accent'])}\n\n"
        "Multi-choice fields (select every clearly supported candidate; [] is valid; do not guess):\n"
        f"- intrinsic_traits: {', '.join(multis['intrinsic_traits'])}\n"
        f"- situational_traits: {', '.join(multis['situational_traits'])}\n\n"
        "IMPORTANT VALIDATION RULE: the two multi-choice vocabularies are disjoint and field-specific. "
        "intrinsic_traits MUST be a subset of the intrinsic_traits candidates only; situational_traits "
        "MUST be a subset of the situational_traits candidates only. Never move a label to the other "
        "field based on your own interpretation. For example, whispered is allowed only in "
        "situational_traits, never in intrinsic_traits. Check every selected label against its field's "
        "candidate list before answering.\n\n"
        "Return exactly one JSON object with these six keys and no explanation:\n"
        '{"gender":"...","pitch":"...","speaking_rate":"...","accent":"...",'
        '"intrinsic_traits":["..."],"situational_traits":["..."]}'
    )


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", str(text or "")):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def parse_prediction_with_violations(
    raw: str, schema: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[str]]:
    schema = schema or load_schema()
    candidates = {**schema["single_choice"], **schema["multi_choice"]}
    objects = extract_json_objects(raw)
    value = next((item for item in objects if set(ALL_FIELDS).issubset(item)), None)
    if value is None:
        raise ValueError("response does not contain a JSON object with all six fields")
    extras = sorted(set(value) - set(ALL_FIELDS))
    if extras:
        raise ValueError(f"unexpected JSON fields: {extras}")
    parsed: dict[str, Any] = {}
    violations: list[str] = []
    for field in SINGLE_FIELDS:
        if not isinstance(value[field], str):
            raise ValueError(f"{field} must be a string")
        item = normalize_label(value[field])
        allowed = set(candidates[field]) | {"unknown"}
        if item not in allowed:
            raise ValueError(f"invalid {field}: {value[field]!r}")
        parsed[field] = item
    for field in MULTI_FIELDS:
        if not isinstance(value[field], list):
            raise ValueError(f"{field} must be a JSON list")
        global_multi_order = {
            label: index
            for index, label in enumerate(
                schema["multi_choice"]["intrinsic_traits"]
                + schema["multi_choice"]["situational_traits"]
            )
        }
        global_multi = set(global_multi_order)
        field_allowed = set(candidates[field])
        normalized: list[str] = []
        for item in value[field]:
            if not isinstance(item, str):
                raise ValueError(f"{field} contains a non-string value")
            label = normalize_label(item)
            if label not in global_multi:
                raise ValueError(f"invalid {field} label: {item!r}")
            if label not in field_allowed:
                violations.append(f"cross_field_label:{field}:{label}")
            if label not in normalized:
                normalized.append(label)
        order = global_multi_order
        parsed[field] = sorted(normalized, key=order.__getitem__)
    return parsed, sorted(set(violations))


def parse_prediction(raw: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    parsed, _ = parse_prediction_with_violations(raw, schema)
    return parsed


def latest_records(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                continue
            previous = latest.get(sample_id)
            if previous is None:
                latest[sample_id] = row
            elif row.get("status") == "success" or previous.get("status") != "success":
                latest[sample_id] = row
    return latest


def duration_balanced_partition(
    samples: list[dict[str, Any]], workers: int
) -> list[list[dict[str, Any]]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    totals = [0.0] * workers
    for sample in sorted(
        samples, key=lambda row: (-float(row["duration_seconds"]), row["sample_id"])
    ):
        worker = min(range(workers), key=lambda index: (totals[index], index))
        shards[worker].append(sample)
        totals[worker] += float(sample["duration_seconds"])
    for shard in shards:
        shard.sort(key=lambda row: row["sample_id"])
    return shards
