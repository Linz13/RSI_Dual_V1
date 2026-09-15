#!/usr/bin/env python3
"""Inventory, resume checks, and summaries for later Captioner checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = ("emotiontalk", "stylecap", "paraspeechcaps")
SUITE_PRIORITY = {"emotiontalk": 300, "stylecap": 200, "paraspeechcaps": 100}
EXPECTED = {
    ("emotiontalk", "smoke"): (4, 1),
    ("emotiontalk", "full"): (7716, 1929),
    ("stylecap", "smoke"): (4, 4),
    ("stylecap", "full"): (3112, 3112),
    ("paraspeechcaps", "smoke"): (1, 6),
    ("paraspeechcaps", "full"): (140, 840),
}


def reward_profile_selected() -> bool:
    profile = os.environ.get("LATER_CAPTION_PROFILE", "legacy")
    if profile not in ("legacy", "midasheng_rewardv2_v3", "midasheng_rewardv3", "qwen25_v1_v2"):
        raise ValueError(f"Unknown LATER_CAPTION_PROFILE: {profile}")
    return profile != "legacy"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    trajectory: str
    label: str
    family: str
    round: int
    model: str
    adapter: str
    python: str
    attn: str
    status: str
    detail: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def best_effort_shared_chmod(path: Path, mode: int) -> None:
    """Set shared permissions when owned locally; tolerate cross-UID QuarkFS."""
    try:
        os.chmod(path, mode)
    except PermissionError:
        if not os.access(path, os.W_OK):
            raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_path(path: Path) -> str:
    material = [
        {"path": str(item.relative_to(path)), "sha256": sha256_file(item)}
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        if "training_metrics" not in item.parts
    ]
    return stable_hash(material)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def adapter_check(adapter: Path, model: Path, compute_hash: bool = True) -> tuple[bool, str]:
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not adapter.is_dir():
        return False, f"missing adapter directory: {adapter}"
    if not config_path.is_file() or not os.access(config_path, os.R_OK):
        return False, f"missing/unreadable adapter config: {config_path}"
    if not weights_path.is_file() or not os.access(weights_path, os.R_OK):
        return False, f"missing/unreadable adapter weights: {weights_path}"
    try:
        config = read_json(config_path)
    except Exception as exc:  # pragma: no cover - exact OS error is host dependent
        return False, f"invalid adapter config: {exc}"
    configured = config.get("base_model_name_or_path")
    if configured and Path(configured).resolve() != model.resolve():
        return False, f"adapter base mismatch: {configured} != {model}"
    return True, sha256_file(weights_path) if compute_hash else "identity hash deferred"


def _fixed_candidate(
    caption_root: Path,
    candidate_id: str,
    trajectory: str,
    family: str,
    round_number: int,
    run: Path,
    verify_adapter_hash: bool,
) -> Candidate:
    cluster_root = caption_root.parent
    is_qwen = family == "qwen"
    model = caption_root / "models" / (
        "Qwen3-Omni-30B-A3B-Captioner" if is_qwen else "MiDashengLM-7B-1021-BF16"
    )
    python = cluster_root / "miniconda3/envs" / (
        "qwen3-captioner/bin/python" if is_qwen else "midasheng-captioner/bin/python"
    )
    adapter = run / f"round_{round_number:03d}/checkpoints/caption_final"
    ok, detail = adapter_check(adapter, model, compute_hash=verify_adapter_hash)
    commit = run / f"round_{round_number:03d}/commit.json"
    summary = run / f"round_{round_number:03d}/summary.json"
    if not commit.exists() or not summary.exists():
        ok = False
        detail = f"round is not committed: {commit} / {summary}"
    return Candidate(
        candidate_id, trajectory, f"r{round_number}", family, round_number,
        str(model.resolve()), str(adapter.resolve()), str(python.resolve()),
        "flash_attention_2" if is_qwen else "sdpa", "ready" if ok else "invalid", detail,
    )


def discover_candidates(
    caption_root: Path,
    verify_reward_hash: bool = True,
    verify_adapter_hash: bool = True,
) -> list[Candidate]:
    caption_root = caption_root.resolve()
    if os.environ.get("LATER_CAPTION_PROFILE") == "qwen25_v1_v2":
        runs = [
            (caption_root / "DualISL_Train/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_20260903_run01", "qwen25_v1", range(10)),
            (caption_root / "DualISL_Train_RewardV2/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_reward_v2_20260903_run02", "qwen25_rewardv2", range(10)),
        ]
        return discover_reward_candidates(
            caption_root, runs, verify_reward_hash, verify_adapter_hash,
            family="qwen25", model_name="Qwen2.5-Omni-3B",
            python_path=os.environ.get("QWEN25_PY") or str(caption_root.parent / "miniconda3/envs/qwen2_5-omni-3b-captioner/bin/python"),
            attn="flash_attention_2",
        )
    if reward_profile_selected():
        v2 = caption_root / "DualISL_Train_RewardV2/runs"
        runs = [
            (v2 / "dual_recursive_8gpu_h100_midasheng_reward_v2_20260901_run01", "midasheng_rewardv2", range(10)),
            (v2 / "dual_recursive_8gpu_h100_midasheng_reward_v2_from_r9_20260904_run01", "midasheng_rewardv2", range(10, 20)),
            (caption_root / "DualISL_Train_RewardV3/runs/midasheng_7b_reward_v3_10rounds_20260905_run01", "midasheng_rewardv3", range(10)),
        ]
        if os.environ.get("LATER_CAPTION_PROFILE") == "midasheng_rewardv3":
            runs = [item for item in runs if item[1] == "midasheng_rewardv3"]
        return discover_reward_candidates(caption_root, runs, verify_reward_hash, verify_adapter_hash)
    train_runs = caption_root / "DualISL_Train/runs"
    candidates = [
        _fixed_candidate(
            caption_root, f"qwen_v1_r{round_number}", "qwen_v1", "qwen", round_number,
            train_runs / "dual_recursive_8gpu_h100_qwen_from_r2_20260831_run01",
            verify_adapter_hash,
        )
        for round_number in range(3, 5)
    ]
    candidates.extend(
        _fixed_candidate(
            caption_root, f"midasheng_v1_r{round_number}", "midasheng_v1", "midasheng", round_number,
            train_runs / "dual_recursive_8gpu_h100_midasheng_from_r2_20260831_run01",
            verify_adapter_hash,
        )
        for round_number in range(3, 13)
    )

    reward_run = (
        caption_root / "DualISL_Train_RewardV2/runs"
        / "dual_recursive_8gpu_h100_midasheng_reward_v2_20260901_run01"
    )
    candidates.extend(discover_reward_candidates(
        caption_root, [(reward_run, "midasheng_rewardv2", range(10))],
        verify_reward_hash, verify_adapter_hash,
    ))
    return candidates


def discover_reward_candidates(
    caption_root: Path,
    runs: Iterable[tuple[Path, str, range]],
    verify_reward_hash: bool,
    verify_adapter_hash: bool,
    *,
    family: str = "midasheng",
    model_name: str = "MiDashengLM-7B-1021-BF16",
    python_path: str | None = None,
    attn: str = "sdpa",
) -> list[Candidate]:
    candidates = []
    model = caption_root / "models" / model_name
    python = Path(python_path) if python_path else caption_root.parent / "miniconda3/envs/midasheng-captioner/bin/python"
    for reward_run, trajectory, round_number in (
        (run, trajectory, number) for run, trajectory, rounds in runs for number in rounds
    ):
        round_root = reward_run / f"round_{round_number:03d}"
        adapter = round_root / "checkpoints/caption_final"
        commit_path = round_root / "commit.json"
        candidate_id = f"{trajectory}_r{round_number}"
        base = dict(
            candidate_id=candidate_id, trajectory=trajectory,
            label=f"r{round_number}", family=family, round=round_number,
            model=str(model.resolve()), adapter=str(adapter.resolve()),
            python=str(python.resolve()), attn=attn,
        )
        if not commit_path.exists():
            candidates.append(Candidate(**base, status="pending", detail="commit.json not present"))
            continue
        try:
            commit = read_json(commit_path)
        except Exception as exc:
            candidates.append(Candidate(**base, status="invalid", detail=f"invalid commit.json: {exc}"))
            continue
        ok, detail = adapter_check(adapter, model, compute_hash=verify_adapter_hash)
        committed_path = Path(str(commit.get("captioner", {}).get("path", ""))).resolve()
        if commit.get("round") != round_number:
            ok, detail = False, f"commit round mismatch: {commit.get('round')}"
        elif committed_path != adapter.resolve():
            ok, detail = False, f"commit Captioner path mismatch: {committed_path}"
        elif ok:
            try:
                actual_hash = hash_path(adapter) if verify_reward_hash else commit.get("captioner", {}).get("sha256")
            except OSError as exc:
                candidates.append(Candidate(**base, status="invalid", detail=f"cannot hash checkpoint: {exc}"))
                continue
            committed_hash = commit.get("captioner", {}).get("sha256")
            if actual_hash != committed_hash:
                ok, detail = False, f"commit Captioner hash mismatch: {actual_hash} != {committed_hash}"
            else:
                detail = actual_hash
        candidates.append(Candidate(**base, status="ready" if ok else "invalid", detail=detail))
    return candidates


def output_dir(output_root: Path, suite: str, size: str, candidate_id: str) -> Path:
    return output_root / suite / size / candidate_id


def completion_status(
    output_root: Path,
    suite: str,
    size: str,
    candidate_id: str,
    candidate: Candidate | None = None,
) -> tuple[bool, str]:
    root = output_dir(output_root, suite, size, candidate_id)
    expected_primary, expected_secondary = EXPECTED[(suite, size)]
    try:
        if suite == "emotiontalk":
            rows = count_jsonl(root / "predictions.jsonl")
            scores = read_json(root / "metrics_standard_public/scores.json")
            counts = {name: value.get("count") for name, value in scores.get("tasks", {}).items()}
            ok = rows == expected_primary and counts == {
                name: expected_secondary for name in ("speaker", "style", "emotion", "overall")
            }
            detail = f"prediction_rows={rows}; task_counts={counts}"
        if suite == "paraspeechcaps":
            samples = count_jsonl(root / "outputs/predictions.jsonl")
            fields = count_jsonl(root / "outputs/field_records.jsonl")
            summary = read_json(root / "reports/summary.json")
            ok = samples == expected_primary and fields == expected_secondary and summary.get("samples") == expected_primary
            detail = f"samples={samples}; fields={fields}; scored={summary.get('samples')}"
        elif suite == "stylecap":
            predictions = count_jsonl(root / "predictions.jsonl")
            if size == "smoke":
                metadata = read_json(root / "run_metadata.json")
                ok = predictions == expected_primary and metadata.get("prediction_rows") == expected_primary
                detail = f"prediction_rows={predictions}"
            else:
                summary = read_json(root / "evaluation_summary.json")
                ok = predictions == expected_primary and summary.get("prediction_rows") == expected_primary
                detail = f"prediction_rows={predictions}; scored={summary.get('prediction_rows')}"
        if ok and candidate is not None:
            identity = _result_identity(suite, root)
            metadata_adapter = identity["adapter"]
            actual_path = Path(str((metadata_adapter or {}).get("path", ""))).resolve()
            expected_path = Path(candidate.adapter).resolve()
            expected_hash = sha256_file(expected_path / "adapter_model.safetensors")
            expected_backend = "qwen3_omni" if candidate.family == "qwen" and suite == "emotiontalk" else (
                "qwen3" if candidate.family == "qwen" else "midasheng"
            )
            if candidate.family == "qwen25":
                expected_backend = "qwen25"
            expected_protocol = {
                "paraspeechcaps": "paraspeechcaps-attr6-content-scheme-a-v1",
                "stylecap": "stylecap-promptspeech-speaker-open-mcq-v1",
            }.get(suite)
            if (
                Path(str(identity["model"])).resolve() != Path(candidate.model).resolve()
                or identity["backend"] != expected_backend
                or identity["attn"] != candidate.attn
                or (expected_protocol is not None and identity["protocol"] != expected_protocol)
                or actual_path != expected_path
                or (metadata_adapter or {}).get("weights_sha256") != expected_hash
            ):
                return False, f"evaluation identity mismatch for {candidate.candidate_id}"
        return ok, detail
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return False, str(exc)


def selected_suites(value: str) -> tuple[str, ...]:
    return BENCHMARKS if value == "all" else (value,)


def task_rows(
    caption_root: Path,
    output_root: Path,
    suite_selection: str,
    size: str,
    candidates: Iterable[Candidate] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates if candidates is not None else discover_candidates(caption_root):
        if candidate.status != "ready":
            continue
        for suite in selected_suites(suite_selection):
            complete, detail = completion_status(output_root, suite, size, candidate.candidate_id, candidate)
            if complete:
                continue
            rows.append({
                "priority": SUITE_PRIORITY[suite] + (20 if candidate.family == "qwen" else 0),
                "suite": suite, "candidate_id": candidate.candidate_id,
                "family": candidate.family, "python": candidate.python,
                "model": candidate.model, "adapter": candidate.adapter,
                "attn": candidate.attn,
                "output": str(output_dir(output_root, suite, size, candidate.candidate_id)),
                "previous_status": detail,
            })
    return sorted(rows, key=lambda row: (-row["priority"], row["candidate_id"]))


def historical_entries(caption_root: Path) -> list[dict[str, Any]]:
    benchmark_root = caption_root / "benchmark"
    original_qwen = caption_root / "DualISL_Train/runs/dual_recursive_8gpu_h100_20260829_run01"
    original_mida = caption_root / "DualISL_Train/runs/dual_recursive_4gpu_h100_midasheng_20260830_run01"
    entries: list[dict[str, Any]] = []
    for trajectory, family, prefix, run in (
        ("qwen_v1", "qwen", "qwen8", original_qwen),
        ("midasheng_v1", "midasheng", "midasheng4", original_mida),
    ):
        labels = [("base", -1, None)] + [
            (f"r{number}", number, run / f"round_{number:03d}/checkpoints/caption_final")
            for number in range(3)
        ]
        for label, round_number, adapter in labels:
            old_name = f"{prefix}_round{round_number}" if round_number >= 0 else f"{family}_base"
            para_name = f"{prefix}_r{round_number}" if round_number >= 0 else f"{family}_base"
            style_date = "20260831" if family == "midasheng" and round_number == 2 else "20260901"
            paths = {
                "emotiontalk": benchmark_root / "emotiontalk_speech_captioning/runs/eight_models_20260901_run01/full" / old_name,
                "paraspeechcaps": benchmark_root / "round_completion_eval_runs_20260831/paraspeechcaps_content_scheme_a_v1" / f"eval_{para_name}_full_20260831_run01",
                "stylecap": benchmark_root / "stylecap_promptspeech_mcq/runs" / f"{old_name}_full_{style_date}_run01",
            }
            entries.append({"trajectory": trajectory, "family": family, "label": label,
                            "round": round_number, "adapter": str(adapter.resolve()) if adapter else "",
                            "model": str((caption_root / "models" / (
                                "Qwen3-Omni-30B-A3B-Captioner" if family == "qwen" else "MiDashengLM-7B-1021-BF16"
                            )).resolve()),
                            "paths": {key: str(value) for key, value in paths.items()}})
    return entries


def _adapter_from_metadata(suite: str, root: Path) -> dict[str, Any] | None:
    return _result_identity(suite, root)["adapter"]


def _result_identity(suite: str, root: Path) -> dict[str, Any]:
    if suite == "emotiontalk":
        identity = read_json(root / "run_metadata.json").get("identity", {})
        return {"adapter": identity.get("adapter"), "model": identity.get("model_path"),
                "backend": identity.get("backend"), "attn": identity.get("attn_backend"),
                "protocol": None}
    if suite == "paraspeechcaps":
        metadata = read_json(root / "outputs/run_metadata.json")
        identity = metadata.get("evaluation_identity", {})
        return {"adapter": metadata.get("adapter"), "model": metadata.get("model_dir"),
                "backend": metadata.get("backend"), "attn": identity.get("attn_backend"),
                "protocol": metadata.get("protocol")}
    identity = read_json(root / "run_metadata.json").get("identity", {})
    return {"adapter": identity.get("adapter"), "model": identity.get("model_dir"),
            # Legacy/current MiDasheng StyleCap identity omitted the backend;
            # qwen3 is always explicit, so absence unambiguously means MiDasheng.
            "backend": identity.get("backend") or "midasheng", "attn": identity.get("attn_backend"),
            "protocol": identity.get("protocol")}


def validate_historical(caption_root: Path) -> list[str]:
    errors: list[str] = []
    for entry in historical_entries(caption_root):
        for suite, value in entry["paths"].items():
            root = Path(value)
            try:
                identity = _result_identity(suite, root)
                metadata_adapter = identity["adapter"]
            except Exception as exc:
                errors.append(f"{entry['trajectory']} {entry['label']} {suite}: missing/invalid result metadata: {exc}")
                continue
            expected = entry["adapter"]
            if Path(str(identity["model"])).resolve() != Path(entry["model"]).resolve():
                errors.append(f"{entry['trajectory']} {entry['label']} {suite}: base model path mismatch")
            expected_backend = "qwen3_omni" if entry["family"] == "qwen" and suite == "emotiontalk" else (
                "qwen3" if entry["family"] == "qwen" else "midasheng"
            )
            expected_attn = "flash_attention_2" if entry["family"] == "qwen" else "sdpa"
            expected_protocol = {
                "paraspeechcaps": "paraspeechcaps-attr6-content-scheme-a-v1",
                "stylecap": "stylecap-promptspeech-speaker-open-mcq-v1",
            }.get(suite)
            if identity["backend"] != expected_backend or identity["attn"] != expected_attn:
                errors.append(f"{entry['trajectory']} {entry['label']} {suite}: backend/attention identity mismatch")
            if expected_protocol is not None and identity["protocol"] != expected_protocol:
                errors.append(f"{entry['trajectory']} {entry['label']} {suite}: protocol identity mismatch")
            if not expected:
                if metadata_adapter is not None:
                    errors.append(f"{entry['trajectory']} base {suite}: result unexpectedly uses an adapter")
                continue
            ok, weights_hash = adapter_check(Path(expected), Path(metadata_adapter.get("base_model_name_or_path", "")))
            if not ok:
                errors.append(f"{entry['trajectory']} {entry['label']} {suite}: {weights_hash}")
            elif Path(metadata_adapter.get("path", "")).resolve() != Path(expected).resolve():
                errors.append(f"{entry['trajectory']} {entry['label']} {suite}: adapter path mismatch")
            elif metadata_adapter.get("weights_sha256") != weights_hash:
                errors.append(f"{entry['trajectory']} {entry['label']} {suite}: adapter hash mismatch")
    return errors


def _metric_scalar(metric: Any, metric_name: str | None = None) -> Any:
    if not isinstance(metric, dict) or metric.get("status", "ok") != "ok":
        return None
    preferred_keys = {
        "bleu_4": ("bleu_4",),
        "rouge_l": ("rouge_l",),
        "meteor": ("meteor",),
        "spider": ("spider",),
        "fense": ("fense",),
        "bertscore": ("bert_score.f1",),
        "clapscore": ("clap_score", "clapscore"),
    }.get(metric_name or "", ())
    for container in (metric.get("corpus"), metric):
        if isinstance(container, dict):
            for key in preferred_keys:
                value = container.get(key)
                if isinstance(value, (int, float)):
                    return value
            for key, value in container.items():
                if key != "status" and isinstance(value, (int, float)):
                    return value
    return None


def load_suite_result(suite: str, root: Path) -> dict[str, Any]:
    if suite == "emotiontalk":
        scores = read_json(root / "metrics_standard_public/scores.json")
        compact_tasks = {}
        for task, task_data in scores.get("tasks", {}).items():
            compact_tasks[task] = {
                "count": task_data.get("count"),
                "metrics": {
                    name: {
                        "status": data.get("status", "ok") if isinstance(data, dict) else "ok",
                        "value": _metric_scalar(data, name),
                        "error_type": data.get("error_type") if isinstance(data, dict) else None,
                        "error": data.get("error") if isinstance(data, dict) else None,
                    }
                    for name, data in task_data.get("metrics", {}).items()
                    if name != "aac_metrics_version"
                },
            }
        return {"profile": scores.get("profile"), "tasks": compact_tasks}
    if suite == "paraspeechcaps":
        return read_json(root / "reports/summary.json")
    return read_json(root / "evaluation_summary.json")


def summary_entries(caption_root: Path, output_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for old in ([] if reward_profile_selected() else historical_entries(caption_root)):
        for suite, path in old["paths"].items():
            root = Path(path)
            try:
                result, state = load_suite_result(suite, root), "complete"
            except Exception as exc:
                result, state = {"error": str(exc)}, "invalid"
            entries.append({**{k: old[k] for k in ("trajectory", "family", "label", "round")},
                            "suite": suite, "source": "historical", "state": state,
                            "result_path": str(root), "result": result})
    for candidate in discover_candidates(caption_root):
        if candidate.status == "pending":
            continue
        for suite in BENCHMARKS:
            root = output_dir(output_root, suite, "full", candidate.candidate_id)
            complete, detail = completion_status(output_root, suite, "full", candidate.candidate_id, candidate)
            if complete:
                result, state = load_suite_result(suite, root), "complete"
            else:
                result, state = {"detail": detail}, "missing" if candidate.status == "ready" else "invalid"
            entries.append({"trajectory": candidate.trajectory, "family": candidate.family,
                            "label": candidate.label, "round": candidate.round, "suite": suite,
                            "source": "later", "state": state, "result_path": str(root), "result": result})
    # RewardV2 shares the same MiDasheng base result; include it explicitly in that trajectory.
    for entry in list(entries):
        if entry["trajectory"] == "midasheng_v1" and entry["label"] == "base":
            clone = dict(entry)
            clone["trajectory"] = "midasheng_rewardv2"
            entries.append(clone)
    return sorted(entries, key=lambda row: (row["trajectory"], row["round"], row["suite"]))


def markdown_value(entry: dict[str, Any]) -> str:
    if entry["state"] != "complete":
        return entry["state"]
    result, suite = entry["result"], entry["suite"]
    if suite == "paraspeechcaps":
        return f"{result.get('final_score', 0):.6f}"
    if suite == "stylecap":
        return f"{result.get('macro_average_accuracy', 0):.6f}"
    overall = result.get("tasks", {}).get("overall", {}).get("metrics", {})
    values = [f"{name}={data['value']:.4g}" for name, data in overall.items() if data.get("value") is not None]
    unavailable = [name for name, data in overall.items() if data.get("status") != "ok"]
    suffix = f"; unavailable={','.join(unavailable)}" if unavailable else ""
    return ", ".join(values) + suffix


def write_summaries(caption_root: Path, output_root: Path) -> Path:
    errors = [] if reward_profile_selected() else validate_historical(caption_root)
    if errors:
        raise RuntimeError("historical result validation failed:\n" + "\n".join(errors))
    entries = summary_entries(caption_root, output_root)
    summary_root = output_root / (
        f"summary_{os.environ['LATER_CAPTION_PROFILE']}" if reward_profile_selected() else "summary"
    )
    summary_root.mkdir(parents=True, exist_ok=True)
    best_effort_shared_chmod(output_root, 0o2777)
    best_effort_shared_chmod(summary_root, 0o2777)
    payload = {"created_utc": utc_now(), "output_root": str(output_root), "entries": entries}
    json_path = summary_root / "results.json"
    csv_path = summary_root / "results.csv"
    markdown_path = summary_root / "results.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trajectory", "label", "round", "suite", "source", "state", "value", "result_path"])
        writer.writeheader()
        for entry in entries:
            writer.writerow({**{key: entry[key] for key in ("trajectory", "label", "round", "suite", "source", "state", "result_path")},
                             "value": markdown_value(entry)})
    lines = ["# Later Captioner checkpoint benchmark summary", "", f"Generated: `{payload['created_utc']}`", ""]
    for trajectory in dict.fromkeys(entry["trajectory"] for entry in entries):
        lines.extend([f"## {trajectory}", "", "| checkpoint | EmotionTalk (overall) | ParaSpeechCaps Scheme A | StyleCap MCQ |", "|---|---:|---:|---:|"])
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for entry in entries:
            if entry["trajectory"] == trajectory:
                grouped.setdefault(entry["label"], {})[entry["suite"]] = entry
        labels = sorted(grouped, key=lambda label: -1 if label == "base" else int(label[1:]))
        for label in labels:
            row = grouped[label]
            lines.append(f"| {label} | {markdown_value(row['emotiontalk'])} | {markdown_value(row['paraspeechcaps'])} | {markdown_value(row['stylecap'])} |")
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    for path in (json_path, csv_path, markdown_path):
        best_effort_shared_chmod(path, 0o666)
    return summary_root


def inventory_payload(caption_root: Path) -> dict[str, Any]:
    candidates = discover_candidates(caption_root)
    return {
        "created_utc": utc_now(),
        "counts": {state: sum(candidate.status == state for candidate in candidates) for state in ("ready", "pending", "invalid")},
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def print_inventory(caption_root: Path, as_json: bool) -> int:
    payload = inventory_payload(caption_root)
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("STATUS   CANDIDATE                   TRAJECTORY            ADAPTER / DETAIL")
        for candidate in payload["candidates"]:
            value = candidate["adapter"] if candidate["status"] == "ready" else candidate["detail"]
            print(f"{candidate['status']:<8} {candidate['candidate_id']:<27} {candidate['trajectory']:<21} {value}")
        print("counts:", ", ".join(f"{key}={value}" for key, value in payload["counts"].items()))
    return 1 if payload["counts"]["invalid"] else 0


def main(argv: list[str] | None = None) -> int:
    default_caption_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-root", type=Path, default=default_caption_root)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--json", action="store_true")
    tasks = sub.add_parser("tasks")
    tasks.add_argument("--output-root", type=Path, required=True)
    tasks.add_argument("--size", choices=("smoke", "full"), required=True)
    tasks.add_argument("--suite", choices=("all",) + BENCHMARKS, required=True)
    tasks.add_argument("--require-smoke", action="store_true")
    tasks.add_argument("--inventory-file", type=Path,
                       help="Use this frozen inventory instead of rescanning checkpoints.")
    verify = sub.add_parser("verify-task")
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--size", choices=("smoke", "full"), required=True)
    verify.add_argument("--suite", choices=BENCHMARKS, required=True)
    verify.add_argument("--candidate", required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--historical", action="store_true")
    args = parser.parse_args(argv)
    caption_root = args.caption_root.resolve()

    if args.command == "inventory":
        return print_inventory(caption_root, args.json)
    if args.command == "validate":
        payload = inventory_payload(caption_root)
        errors = [candidate["detail"] for candidate in payload["candidates"] if candidate["status"] == "invalid"]
        if args.historical and not reward_profile_selected():
            errors.extend(validate_historical(caption_root))
        if errors:
            print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
            return 1
        print(f"validated {payload['counts']['ready']} ready checkpoints; {payload['counts']['pending']} pending")
        return 0
    if args.command == "tasks":
        candidates = None
        if args.inventory_file is not None:
            frozen = read_json(args.inventory_file)
            candidates = [Candidate(**value) for value in frozen.get("candidates", [])]
            invalid = [candidate for candidate in candidates if candidate.status == "invalid"]
            if invalid:
                print("frozen inventory contains invalid candidates", file=sys.stderr)
                return 1
        if args.require_smoke:
            missing = []
            for candidate in candidates if candidates is not None else discover_candidates(caption_root):
                if candidate.status == "ready":
                    for suite in selected_suites(args.suite):
                        if not completion_status(args.output_root, suite, "smoke", candidate.candidate_id, candidate)[0]:
                            missing.append(f"{suite}/{candidate.candidate_id}")
            if missing:
                print("Full evaluation requires completed smoke tasks: " + ", ".join(missing), file=sys.stderr)
                return 1
        rows = task_rows(caption_root, args.output_root, args.suite, args.size, candidates)
        for row in rows:
            print("\t".join(str(row[key]) for key in ("priority", "suite", "candidate_id", "family", "python", "model", "adapter", "attn", "output")))
        return 0
    if args.command == "verify-task":
        # The queue snapshot already performed the expensive directory/commit
        # hash gate. Post-task verification only needs the current adapter file
        # identity recorded by the benchmark runner.
        candidates = {
            candidate.candidate_id: candidate
            for candidate in discover_candidates(
                caption_root, verify_reward_hash=False, verify_adapter_hash=False
            )
        }
        candidate = candidates.get(args.candidate)
        if candidate is None or candidate.status != "ready":
            print(f"candidate is not ready: {args.candidate}", file=sys.stderr)
            return 1
        complete, detail = completion_status(args.output_root, args.suite, args.size, args.candidate, candidate)
        print(detail)
        return 0 if complete else 1
    if args.command == "summarize":
        summary_root = write_summaries(caption_root, args.output_root.resolve())
        print(summary_root)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
