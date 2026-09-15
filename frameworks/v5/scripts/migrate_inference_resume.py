"""Resume a stopped, committed LabelRobust run with inference-only changes.

Keep committed artifacts byte-for-byte; archive unfinished next-round work and
re-key equivalent evaluator caches for the new source directory. No GPU/API work.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil

from dual_isl_train.checkpoints import checkpoint_record
from dual_isl_train.io import atomic_json, hash_path, load_yaml, sha256_file, stable_hash
from dual_isl_train.labeling import LabelService, MODEL_FIELDS, RUBRIC, evaluator_resources
from dual_isl_train.label_normalization import VERSION as POLICY, parse_fields, finish_fields
from dual_isl_train.dual_space import SYNTHESIZABLE_FIELDS
from dual_isl_train.schema import get_path
from dual_isl_train.stages import StageManager
from scripts.migrate_label_rules import atomic_yaml, implementation_hash, legacy_identity, require

ROOT = Path(__file__).resolve().parents[1]
LEGACY_IMPLEMENTATION = "cd2d9b74410d56d31dcbeb46f2a3cb781956a711b91a97ca62664235dc6b6765"
MIGRATION_ID = "v5-inference-only-resume-20260910-v1"
ALLOWED_CORE_CHANGES = {"dual_isl_train/stages.py", "dual_isl_train/workers/tts_batch.py",
                        "dual_isl_train/workers/qwen_voice_design.py"}


def source_audit(legacy_root):
    require(implementation_hash(legacy_root) == LEGACY_IMPLEMENTATION, "Unexpected legacy implementation")
    def files(root):
        return {str(p.relative_to(root)): sha256_file(p)
                for p in (list((root / "dual_isl_train").rglob("*.py")) +
                          list((root / "dual_isl_train").rglob("*.json")) +
                          list((root / "scripts").glob("*captioner_candidate.py")))}
    old, new = files(legacy_root), files(ROOT)
    require(set(old) == set(new), "Unexpected added/removed training or evaluator modules")
    changed = {p for p in old if old[p] != new[p]}
    require(changed == ALLOWED_CORE_CHANGES, "Unreviewed training/evaluator code changes: " + str(changed))
    # These changed files contain training methods too. They must remain intact.
    import ast
    def methods(path):
        tree = ast.parse(path.read_text())
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)}
    rel = Path("dual_isl_train/workers/qwen_voice_design.py")
    a, b = methods(legacy_root / rel), methods(ROOT / rel)
    require(set(a) == set(b) and all(a[k] == b[k] for k in a if k != "main"),
            "TTS generation/training methods changed")
    return {"changed_core_files": sorted(changed), "evaluators_identical": True,
            "grpo_sft_methods_identical": True}


def cache_key(sha, identity):
    require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha), "Invalid audio hash")
    return stable_hash({"audio_sha256": sha, "identity": identity})


def converted_caches(cache, old_identity, new_identity):
    """Only re-key complete records proven equivalent under unchanged evaluators."""
    result, remote, counts = [], {}, {"remote": 0, "labels": 0, "judge": 0}
    for path in sorted((cache / "remote").glob("*.json")):
        record = json.loads(path.read_text())
        old_key, model, ext = path.name.split(".")
        require(ext == "json" and model in MODEL_FIELDS, "Unexpected remote cache filename")
        sha = record["audio_sha256"]
        require(record["identity"] == old_identity and old_key == cache_key(sha, old_identity), "Wrong remote cache identity")
        expected_model = "gemini-3.1-pro-preview" if model == "gemini" else "qwen3.5-omni-plus"
        metadata = record["response_metadata"]
        require(record["requested_model"] == expected_model and
                metadata.get("model_version" if model == "gemini" else "model") == expected_model,
                "Wrong remote cache model")
        require(metadata.get("finish_reasons") and all(v.casefold() == "stop" for v in metadata["finish_reasons"]),
                "Truncated remote cache")
        require(model != "qwen35" or metadata.get("stream_done"), "Incomplete Qwen stream")
        parsed = parse_fields(record["raw_text"], MODEL_FIELDS[model])
        degraded = sorted(parsed["field_errors"])
        require(parsed["valid_fields"] and sorted(record.get("unknown_fallback_fields", [])) == degraded,
                "Cached fallback does not match parsed fields")
        attrs = finish_fields(parsed) if degraded else parsed["attributes"]
        require(record["attributes"] == attrs and record["output_policy_version"] == POLICY,
                "Cached labels would change")
        require(record.get("normalizations") == parsed["normalizations"], "Cached normalizations changed")
        remote[(old_key, model)] = record
        result.append((Path("remote") / (cache_key(sha, new_identity) + f".{model}.json"),
                       {**record, "identity": new_identity}, path))
        counts["remote"] += 1
    for path in sorted((cache / "labels").glob("*.json")):
        record = json.loads(path.read_text())
        sha = record["audio_sha256"]
        require(record["identity"] == old_identity and path.stem == cache_key(sha, old_identity),
                "Wrong complete-label cache identity")
        # Independent evaluators own different fields. Captioner admission also
        # imposes cross-field conditions, which must not alter evaluator caches.
        parsed = parse_fields(json.dumps(record["attributes"], ensure_ascii=False), SYNTHESIZABLE_FIELDS)
        require(not parsed["field_errors"], "Invalid complete-label fields")
        for model, fields in MODEL_FIELDS.items():
            raw = remote.get((path.stem, model))
            require(raw is not None and all(get_path(raw["attributes"], f) == get_path(record["attributes"], f)
                                           for f in fields), "Complete labels disagree with remote records")
        result.append((Path("labels") / (cache_key(sha, new_identity) + ".json"),
                       {**record, "identity": new_identity}, path))
        counts["labels"] += 1
    for path in sorted((cache / "judge").glob("*.json")):
        record = json.loads(path.read_text())
        pairs, scores, judgments = record["pairs"], record["scores"], record["judgments"]
        require(path.stem == stable_hash({"pairs": pairs, "rubric": RUBRIC, "identity": old_identity}),
                "Wrong text-judge cache key")
        require(record["requested_model"] == record["returned_model"] == "gpt-5.5", "Wrong text-judge model")
        ids = [p["id"] for p in pairs]
        require(ids and len(ids) == len(set(ids)) and set(ids) <= {"paralinguistic.pause", "paralinguistic.prosody"},
                "Invalid judge pair IDs")
        require(all(p["field"] == p["id"] for p in pairs), "Wrong judge field")
        require(set(scores) == set(ids) and len(judgments) == len(ids) and
                {r["id"]: r["score"] for r in judgments} == scores and
                all(type(v) in (int, float) and v in (0, 0.5, 1) for v in scores.values()), "Invalid judge scores")
        name = stable_hash({"pairs": pairs, "rubric": RUBRIC, "identity": new_identity}) + ".json"
        result.append((Path("judge") / name, record, path))
        counts["judge"] += 1
    converted = []
    for relative, record, path in result:
        converted.append((relative, {**record, "cache_migration": {
            "id": MIGRATION_ID, "source_path": str(path), "source_sha256": sha256_file(path),
            "source_identity": old_identity}}))
    return converted, counts


def snapshot(run, next_round, cache):
    paths = [run / "run_state.json", run / "resolved_config.yaml", run / "latest.json"]
    paths += list((run / f"round_{next_round:03d}").rglob("*"))
    paths += list((run / "logs").glob(f"round_{next_round:03d}_*.log"))
    for name in ("remote", "labels", "judge"):
        paths += list((cache / name).glob("*.json"))
    return {str(p): sha256_file(p) for p in paths if p.is_file()}


def audit_run(run, legacy_root):
    cfg = load_yaml(run / "resolved_config.yaml")
    state = json.loads((run / "run_state.json").read_text())
    require(cfg.get("version") == 5 and Path(cfg["run"]["output_dir"]).resolve() == run, "Wrong run/config")
    require(state.get("config_hash") == stable_hash({"config": {k:v for k,v in cfg.items() if not k.startswith("_")},
            "implementation_hash": LEGACY_IMPLEMENTATION}), "Legacy state/config lock mismatch")
    current = state.get("current", {})
    require(current.get("round") == 0, "This migration requires exactly one committed round")
    require(cfg["training"]["rounds"] > 1 and cfg["training"].get("round_offset", 0) == 0, "No next round to resume")
    require(cfg["tts"]["generation"]["rollout_batch_size"] == 4, "Unexpected legacy TTS rollout batch")
    require(cfg["labeling"]["frozen_resources"] == evaluator_resources(cfg["labeling"]), "External evaluators changed")
    commit = json.loads((run / "round_000/commit.json").read_text())
    require(commit == json.loads((run / "latest.json").read_text()) and commit["round"] == 0, "Latest commit mismatch")
    summary = json.loads((run / "round_000/summary.json").read_text())
    for role, key in (("captioner", "caption"), ("tts", "tts")):
        record = checkpoint_record(current[key + "_checkpoint"])
        require(record == commit[role] == summary["output_" + role], "Committed checkpoint mismatch: " + role)
        require(record["sha256"] == current[key + "_checkpoint_sha256"], "Current checkpoint hash mismatch")
    checked = {}
    for name, entry in state["stages"].items():
        if name.startswith("round_001_"):
            require(name == "round_001_audio_caption_rollout", "Next round has progressed beyond supported migration point")
            continue
        require(not name.startswith("round_") or name.startswith("round_000_"), "Unexpected later-round stages")
        require(entry["status"] == "complete", "Incomplete committed stage: " + name)
        for kind in ("input", "output"):
            require(sha256_file(entry[kind + "_path"]) == entry[kind + "_sha256"], "Stage artifact changed: " + name)
        path = entry.get("checkpoint_path")
        if path not in checked:
            checked[path] = hash_path(path)
        require(checked[path] == entry.get("checkpoint_sha256"), "Stage checkpoint changed: " + name)
    next_dir = run / "round_001"
    require(not any(p.is_file() for d in ("training", "checkpoints", "rewards") for p in (next_dir / d).rglob("*")),
            "Uncommitted optimizer/reward work requires a separate audit")
    require(not any(p.name not in {"round_000", "round_001"} for p in run.glob("round_[0-9][0-9][0-9]")),
            "Unexpected later-round directories")
    return cfg, state, commit


def migrate(run, legacy_root, *, apply=False, stopped=False):
    require(stopped, "Confirm the GPU job and child processes are stopped with --stopped")
    run, legacy_root = Path(run).resolve(), Path(legacy_root).resolve()
    source_report = source_audit(legacy_root)
    cfg, state, commit = audit_run(run, legacy_root)
    old_cache = Path(cfg["labeling"]["cache_dir"])
    before = snapshot(run, 1, old_cache)
    old_identity = legacy_identity(cfg, legacy_root, run / "resolved_config.yaml")
    new_cfg = deepcopy(cfg)
    new_cfg["tts"]["generation"]["rollout_batch_size"] = 1
    new_cache = run / "label_cache_fast_resume_v1"
    require(not new_cache.exists(), "Target cache already exists; refusing to overwrite it")
    new_cfg["labeling"]["cache_dir"] = str(new_cache)
    service = LabelService(new_cfg)
    try:
        new_identity = service.identity
    finally:
        service.close()
        new_cache.rmdir()
    converted, counts = converted_caches(old_cache, old_identity, new_identity)
    new_impl = implementation_hash(ROOT)
    new_hash = stable_hash({"config": {k:v for k,v in new_cfg.items() if not k.startswith("_")},
                            "implementation_hash": new_impl})
    report = {"id": MIGRATION_ID, "run_dir": str(run), "old_source_root": str(legacy_root),
              "new_source_root": str(ROOT), "committed_round": 0, "resume_round": 1,
              "total_rounds": cfg["training"]["rounds"], "remaining_rounds": cfg["training"]["rounds"] - 1,
              "paired_anchor_enabled": cfg["training"].get("paired_anchor_enabled", True),
              "world_size": cfg["distributed"]["world_size"], "checkpoint_commit": commit,
              "from_implementation_hash": LEGACY_IMPLEMENTATION, "to_implementation_hash": new_impl,
              "from_config_hash": state["config_hash"], "to_config_hash": new_hash,
              "old_identity": old_identity, "new_identity": new_identity,
              "old_cache_dir": str(old_cache), "new_cache_dir": str(new_cache), "caches_preserved": counts,
              "gpu_execution": "not_performed", "api_requests": 0, **source_report}
    require(before == snapshot(run, 1, old_cache), "Stopped run changed during audit")
    if not apply:
        return {**report, "dry_run": True}
    backup = run / ("inference_resume_migration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    backup.mkdir()
    for name in ("resolved_config.yaml", "run_state.json"):
        shutil.copyfile(run / name, backup / name)
        (backup / name).chmod(0o666)
    record = {**report, "backup_path": str(backup), "applied_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(backup / "planned_migration.json", record)
    staging = backup / "prepared_cache"
    staging.mkdir()
    for relative, data in converted:
        atomic_json(staging / relative, data)
    require(before == snapshot(run, 1, old_cache) and implementation_hash(ROOT) == new_impl and
            implementation_hash(legacy_root) == LEGACY_IMPLEMENTATION, "Run/source changed before migration commit")
    updated = deepcopy(state)
    updated["config_hash"] = new_hash
    updated["stages"] = {k:v for k,v in state["stages"].items() if not k.startswith("round_001_")}
    updated["implementation_migrations"] = [*state.get("implementation_migrations", []), record]
    archived = []
    moved_cache = False
    try:
        paths = [run / "round_001", *sorted((run / "logs").glob("round_001_*.log"))]
        for path in paths:
            if path.exists():
                target = backup / "unfinished" / path.relative_to(run)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
                archived.append((path, target))
        os.replace(staging, new_cache)
        moved_cache = True
        atomic_yaml(run / "resolved_config.yaml", new_cfg)
        atomic_json(run / "run_state.json", updated)
        StageManager(run, new_cfg)
        atomic_json(backup / "migration.json", {**record, "applied": True})
    except BaseException:
        atomic_yaml(run / "resolved_config.yaml", cfg)
        atomic_json(run / "run_state.json", state)
        for path, target in reversed(archived):
            os.replace(target, path)
        if moved_cache:
            os.replace(new_cache, staging)
        atomic_json(backup / "rollback.json", {"rolled_back": True})
        raise
    return {**record, "applied": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--stopped", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    os.umask(0)
    os.environ["DUALISL_SHARED_WRITABLE"] = "1"
    print(json.dumps(migrate(args.run_dir, args.legacy_root, apply=args.apply, stopped=args.stopped),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
