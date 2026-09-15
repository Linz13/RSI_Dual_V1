#!/usr/bin/env python3
"""Audit and migrate a stopped Qwen run across the 2026-08-30 parser hotfix.

The normal StageManager intentionally rejects every implementation change.
This one-off tool permits only the known container-type guard change, only
after a committed round 0, and only when round 1 has no completed stage.  It
keeps the failed stage record, writes a pre-migration run-state backup, and
adds an explicit implementation-migration record before resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_isl_train.config import load_config, public_config
from dual_isl_train.io import atomic_json, hash_path, load_yaml, sha256_file, stable_hash


MIGRATION_ID = "qwen-caption-parser-container-type-guard-20260830"
LEGACY_IMPLEMENTATION_HASH = "8f5baca196947609a0b73986bffaab8586bba437b267d5c51d9076f219deb1a7"
FIXED_IMPLEMENTATION_HASH = "d16ce771840cb8cb13ce4e8754581aa39e1a30192770c2d2094a49f77a4e5eba"
FAILED_STAGE = "round_001_audio_caption_rollout"
EXPECTED_ERROR = "TypeError: unhashable type: 'list'"
FIXED_FILE_HASHES = {
    "dual_space.py": "47301034d97128ccdb91dc7064882a106453d92ef48fca8a64664e595f052ca9",
    "schema.py": "3745458b84eaececd746cb8f59357e7a4cd9f23f97de5e1ab9cad245119b8d72",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _implementation_hash(package_root: Path) -> str:
    files = sorted(package_root.rglob("*.py")) + sorted(package_root.rglob("*.json"))
    return stable_hash([
        {"path": str(path.relative_to(package_root)), "sha256": sha256_file(path)}
        for path in files
    ])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit(run_dir: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = Path(__file__).resolve().parent.parent
    package_root = project_root / "dual_isl_train"
    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    state_path = run_dir / "run_state.json"
    resolved_config_path = run_dir / "resolved_config.yaml"
    latest_path = run_dir / "latest.json"
    commit_path = run_dir / "round_000" / "commit.json"
    metadata_path = run_dir / "round_001" / "collections" / f"{FAILED_STAGE}.stage.json"
    log_path = run_dir / "logs" / f"{FAILED_STAGE}.log"

    for path in (state_path, resolved_config_path, latest_path, commit_path, metadata_path, log_path):
        _require(path.is_file(), f"Required migration evidence is missing: {path}")

    current_implementation_hash = _implementation_hash(package_root)
    _require(
        current_implementation_hash == FIXED_IMPLEMENTATION_HASH,
        "The checked-out dual_isl_train implementation is not the exact reviewed parser hotfix "
        f"(expected {FIXED_IMPLEMENTATION_HASH}, found {current_implementation_hash})",
    )
    for relative, expected in FIXED_FILE_HASHES.items():
        actual = sha256_file(package_root / relative)
        _require(actual == expected, f"Unexpected hash for dual_isl_train/{relative}: {actual}")

    os.environ["DUALISL_RUN_DIR"] = str(run_dir)
    config = load_config(config_path)
    public = public_config(config)
    _require(Path(public["run"]["output_dir"]).resolve() == run_dir, "Config output_dir does not match --run-dir")
    _require(
        stable_hash(load_yaml(resolved_config_path)) == stable_hash(public),
        "Resolved config differs from the configuration selected for migration",
    )
    legacy_config_hash = stable_hash({
        "config": public,
        "implementation_hash": LEGACY_IMPLEMENTATION_HASH,
    })
    fixed_config_hash = stable_hash({
        "config": public,
        "implementation_hash": current_implementation_hash,
    })

    state = _read_json(state_path)
    migrations = state.get("implementation_migrations", [])
    already_applied = state.get("config_hash") == fixed_config_hash and any(
        isinstance(item, dict) and item.get("id") == MIGRATION_ID for item in migrations
    )
    if already_applied:
        return state, {
            "ok": True,
            "already_applied": True,
            "migration_id": MIGRATION_ID,
            "run_dir": str(run_dir),
            "config_hash": fixed_config_hash,
            "implementation_hash": current_implementation_hash,
        }

    _require(
        state.get("config_hash") == legacy_config_hash,
        "run_state.json is not locked to the exact legacy implementation and selected config "
        f"(expected {legacy_config_hash}, found {state.get('config_hash')})",
    )
    current = state.get("current", {})
    _require(current.get("round") == 0, "Migration requires round 0 to be the latest committed round")

    commit = _read_json(commit_path)
    latest = _read_json(latest_path)
    _require(commit == latest and commit.get("round") == 0, "round_000/commit.json and latest.json must match")
    for role, state_key in (("captioner", "caption_checkpoint"), ("tts", "tts_checkpoint")):
        record = commit.get(role)
        _require(isinstance(record, dict), f"Round-0 commit is missing the {role} checkpoint record")
        _require(current.get(state_key) == record.get("path"), f"Current {role} checkpoint path differs from commit")
        actual_hash = hash_path(record.get("path"))
        _require(actual_hash == record.get("sha256"), f"Committed {role} checkpoint hash mismatch")
        _require(
            current.get(f"{state_key}_sha256") == actual_hash,
            f"run_state current {role} checkpoint hash mismatch",
        )

    round_one_stages = {
        name: entry for name, entry in state.get("stages", {}).items()
        if name.startswith("round_001_")
    }
    completed = sorted(name for name, entry in round_one_stages.items() if entry.get("status") == "complete")
    _require(not completed, f"Round 1 already has completed stages and cannot use this migration: {completed}")
    _require(set(round_one_stages).issubset({FAILED_STAGE}), f"Unexpected round-1 stage records: {sorted(round_one_stages)}")
    failed = round_one_stages.get(FAILED_STAGE, {})
    _require(failed.get("status") == "failed", f"{FAILED_STAGE} is not recorded as failed")

    metadata = _read_json(metadata_path)
    _require(metadata.get("status") == "failed", f"{metadata_path} is not a failed stage record")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    _require(EXPECTED_ERROR in log_text, f"Stage log does not contain the expected root error: {EXPECTED_ERROR}")

    record = {
        "id": MIGRATION_ID,
        "scope": "Type guards for model-emitted container values in caption validation; no training rule changed",
        "failed_stage": FAILED_STAGE,
        "root_error": EXPECTED_ERROR,
        "from_config_hash": legacy_config_hash,
        "to_config_hash": fixed_config_hash,
        "from_implementation_hash": LEGACY_IMPLEMENTATION_HASH,
        "to_implementation_hash": current_implementation_hash,
        "fixed_file_sha256": FIXED_FILE_HASHES,
        "round_0_commit_sha256": sha256_file(commit_path),
    }
    report = {
        "ok": True,
        "already_applied": False,
        "migration_id": MIGRATION_ID,
        "run_dir": str(run_dir),
        "legacy_config_hash": legacy_config_hash,
        "fixed_config_hash": fixed_config_hash,
        "legacy_implementation_hash": LEGACY_IMPLEMENTATION_HASH,
        "fixed_implementation_hash": current_implementation_hash,
        "round_1_completed_stages": completed,
        "record": record,
    }
    return state, report


def apply_migration(run_dir: Path, state: dict[str, Any], report: dict[str, Any]) -> None:
    if report.get("already_applied"):
        return
    state_path = run_dir / "run_state.json"
    backup_path = run_dir / "run_state.pre_caption_parser_hotfix.json"
    audit_path = run_dir / "implementation_migration.caption_parser_hotfix.json"
    _require(not backup_path.exists(), f"Refusing to overwrite existing backup: {backup_path}")
    atomic_json(backup_path, state)

    applied_at = datetime.now(timezone.utc).isoformat()
    migration_record = {**report["record"], "applied_at": applied_at, "backup_path": str(backup_path)}
    migrated_state = dict(state)
    migrated_state["config_hash"] = report["fixed_config_hash"]
    migrated_state["implementation_migrations"] = [
        *state.get("implementation_migrations", []), migration_record,
    ]
    atomic_json(audit_path, migration_record)
    atomic_json(state_path, migrated_state)
    report.update({
        "applied": True,
        "applied_at": applied_at,
        "backup_path": str(backup_path),
        "audit_path": str(audit_path),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_8gpu_h100.yaml")
    parser.add_argument("--apply", action="store_true", help="Write the audited migration; default is dry-run only")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    state, report = audit(run_dir, args.config)
    if args.apply:
        apply_migration(run_dir, state, report)
    else:
        report["applied"] = bool(report.get("already_applied"))
        report["dry_run"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
