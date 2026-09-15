"""Audit/migrate the stopped round-0 smoke across two exact, reviewed fixes.

Only the known failed Captioner SFT state before any TTS update is accepted.
Archive affected audio-only collection and Captioner updates; retain codecs,
label caches, caption-only collection, and its frozen calibration. No GPUs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from dual_isl_train.constants import FRAMEWORK_VERSION, RUN_STATE_VERSION
from dual_isl_train.io import atomic_json, hash_path, load_yaml, sha256_file, stable_hash

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_ID = "v5-smoke-sft-padding-and-caption-batch-fallback-20260910"
LEGACY_IMPLEMENTATION_HASH = "ab819fd510e5569d3d377370e1aa43d0b51a6e7a4a148c0fda18504390ec90b8"
FIXED_IMPLEMENTATION_HASH = "52bc080f78aadcda51a2e4bec8d9ff1ebe4f2d564bad77a60887338d278d81d9"
FAILED_STAGE = "round_000_caption_sft"
INVALIDATE = {
    "round_000_audio_caption_rollout", "round_000_audio_attribute_synthesis",
    "round_000_audio_collection", "round_000_audio_rewards",
    "round_000_caption_grpo", FAILED_STAGE,
}


def implementation_hash():
    pkg = ROOT / "dual_isl_train"
    files = sorted(pkg.rglob("*.py")) + sorted(pkg.rglob("*.json"))
    files += sorted((ROOT / "scripts").glob("*captioner_candidate.py"))
    return stable_hash([{"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)} for p in files])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(run_dir):
    root = Path(run_dir).resolve()
    require(implementation_hash() == FIXED_IMPLEMENTATION_HASH, "Implementation is not the exact reviewed hotfix")
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text())
    cfg = {k: v for k, v in load_yaml(root / "resolved_config.yaml").items() if not k.startswith("_")}
    require(Path(cfg["run"]["output_dir"]).resolve() == root, "Resolved config points at another run")
    require(state.get("version") == RUN_STATE_VERSION and state.get("framework_version") == FRAMEWORK_VERSION,
            "Not a compatible V5 run state")
    old = stable_hash({"config": cfg, "implementation_hash": LEGACY_IMPLEMENTATION_HASH})
    new = stable_hash({"config": cfg, "implementation_hash": FIXED_IMPLEMENTATION_HASH})
    previous = [r for r in state.get("implementation_migrations", []) if r.get("id") == MIGRATION_ID]
    if state.get("config_hash") == new and previous:
        return state, {"already_applied": True, "run_dir": str(root), "migration_id": MIGRATION_ID}
    require(state.get("config_hash") == old, "Run is not locked to the exact legacy source/config")
    require(not state.get("current") and not (root / "latest.json").exists(), "Only uncommitted round 0 is supported")
    stages = state.get("stages", {})
    require(not any(n.startswith("round_") and not n.startswith("round_000_") for n in stages), "Later round exists")
    require(not any(n.startswith(("round_000_tts_grpo", "round_000_tts_sft", "round_000_reload")) for n in stages),
            "TTS updates/reload already started; this run needs a separate audit")
    require(stages.get(FAILED_STAGE, {}).get("status") == "failed", "Expected stopped Captioner SFT failure is missing")
    require("Captioner SFT gradient connectivity failed" in (root / "logs" / (FAILED_STAGE + ".log")).read_text(),
            "Captioner SFT log does not contain the known failure")
    require(not any(json.loads(p.read_text()).get("status") == "running"
                    for p in (root / "round_000").rglob("*.stage.json")), "A stage is marked running")
    for name, entry in stages.items():
        require(sha256_file(entry["input_path"]) == entry["input_sha256"], f"Changed stage input: {name}")
        if entry["status"] == "complete":
            require(sha256_file(entry["output_path"]) == entry["output_sha256"], f"Changed stage output: {name}")
            require(hash_path(entry.get("checkpoint_path")) == entry.get("checkpoint_sha256"),
                    f"Changed checkpoint: {name}")
    return state, {
        "already_applied": False, "run_dir": str(root), "migration_id": MIGRATION_ID,
        "from_config_hash": old, "to_config_hash": new,
        "from_implementation_hash": LEGACY_IMPLEMENTATION_HASH,
        "to_implementation_hash": FIXED_IMPLEMENTATION_HASH,
        "state_sha256": sha256_file(state_path),
        "invalidated_stages": sorted(INVALIDATE & stages.keys()),
        "preserved_stages": sorted(stages.keys() - INVALIDATE),
    }


def apply_migration(state, report):
    if report["already_applied"]:
        return report
    root = Path(report["run_dir"])
    state_path = root / "run_state.json"
    backup = root / "hotfix_backup_20260910"
    require(not backup.exists(), "Hotfix backup already exists; refusing to overwrite it")
    require(sha256_file(state_path) == report["state_sha256"], "State changed during audit; stop the run before migrating")
    backup.mkdir()
    shutil.copy2(state_path, backup / "run_state.json")
    shutil.copy2(root / "resolved_config.yaml", backup / "resolved_config.yaml")
    artifacts = set()
    for name in report["invalidated_stages"]:
        artifacts.update(p for p in (root / "round_000").rglob(name + ".*") if p.is_file())
        log = root / "logs" / (name + ".log")
        if log.is_file():
            artifacts.add(log)
    for name in ("caption_after_grpo", "caption_final"):
        path = root / "round_000/checkpoints" / name
        if path.exists():
            artifacts.add(path)
    # Copy before changing state; if a copy fails, the original run is untouched.
    for path in sorted(artifacts):
        target = backup / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    require(sha256_file(state_path) == report["state_sha256"], "State changed while backing up; run state left unchanged")
    record = {**report, "id": MIGRATION_ID, "applied_at": datetime.now(timezone.utc).isoformat(), "backup_path": str(backup),
              "scope": "SFT padding-aware audit; regenerate incompatible batched Captioner trajectories with serial fallback"}
    updated = deepcopy(state)
    updated["config_hash"] = report["to_config_hash"]
    updated["stages"] = {name: entry for name, entry in state["stages"].items() if name not in INVALIDATE}
    updated["implementation_migrations"] = [*state.get("implementation_migrations", []), record]
    atomic_json(backup / "migration.json", record)
    # Archived failed checkpoints/metrics must not be mixed into newly trained outputs.
    for path in sorted(artifacts):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    atomic_json(state_path, updated)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply only after the GPU run has exited")
    args = parser.parse_args()
    state, report = audit(args.run_dir)
    if args.apply:
        report = apply_migration(state, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
