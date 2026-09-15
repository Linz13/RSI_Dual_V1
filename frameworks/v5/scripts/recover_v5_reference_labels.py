"""Recover a stopped V5 run before its first rollout; remote API calls only.

Preserve successful caches. Re-request schema-invalid labels with explicit
schema feedback, using the same evaluator and strict validation. Permit only
the already reviewed worker hotfix when upgrading an old run's source lock.
This standalone recovery does not change the running training/evaluator code.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from dual_isl_train.constants import FRAMEWORK_VERSION, RUN_STATE_VERSION
from dual_isl_train.data import load_records
from dual_isl_train.io import atomic_json, hash_path, load_yaml, sha256_file, stable_hash
from dual_isl_train.labeling import LabelService, MODEL_FIELDS, evaluator_resources
from scripts.migrate_v5_smoke_hotfix import (
    LEGACY_IMPLEMENTATION_HASH, FIXED_IMPLEMENTATION_HASH, implementation_hash, require,
)

RECOVERY_ID = "v5-reference-label-schema-feedback-20260910"
SCHEMA_FEEDBACK = (
    "Validation feedback: a previous response used values outside the supplied field schemas. "
    "Every enum is an exhaustive list, not examples. Use only its exact strings. "
    "For nonverbal_vocalization, if a clearly audible vocal event has no suitable listed category, "
    "use the allowed label other; do not invent an event name. Keep any other correctly identified "
    "listed events. Use unknown only if the audio evidence is uncertain. "
    "The sentinel arrays must be exactly [\"none\"] or [\"unknown\"] and cannot be mixed with events. "
    "Listen to the same audio again and return all requested fields, with no explanation."
)


def audit_run(root):
    root = Path(root).resolve()
    require(implementation_hash() == FIXED_IMPLEMENTATION_HASH, "Unreviewed implementation; recovery needs a new audit")
    cfg = load_yaml(root / "resolved_config.yaml")
    require(cfg.get("version") == 5 and Path(cfg["run"]["output_dir"]).resolve() == root, "Wrong resolved V5 config")
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text())
    require(state.get("version") == RUN_STATE_VERSION and state.get("framework_version") == FRAMEWORK_VERSION,
            "Wrong V5 run state version")
    public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    old_hash = stable_hash({"config": public, "implementation_hash": LEGACY_IMPLEMENTATION_HASH})
    new_hash = stable_hash({"config": public, "implementation_hash": FIXED_IMPLEMENTATION_HASH})
    require(state.get("config_hash") in (old_hash, new_hash), "Run config/source lock does not match either reviewed version")
    require(not state.get("current") and not (root / "latest.json").exists(), "Run already committed a round")
    require(set(state["stages"]) == {"audio_pool_codecs", "paired_anchor_codecs"},
            "Recovery supports only the initial reference-label stage, before any rollout")
    for name, entry in state["stages"].items():
        require(entry["status"] == "complete", "Codec stage is not complete: " + name)
        require(sha256_file(entry["input_path"]) == entry["input_sha256"], "Changed codec input: " + name)
        require(sha256_file(entry["output_path"]) == entry["output_sha256"], "Changed codec output: " + name)
        require(hash_path(entry.get("checkpoint_path")) == entry.get("checkpoint_sha256"), "Changed codec checkpoint")
    require(not any(json.loads(p.read_text()).get("status") == "running" for p in root.glob("round_*/**/*.stage.json")),
            "A GPU stage is marked running; recover only after the run has exited")
    require(cfg["labeling"].get("frozen_resources") == evaluator_resources(cfg["labeling"]), "Frozen evaluators changed")
    return cfg, state, {"state_sha256": sha256_file(state_path),
                        "from_config_hash": state["config_hash"], "to_config_hash": new_hash}


def missing_references(service, rows):
    missing = []
    for row in rows:
        key = service.audio_key(row["audio_path"])
        for model in MODEL_FIELDS:
            cache = service.cache / "remote" / f"{key}.{model}.json"
            if cache.is_file():
                continue
            evidence = sorted((service.cache / "invalid").glob(f"{key}.{model}.*.json"))
            require(bool(evidence), "Missing cache has no schema-error evidence: " + key + ":" + model)
            missing.append({"id": row["id"], "audio_path": row["audio_path"], "key": key, "model": model,
                            "invalid_evidence": [str(p) for p in evidence]})
    return missing


def repair_one(service, item, backup):
    key, model = item["key"], item["model"]
    target = service.cache / "remote" / f"{key}.{model}.json"
    if target.is_file():
        return {"id": item["id"], "model": model, "key": key, "status": "already_cached"}
    for filename in item["invalid_evidence"]:
        source = Path(filename)
        copied = backup / "invalid" / source.name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
    original_request = service._request_with_metadata

    def request_with_feedback(request_model, path, prompt):
        amended = prompt + "\n\n" + SCHEMA_FEEDBACK
        raw, metadata = original_request(request_model, path, amended)
        return raw, {**metadata, "schema_recovery": {
            "id": RECOVERY_ID, "supplement": SCHEMA_FEEDBACK,
            "prompt_sha256": stable_hash(amended), "source_sample_id": item["id"],
        }}

    service._request_with_metadata = request_with_feedback
    try:
        # The production request/identity/stream checks and field validator remain authoritative.
        service._remote(item["audio_path"], key, model)
    finally:
        service._request_with_metadata = original_request
    return {"id": item["id"], "model": model, "key": key, "status": "repaired",
            "cache_path": str(target), "cache_sha256": sha256_file(target)}


def finish_recovery(root, state, audit, backup, repaired):
    state_path = root / "run_state.json"
    require(sha256_file(state_path) == audit["state_sha256"], "Run state changed during recovery; source lock left untouched")
    record = {"id": RECOVERY_ID, "applied_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Retry invalid reference labels with schema feedback; migrate reviewed workers before any rollout/update",
              "backup_path": str(backup), **audit, "repaired": repaired}
    updated = deepcopy(state)
    updated["config_hash"] = audit["to_config_hash"]
    if updated["config_hash"] != state["config_hash"]:
        updated["implementation_migrations"] = [*state.get("implementation_migrations", []), record]
    atomic_json(backup / "recovery.json", record)
    atomic_json(state_path, updated)
    return record


def recover(root, apply=False):
    root = Path(root).resolve()
    cfg, state, audit = audit_run(root)
    rows = load_records(cfg["data"]["audio_only_path"], {"audio_only", "paired"}, "train")
    limit = int(cfg["data"].get("max_records_per_role", 0))
    if limit:
        rows = rows[:limit]
    service = LabelService(cfg)
    try:
        missing = missing_references(service, rows)
        if not apply:
            return {"dry_run": True, "missing": missing, "source_records": len(rows), **audit}
        backup = root / ("reference_label_recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        backup.mkdir()
        shutil.copy2(root / "run_state.json", backup / "run_state.json")
        shutil.copy2(root / "resolved_config.yaml", backup / "resolved_config.yaml")
        repaired = [repair_one(service, item, backup) for item in missing]
        require(not missing_references(service, rows), "Some reference API labels are still missing")
        result = finish_recovery(root, state, audit, backup, repaired)
        return {"ok": True, "source_records": len(rows), "remote_cache_entries": len(rows) * len(MODEL_FIELDS),
                "report": str(backup / "recovery.json"), **result}
    finally:
        service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="Repair via remote APIs and migrate the stopped run")
    args = parser.parse_args()
    os.umask(0)
    os.environ["DUALISL_SHARED_WRITABLE"] = "1"
    print(json.dumps(recover(args.run_dir, args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
