"""Move a stopped pre-rollout run to this release, importing only equivalent labels.

No GPU execution or API requests. Originals and both states/configs are retained.
The running legacy eight-GPU smoke is intentionally outside this migration.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import yaml

from dual_isl_train.constants import FRAMEWORK_VERSION, RUN_STATE_VERSION
from dual_isl_train.io import atomic_json, dump_yaml, hash_path, load_yaml, sha256_file, stable_hash
from dual_isl_train.labeling import LabelService, MODEL_FIELDS, evaluator_resources
from dual_isl_train.label_normalization import VERSION as POLICY, parse_fields

ROOT = Path(__file__).resolve().parents[1]
LEGACY_IMPLEMENTATION = "52bc080f78aadcda51a2e4bec8d9ff1ebe4f2d564bad77a60887338d278d81d9"
MIGRATION_ID = "v5-evaluator-fields-v52-pre-rollout-20260910"


def atomic_yaml(path, value):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)
    try:
        temporary.chmod(0o666)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def implementation_hash(root):
    pkg = root / "dual_isl_train"
    files = sorted(pkg.rglob("*.py")) + sorted(pkg.rglob("*.json"))
    files += sorted((root / "scripts").glob("*captioner_candidate.py"))
    return stable_hash([{"path": str(p.relative_to(root)), "sha256": sha256_file(p)} for p in files])


def audit_run(run, legacy_root):
    cfg_path, state_path = run / "resolved_config.yaml", run / "run_state.json"
    cfg = load_yaml(cfg_path)
    state = json.loads(state_path.read_text())
    require(state.get("version") == RUN_STATE_VERSION and state.get("framework_version") == FRAMEWORK_VERSION,
            "Not a compatible V5 run state")
    require(cfg.get("version") == 5 and Path(cfg["run"]["output_dir"]).resolve() == run, "Not this V5 run/config")
    require(not state.get("current") and not (run / "latest.json").exists(), "Only uncommitted round 0 can be migrated")
    require(set(state["stages"]) == {"audio_pool_codecs", "paired_anchor_codecs"},
            "Only the stopped pre-rollout reference-label stage is supported")
    require(implementation_hash(legacy_root) == LEGACY_IMPLEMENTATION, "Legacy source differs from reviewed worker hotfix")
    old_hash = stable_hash({"config": {k:v for k,v in cfg.items() if not k.startswith("_")},
                            "implementation_hash": LEGACY_IMPLEMENTATION})
    require(state["config_hash"] == old_hash, "Run has a different source/config lock")
    require(not any(json.loads(p.read_text()).get("status") == "running" for p in run.glob("round_*/**/*.stage.json")),
            "A stage is still running")
    for name, item in state["stages"].items():
        require(item["status"] == "complete", "Incomplete codec stage: " + name)
        require(sha256_file(item["input_path"]) == item["input_sha256"], "Codec input changed")
        require(sha256_file(item["output_path"]) == item["output_sha256"], "Codec output changed")
        require(hash_path(item.get("checkpoint_path")) == item.get("checkpoint_sha256"), "Codec checkpoint changed")
    require(cfg["labeling"]["frozen_resources"] == evaluator_resources(cfg["labeling"]), "Frozen external evaluators changed")
    return cfg, state, {"state_sha256": sha256_file(state_path), "config_sha256": sha256_file(cfg_path)}


def legacy_identity(cfg, legacy_root, cfg_path):
    code = ("from dual_isl_train.io import load_yaml; from dual_isl_train.labeling import LabelService; "
            "import sys; s=LabelService(load_yaml(sys.argv[1])); print(s.identity); s.close()")
    env = dict(os.environ, PYTHONPATH=str(legacy_root), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([cfg["critics"]["python"], "-c", code, str(cfg_path)], cwd=legacy_root,
                            env=env, capture_output=True, text=True, check=True)
    identity = result.stdout.strip()
    require(len(identity) == 64 and all(c in "0123456789abcdef" for c in identity), "Invalid legacy cache identity")
    return identity


def convert_remote(record, filename, old_identity, new_identity):
    """Successful old labels are retained only if this parser produces exactly those labels."""
    key, model, ext = filename.name.split(".")
    require(model in MODEL_FIELDS and ext == "json", "Unexpected remote cache name")
    require(record.get("identity") == old_identity, "Cache identity mismatch")
    sha = record["audio_sha256"]
    require(key == stable_hash({"audio_sha256": sha, "identity": old_identity}), "Cache key/content identity mismatch")
    expected_model = "gemini-3.1-pro-preview" if model == "gemini" else "qwen3.5-omni-plus"
    meta = record["response_metadata"]
    returned = meta.get("model_version" if model == "gemini" else "model")
    require(record["requested_model"] == expected_model and returned == expected_model, "Wrong cached evaluator model")
    require(meta.get("finish_reasons") and all(r.casefold() == "stop" for r in meta["finish_reasons"]),
            "Incomplete cached evaluator response")
    require(model != "qwen35" or meta.get("stream_done"), "Incomplete cached evaluator stream")
    parsed = parse_fields(record["raw_text"], MODEL_FIELDS[model])
    require(not parsed["field_errors"], "Cached raw label needs reevaluation under the new field policy")
    require(parsed["attributes"] == record["attributes"], "Cached label would change under the new policy")
    new_key = stable_hash({"audio_sha256": sha, "identity": new_identity})
    return new_key + "." + model + ".json", {
        **record, "identity": new_identity, "output_policy_version": POLICY,
        "normalizations": parsed["normalizations"], "unknown_fallback_fields": [],
        "cache_migration": {"id": MIGRATION_ID, "source_path": str(filename),
                            "source_sha256": sha256_file(filename), "source_identity": old_identity},
    }


def migrate(run, legacy_root, apply=False):
    run, legacy_root = Path(run).resolve(), Path(legacy_root).resolve()
    cfg, state, snapshot = audit_run(run, legacy_root)
    cfg_path, state_path = run / "resolved_config.yaml", run / "run_state.json"
    old_identity = legacy_identity(cfg, legacy_root, cfg_path)
    old_cache = Path(cfg["labeling"]["cache_dir"])
    new_cfg = deepcopy(cfg)
    new_cache = run / "label_cache_fields_v52"
    require(not new_cache.exists(), "New cache already exists; refusing to overwrite an earlier migration")
    new_cfg["labeling"]["cache_dir"] = str(new_cache)
    # Avoid creating any run output during a dry audit; constructor identity calculation
    # uses the desired cache path but its mkdir is the only initialization mutation.
    service = LabelService(new_cfg)
    try:
        new_identity = service.identity
    finally:
        service.close()
        new_cache.rmdir()
    converted, rejected = [], []
    for path in sorted((old_cache / "remote").glob("*.json")):
        try:
            converted.append(convert_remote(json.loads(path.read_text()), path, old_identity, new_identity))
        except (ValueError, KeyError, RuntimeError) as exc:
            rejected.append({"path": str(path), "reason": type(exc).__name__})
    # This migration is a reuse-only transition. Different/malformed caches need a new audit.
    require(not rejected, "Some existing caches are not equivalent: " + json.dumps(rejected))
    new_impl = implementation_hash(ROOT)
    new_hash = stable_hash({"config": {k:v for k,v in new_cfg.items() if not k.startswith("_")},
                            "implementation_hash": new_impl})
    report = {"id": MIGRATION_ID, "run_dir": str(run), "old_source_root": str(legacy_root),
              "new_source_root": str(ROOT), "old_identity": old_identity, "new_identity": new_identity,
              "from_config_hash": state["config_hash"], "to_config_hash": new_hash,
              "from_implementation_hash": LEGACY_IMPLEMENTATION, "to_implementation_hash": new_impl,
              "remote_caches_preserved": len(converted), "new_cache_dir": str(new_cache),
              "gpu_execution": "not_performed", "api_requests": 0, **snapshot}
    if not apply:
        return {**report, "dry_run": True}
    backup = run / ("label_rules_migration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    backup.mkdir()
    shutil.copy2(state_path, backup / "run_state.json")
    shutil.copy2(cfg_path, backup / "resolved_config.yaml")
    # Import remote-only results; GPU expert tasks restart from their stage. All
    # old remote, local, complete-label and judge caches remain in the original tree.
    for name, record in converted:
        atomic_json(new_cache / "remote" / name, record)
    require(sha256_file(state_path) == snapshot["state_sha256"] and sha256_file(cfg_path) == snapshot["config_sha256"],
            "Run changed during migration; config and state left untouched")
    updated = deepcopy(state)
    updated["config_hash"] = new_hash
    record = {**report, "backup_path": str(backup), "applied_at": datetime.now(timezone.utc).isoformat()}
    updated["implementation_migrations"] = [*state.get("implementation_migrations", []), record]
    atomic_json(backup / "migration.json", record)
    atomic_yaml(cfg_path, new_cfg)
    atomic_json(state_path, updated)
    return {**record, "applied": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    os.umask(0)
    os.environ["DUALISL_SHARED_WRITABLE"] = "1"
    print(json.dumps(migrate(args.run_dir, args.legacy_root, args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
