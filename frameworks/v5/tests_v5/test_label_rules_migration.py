from copy import deepcopy
import json
from pathlib import Path

import pytest

from dual_isl_train.constants import FRAMEWORK_VERSION, RUN_STATE_VERSION
from dual_isl_train.io import atomic_json, dump_yaml, load_yaml, sha256_file, stable_hash
from dual_isl_train.schema import set_path
from scripts import migrate_label_rules as migration
from tests_v5.test_integration import remote_caption


def cache_record(tmp_path):
    audio_sha, identity = "a" * 64, "b" * 64
    key = stable_hash({"audio_sha256": audio_sha, "identity": identity})
    path = tmp_path / (key + ".qwen35.json")
    attrs = remote_caption("qwen35")
    record = {"identity": identity, "audio_sha256": audio_sha,
              "attributes": attrs, "raw_text": json.dumps(attrs), "requested_model": "qwen3.5-omni-plus",
              "response_metadata": {"model": "qwen3.5-omni-plus", "finish_reasons": ["stop"], "stream_done": True}}
    atomic_json(path, record)
    return path, record


def test_remote_conversion_preserves_labels_and_audits_provenance(tmp_path):
    path, record = cache_record(tmp_path)
    before = path.read_bytes()
    name, imported = migration.convert_remote(record, path, "b" * 64, "c" * 64)
    assert name != path.name
    assert imported["attributes"] == record["attributes"]
    assert imported["raw_text"] == record["raw_text"]
    assert imported["cache_migration"]["source_sha256"] == sha256_file(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["identity", "raw", "attrs", "model", "finish", "stream"])
def test_incompatible_remote_cache_is_not_silently_imported(tmp_path, change):
    path, record = cache_record(tmp_path)
    if change == "identity":
        record["identity"] = "different"
    elif change == "raw":
        record["raw_text"] = "{"
    elif change == "attrs":
        record["attributes"]["speaker_profile"]["gender"] = "unknown"
    elif change == "model":
        record["response_metadata"]["model"] = "different"
    elif change == "finish":
        record["response_metadata"]["finish_reasons"] = ["length"]
    else:
        record["response_metadata"]["stream_done"] = False
    with pytest.raises((RuntimeError, ValueError)):
        migration.convert_remote(record, path, "b" * 64, "c" * 64)


def stopped_run(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    legacy = tmp_path / "legacy"
    (legacy / "dual_isl_train").mkdir(parents=True)
    (legacy / "scripts").mkdir()
    (legacy / "dual_isl_train/__init__.py").write_text("# fixture\n")
    legacy_hash = migration.implementation_hash(legacy)
    monkeypatch.setattr(migration, "LEGACY_IMPLEMENTATION", legacy_hash)
    monkeypatch.setattr(migration, "evaluator_resources", lambda cfg: {"fixture": True})
    monkeypatch.setattr(migration, "legacy_identity", lambda *args: "b" * 64)
    class Service:
        def __init__(self, cfg):
            self.identity = "c" * 64
            Path(cfg["labeling"]["cache_dir"]).mkdir()
        def close(self):
            pass
    monkeypatch.setattr(migration, "LabelService", Service)
    cfg = {"version": 5, "run": {"output_dir": str(root)},
           "training": {"rounds": 10, "paired_anchor_enabled": False},
           "labeling": {"cache_dir": str(root / "label_cache"), "frozen_resources": {"fixture": True}}}
    dump_yaml(root / "resolved_config.yaml", cfg)
    stages = {}
    for name in ("audio_pool_codecs", "paired_anchor_codecs"):
        inp, out = root / (name + ".input"), root / (name + ".output")
        inp.write_text("fixture input")
        out.write_text("fixture output")
        stages[name] = {"status": "complete", "input_path": str(inp), "output_path": str(out),
                        "input_sha256": sha256_file(inp), "output_sha256": sha256_file(out),
                        "checkpoint_path": None, "checkpoint_sha256": None}
    state = {"version": RUN_STATE_VERSION, "framework_version": FRAMEWORK_VERSION, "current": {}, "stages": stages,
             "config_hash": stable_hash({"config": cfg, "implementation_hash": legacy_hash})}
    atomic_json(root / "run_state.json", state)
    path, _ = cache_record(root / "label_cache/remote")
    return root, legacy, cfg, state, path


def test_full_migration_keeps_originals_and_resumes_under_new_source(tmp_path, monkeypatch):
    root, legacy, cfg, state, cache_path = stopped_run(tmp_path, monkeypatch)
    cache_before = cache_path.read_bytes()
    preview = migration.migrate(root, legacy)
    assert preview["dry_run"] and preview["remote_caches_preserved"] == 1
    assert not (root / "label_cache_fields_v52").exists()
    assert json.loads((root / "run_state.json").read_text()) == state
    report = migration.migrate(root, legacy, apply=True)
    backup = Path(report["backup_path"])
    assert json.loads((backup / "run_state.json").read_text()) == state
    assert load_yaml(backup / "resolved_config.yaml") == cfg
    new_cfg = load_yaml(root / "resolved_config.yaml")
    assert new_cfg["training"] == cfg["training"]
    assert cache_path.read_bytes() == cache_before
    from dual_isl_train.stages import StageManager
    manager = StageManager(root, new_cfg)
    assert manager.state["stages"] == state["stages"]
    assert len(list((root / "label_cache_fields_v52/remote").glob("*.json"))) == 1
    for name, stage in state["stages"].items():
        assert manager.reusable(name, Path(stage["input_path"]))


@pytest.mark.parametrize("change", ["committed", "progressed", "running", "output", "source", "config"])
def test_run_migration_rejects_unsupported_state(tmp_path, monkeypatch, change):
    root, legacy, cfg, state, _ = stopped_run(tmp_path, monkeypatch)
    if change == "committed":
        state["current"] = {"round": 0}
        atomic_json(root / "run_state.json", state)
    elif change == "progressed":
        state["stages"]["round_000_audio_caption_rollout"] = {"status": "complete"}
        atomic_json(root / "run_state.json", state)
    elif change == "running":
        atomic_json(root / "round_000/collections/running.stage.json", {"status": "running"})
    elif change == "output":
        Path(state["stages"]["audio_pool_codecs"]["output_path"]).write_text("changed")
    elif change == "source":
        (legacy / "dual_isl_train/__init__.py").write_text("changed")
    else:
        cfg["training"]["rounds"] = 20
        dump_yaml(root / "resolved_config.yaml", cfg)
    with pytest.raises(RuntimeError):
        migration.audit_run(root, legacy)
