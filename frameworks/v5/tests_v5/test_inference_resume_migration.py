from copy import deepcopy
import json
from pathlib import Path

import pytest

from dual_isl_train.checkpoints import checkpoint_record
from dual_isl_train.constants import FRAMEWORK_VERSION, RUN_STATE_VERSION
from dual_isl_train.io import atomic_json, load_yaml, sha256_file, stable_hash
from dual_isl_train.label_normalization import parse_fields, finish_fields
from dual_isl_train.dual_space import project_synth_caption
from dual_isl_train.workers.mock import mock_caption
from dual_isl_train.schema import get_path, set_path
from scripts import migrate_inference_resume as migration


OLD, NEW, AUDIO = "b" * 64, "c" * 64, "a" * 64


def cache_fixture(cache, degraded=False):
    caption = project_synth_caption(mock_caption(0))
    key = migration.cache_key(AUDIO, OLD)
    for model, fields in migration.MODEL_FIELDS.items():
        attrs = {}
        for field in fields:
            set_path(attrs, field, get_path(caption, field))
        if degraded and model == "qwen35":
            attrs["speaker_profile"]["age"] = "ancient"
        raw = json.dumps(attrs)
        parsed = parse_fields(raw, fields)
        normalized = finish_fields(parsed) if parsed["field_errors"] else parsed["attributes"]
        for field in fields:
            set_path(caption, field, get_path(normalized, field))
        name = "gemini-3.1-pro-preview" if model == "gemini" else "qwen3.5-omni-plus"
        atomic_json(cache / "remote" / f"{key}.{model}.json", {
            "attributes": normalized, "raw_text": raw, "identity": OLD, "audio_sha256": AUDIO,
            "requested_model": name, "output_policy_version": migration.POLICY,
            "normalizations": parsed["normalizations"], "unknown_fallback_fields": sorted(parsed["field_errors"]),
            "response_metadata": {"model": name, "model_version": name, "finish_reasons": ["stop"], "stream_done": True}})
    atomic_json(cache / "labels" / f"{key}.json", {"attributes": caption, "identity": OLD, "audio_sha256": AUDIO})
    pairs = [{"id": "paralinguistic.pause", "field": "paralinguistic.pause", "reference": "brief pause", "candidate": "short pause"}]
    key = stable_hash({"pairs": pairs, "rubric": migration.RUBRIC, "identity": OLD})
    atomic_json(cache / "judge" / f"{key}.json", {"pairs": pairs, "scores": {"paralinguistic.pause": 1},
        "judgments": [{"id": "paralinguistic.pause", "score": 1, "reason": "equivalent"}],
        "requested_model": "gpt-5.5", "returned_model": "gpt-5.5"})


@pytest.mark.parametrize("degraded", [False, True])
def test_complete_and_degraded_labels_survive_rekey_without_changing_values(tmp_path, degraded):
    cache_fixture(tmp_path, degraded)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    records, counts = migration.converted_caches(tmp_path, OLD, NEW)
    assert counts == {"remote": 2, "labels": 1, "judge": 1}
    for relative, record in records:
        src = Path(record["cache_migration"]["source_path"])
        original = json.loads(src.read_text())
        for key in ("attributes", "raw_text", "pairs", "scores", "unknown_fallback_fields"):
            if key in original:
                assert record[key] == original[key]
        assert src.read_bytes() == before[str(src)]
        assert record["cache_migration"]["source_sha256"] == sha256_file(src)
        assert relative.name != src.name


@pytest.mark.parametrize("corruption", ["raw", "identity", "model", "labels", "judge_score", "judge_model"])
def test_corrupt_caches_are_rejected(tmp_path, corruption):
    cache_fixture(tmp_path)
    part = "labels" if corruption == "labels" else "judge" if corruption.startswith("judge") else "remote"
    path = next((tmp_path / part).glob("*.json"))
    data = json.loads(path.read_text())
    if corruption == "raw":
        data["raw_text"] = "{"
    elif corruption == "identity":
        data["identity"] = "wrong"
    elif corruption == "model":
        data["response_metadata"]["finish_reasons"] = ["length"]
    elif corruption == "labels":
        data["attributes"]["speaker_profile"]["gender"] = "unknown"
    elif corruption == "judge_score":
        data["scores"]["paralinguistic.pause"] = 0.123
    else:
        data["returned_model"] = "another-model"
    atomic_json(path, data)
    with pytest.raises((RuntimeError, ValueError)):
        migration.converted_caches(tmp_path, OLD, NEW)


def test_evaluator_fields_are_not_rewritten_by_captioner_admission(tmp_path):
    cache_fixture(tmp_path)
    path = next((tmp_path / "labels").glob("*.json"))
    label = json.loads(path.read_text())
    label["attributes"]["paralinguistic"]["emotion"] = "neutral"
    qwen = next((tmp_path / "remote").glob("*.qwen35.json"))
    remote = json.loads(qwen.read_text())
    attrs = json.loads(remote["raw_text"])
    attrs["paralinguistic"]["emotion_intensity"] = "medium"
    remote["raw_text"] = json.dumps(attrs)
    parsed = parse_fields(remote["raw_text"], migration.MODEL_FIELDS["qwen35"])
    remote["attributes"] = parsed["attributes"]
    remote["normalizations"] = parsed["normalizations"]
    label["attributes"]["paralinguistic"]["emotion_intensity"] = "medium"
    atomic_json(qwen, remote)
    atomic_json(path, label)
    values, _ = migration.converted_caches(tmp_path, OLD, NEW)
    new = next(v for p, v in values if p.parent.name == "labels")
    assert new["attributes"] == label["attributes"]


def fixture_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    legacy = tmp_path / "legacy"
    (legacy / "dual_isl_train").mkdir(parents=True)
    (legacy / "dual_isl_train/__init__.py").write_text("# fixture")
    monkeypatch.setattr(migration, "LEGACY_IMPLEMENTATION", migration.implementation_hash(legacy))
    monkeypatch.setattr(migration, "source_audit", lambda root: {"evaluators_identical": True})
    monkeypatch.setattr(migration, "evaluator_resources", lambda cfg: {"fixture": True})
    monkeypatch.setattr(migration, "legacy_identity", lambda *args: OLD)
    class Service:
        def __init__(self, cfg):
            self.identity = NEW
            Path(cfg["labeling"]["cache_dir"]).mkdir()
        def close(self):
            pass
    monkeypatch.setattr(migration, "LabelService", Service)
    cfg = {"version": 5, "run": {"output_dir": str(run)}, "distributed": {"enabled": True, "world_size": 8},
           "tts": {"generation": {"rollout_batch_size": 4, "synthesis_batch_size": 4}},
           "training": {"rounds": 10, "paired_anchor_enabled": True},
           "labeling": {"cache_dir": str(run / "label_cache"), "frozen_resources": {"fixture": True}}}
    migration.atomic_yaml(run / "resolved_config.yaml", cfg)
    commit, current = {"round": 0}, {"round": 0}
    for role, key in (("captioner", "caption"), ("tts", "tts")):
        ckpt = run / "round_000/checkpoints" / role
        ckpt.mkdir(parents=True)
        (ckpt / "weights").write_text("fixture " + role)
        commit[role] = checkpoint_record(ckpt)
        current[key + "_checkpoint"] = str(ckpt)
        current[key + "_checkpoint_sha256"] = commit[role]["sha256"]
    atomic_json(run / "round_000/commit.json", commit)
    atomic_json(run / "latest.json", commit)
    atomic_json(run / "round_000/summary.json", {"output_captioner": commit["captioner"], "output_tts": commit["tts"]})
    inp, out = run / "round_000/input", run / "round_000/output"
    inp.write_text("input")
    out.write_text("output")
    stages = {"round_000_tts_sft": {"status": "complete", "input_path": str(inp), "output_path": str(out),
              "input_sha256": sha256_file(inp), "output_sha256": sha256_file(out),
              "checkpoint_path": None, "checkpoint_sha256": None}}
    state = {"version": RUN_STATE_VERSION, "framework_version": FRAMEWORK_VERSION, "current": current, "stages": stages,
             "config_hash": stable_hash({"config": cfg, "implementation_hash": migration.LEGACY_IMPLEMENTATION})}
    atomic_json(run / "run_state.json", state)
    interrupted = run / "round_001/collections/interrupted.json"
    atomic_json(interrupted, {"status": "running"})
    (run / "logs").mkdir()
    (run / "logs/round_001_audio_caption_rollout.log").write_text("KeyboardInterrupt")
    cache_fixture(run / "label_cache")
    return run, legacy, cfg, state


def test_committed_checkpoint_and_cache_preservation_and_next_round_archive(tmp_path, monkeypatch):
    run, legacy, cfg, state = fixture_run(tmp_path, monkeypatch)
    committed = {str(p): p.read_bytes() for p in (run / "round_000").rglob("*") if p.is_file()}
    preview = migration.migrate(run, legacy, stopped=True)
    assert preview["dry_run"] and preview["remaining_rounds"] == 9
    assert not (run / "label_cache_fast_resume_v1").exists()
    assert json.loads((run / "run_state.json").read_text()) == state
    report = migration.migrate(run, legacy, stopped=True, apply=True)
    backup = Path(report["backup_path"])
    assert (backup / "unfinished/round_001/collections/interrupted.json").is_file()
    assert not (run / "round_001").exists()
    assert not (run / "logs/round_001_audio_caption_rollout.log").exists()
    new_cfg = load_yaml(run / "resolved_config.yaml")
    assert new_cfg["training"] == cfg["training"]
    assert new_cfg["tts"]["generation"] == {"rollout_batch_size": 1, "synthesis_batch_size": 4}
    manager = migration.StageManager(run, new_cfg)
    assert manager.state["current"] == state["current"]
    assert {path: Path(path).read_bytes() for path in committed} == committed
    assert len(list((run / "label_cache_fast_resume_v1").rglob("*.json"))) == 4
    assert len(list((run / "label_cache").rglob("*.json"))) == 4


def test_failed_state_write_rolls_back_config_archive_and_cache(tmp_path, monkeypatch):
    run, legacy, cfg, state = fixture_run(tmp_path, monkeypatch)
    original = migration.atomic_json
    raised = []
    def fail_once(path, value):
        if Path(path) == run / "run_state.json" and not raised:
            raised.append(True)
            raise OSError("injected state failure")
        return original(path, value)
    monkeypatch.setattr(migration, "atomic_json", fail_once)
    with pytest.raises(OSError, match="injected"):
        migration.migrate(run, legacy, stopped=True, apply=True)
    assert load_yaml(run / "resolved_config.yaml") == cfg
    assert json.loads((run / "run_state.json").read_text()) == state
    assert (run / "round_001/collections/interrupted.json").exists()
    assert (run / "logs/round_001_audio_caption_rollout.log").exists()
    assert not (run / "label_cache_fast_resume_v1").exists()
    assert len(list((run / "label_cache").rglob("*.json"))) == 4


@pytest.mark.parametrize("corruption", ["checkpoint", "current", "latest", "stage", "next_update", "config"])
def test_migration_rejects_inconsistent_run(tmp_path, monkeypatch, corruption):
    run, legacy, cfg, state = fixture_run(tmp_path, monkeypatch)
    if corruption == "checkpoint":
        (run / "round_000/checkpoints/tts/weights").write_text("changed")
    elif corruption == "current":
        state["current"]["round"] = 1
        atomic_json(run / "run_state.json", state)
    elif corruption == "latest":
        atomic_json(run / "latest.json", {"round": 3})
    elif corruption == "stage":
        (run / "round_000/output").write_text("changed")
    elif corruption == "next_update":
        atomic_json(run / "round_001/training/started.json", {})
    else:
        cfg["training"]["rounds"] = 20
        migration.atomic_yaml(run / "resolved_config.yaml", cfg)
    with pytest.raises(RuntimeError):
        migration.migrate(run, legacy, stopped=True)


def test_stopped_confirmation_is_required(tmp_path):
    with pytest.raises(RuntimeError, match="stopped"):
        migration.migrate(tmp_path, tmp_path)
