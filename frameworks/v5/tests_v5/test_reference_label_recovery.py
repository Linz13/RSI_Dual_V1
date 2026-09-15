import json
from pathlib import Path
import shutil

import pytest

from dual_isl_train.attribute_reward import EvaluationPending
from dual_isl_train.constants import FRAMEWORK_VERSION, RUN_STATE_VERSION
from dual_isl_train.io import atomic_json, dump_yaml, hash_path, sha256_file, stable_hash
from dual_isl_train.schema import set_path
from scripts import recover_v5_reference_labels as recovery
from tests_v5.test_integration import service, remote_caption


@pytest.fixture(autouse=True)
def reviewed_test_tree(monkeypatch):
    # Exercise historical migration logic with a fixture source identity. The
    # production tool intentionally rejects this newer release's source tree.
    monkeypatch.setattr(recovery, "FIXED_IMPLEMENTATION_HASH", recovery.implementation_hash())


@pytest.mark.parametrize("succeeds", [True, False])
def test_schema_repair_uses_model_response_and_preserves_evidence(tmp_path, succeeds):
    svc = service(tmp_path / "cache", lambda *args: "unused")
    svc.cache.mkdir()
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    key = svc.audio_key(audio)
    invalid = svc.cache / "invalid" / f"{key}.qwen35.0.json"
    atomic_json(invalid, {"original_evidence": True})
    item = {"id": "fixture", "audio_path": str(audio), "key": key, "model": "qwen35", "invalid_evidence": [str(invalid)]}
    calls = []
    def request(model, path, prompt):
        calls.append(prompt)
        value = remote_caption(model)
        set_path(value, "paralinguistic.nonverbal_vocalization",
                 ["sigh", "other"] if succeeds and len(calls) == 2 else ["unrecognized_string"])
        return (json.dumps(value) if succeeds else "{"), {
            "model": "qwen3.5-omni-plus", "finish_reasons": ["stop"], "stream_done": True}
    svc._request_with_metadata = request
    before = svc._request_with_metadata
    backup = tmp_path / "backup"
    target = svc.cache / "remote" / f"{key}.qwen35.json"
    try:
        if succeeds:
            result = recovery.repair_one(svc, item, backup)
            assert result["status"] == "repaired"
            cache = json.loads(target.read_text())
            assert cache["attributes"]["paralinguistic"]["nonverbal_vocalization"] == ["sigh", "other"]
            assert cache["response_metadata"]["schema_recovery"]["id"] == recovery.RECOVERY_ID
            assert recovery.repair_one(svc, item, backup)["status"] == "already_cached"
        else:
            with pytest.raises(EvaluationPending):
                recovery.repair_one(svc, item, backup)
            assert not target.exists()  # No invented label, unknown, or zero reward on failure.
        assert len(calls) == 2
        assert all(recovery.SCHEMA_FEEDBACK in prompt for prompt in calls)
        assert svc._request_with_metadata is before
        assert json.loads((backup / "invalid" / invalid.name).read_text()) == {"original_evidence": True}
    finally:
        svc.close()


def legacy_run(root, monkeypatch):
    frozen = {"fixture": True}
    monkeypatch.setattr(recovery, "evaluator_resources", lambda cfg: frozen)
    cfg = {"version": 5, "run": {"output_dir": str(root)}, "labeling": {"frozen_resources": frozen}}
    dump_yaml(root / "resolved_config.yaml", cfg)
    stages = {}
    for name in ("audio_pool_codecs", "paired_anchor_codecs"):
        inp, out = root / (name + ".input"), root / (name + ".output")
        inp.write_text("fixture input")
        out.write_text("fixture output")
        stages[name] = {"status": "complete", "input_path": str(inp), "output_path": str(out),
                        "input_sha256": sha256_file(inp), "output_sha256": sha256_file(out),
                        "checkpoint_path": None, "checkpoint_sha256": hash_path(None)}
    state = {"version": RUN_STATE_VERSION, "framework_version": FRAMEWORK_VERSION, "current": {}, "stages": stages,
             "config_hash": stable_hash({"config": cfg, "implementation_hash": recovery.LEGACY_IMPLEMENTATION_HASH})}
    atomic_json(root / "run_state.json", state)
    return cfg, state


def test_pre_rollout_migration_preserves_stages_and_regular_source_guard(tmp_path, monkeypatch):
    cfg, before = legacy_run(tmp_path, monkeypatch)
    _, state, audit = recovery.audit_run(tmp_path)
    backup = tmp_path / "backup"
    backup.mkdir()
    shutil.copy2(tmp_path / "run_state.json", backup / "run_state.json")
    recovery.finish_recovery(tmp_path, state, audit, backup, [])
    from dual_isl_train.stages import StageManager
    after = StageManager(tmp_path, cfg).state
    assert after["stages"] == before["stages"]
    assert after["current"] == before["current"]
    assert json.loads((backup / "run_state.json").read_text()) == before
    assert recovery.audit_run(tmp_path)[2]["from_config_hash"] == after["config_hash"]


@pytest.mark.parametrize("change", ["config", "stage", "artifact", "during_repair", "resources"])
def test_pre_rollout_migration_refuses_changed_state(tmp_path, monkeypatch, change):
    cfg, state = legacy_run(tmp_path, monkeypatch)
    _, _, audit = recovery.audit_run(tmp_path)
    if change == "during_repair":
        state["unexpected"] = True
        atomic_json(tmp_path / "run_state.json", state)
        with pytest.raises(RuntimeError, match="state changed"):
            recovery.finish_recovery(tmp_path, state, audit, tmp_path / "backup", [])
        return
    if change == "config":
        cfg["version"] = 6
        dump_yaml(tmp_path / "resolved_config.yaml", cfg)
    elif change == "stage":
        state["stages"]["round_000_audio_caption_rollout"] = {"status": "complete"}
        atomic_json(tmp_path / "run_state.json", state)
    elif change == "resources":
        monkeypatch.setattr(recovery, "evaluator_resources", lambda cfg: {"fixture": False})
    else:
        Path(state["stages"]["audio_pool_codecs"]["output_path"]).write_text("changed")
    with pytest.raises(RuntimeError):
        recovery.audit_run(tmp_path)
