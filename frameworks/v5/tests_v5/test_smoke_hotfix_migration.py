import json

import pytest

from dual_isl_train.constants import FRAMEWORK_VERSION, RUN_STATE_VERSION
from dual_isl_train.io import atomic_json, dump_yaml, hash_path, sha256_file, stable_hash
from scripts import migrate_v5_smoke_hotfix as hotfix


@pytest.fixture(autouse=True)
def reviewed_test_tree(monkeypatch):
    monkeypatch.setattr(hotfix, "FIXED_IMPLEMENTATION_HASH", hotfix.implementation_hash())


def stopped_run(root):
    cfg = {"version": 5, "run": {"output_dir": str(root)}}
    dump_yaml(root / "resolved_config.yaml", cfg)
    stages = {}
    for name in ("round_000_audio_caption_rollout", "round_000_caption_grpo", hotfix.FAILED_STAGE,
                 "round_000_caption_tts_rollout"):
        inp = root / "round_000/collections" / (name + ".input.jsonl")
        out = inp.with_name(name + ".output.jsonl")
        inp.parent.mkdir(parents=True, exist_ok=True)
        inp.write_text('{"fixture":"input"}\n')
        out.write_text('{"fixture":"output"}\n')
        failed = name == hotfix.FAILED_STAGE
        stages[name] = {"input_path": str(inp), "input_sha256": sha256_file(inp),
                        "status": "failed" if failed else "complete"}
        if not failed:
            stages[name].update(output_path=str(out), output_sha256=sha256_file(out),
                                checkpoint_path=None, checkpoint_sha256=hash_path(None))
        atomic_json(inp.with_name(name + ".stage.json"), {"status": stages[name]["status"]})
    checkpoint = root / "round_000/checkpoints/caption_after_grpo"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"fixture skipped checkpoint")
    stages["round_000_caption_grpo"].update(checkpoint_path=str(checkpoint), checkpoint_sha256=hash_path(checkpoint))
    (root / "logs").mkdir()
    (root / "logs" / (hotfix.FAILED_STAGE + ".log")).write_text("Captioner SFT gradient connectivity failed: fixture")
    state = {"version": RUN_STATE_VERSION, "framework_version": FRAMEWORK_VERSION, "current": {},
             "config_hash": stable_hash({"config": cfg, "implementation_hash": hotfix.LEGACY_IMPLEMENTATION_HASH}),
             "stages": stages}
    atomic_json(root / "run_state.json", state)
    return cfg, state


def test_hotfix_preserves_unaffected_stage_and_archives_invalidated_outputs(tmp_path):
    cfg, before = stopped_run(tmp_path)
    state, report = hotfix.audit(tmp_path)
    hotfix.apply_migration(state, report)
    after = json.loads((tmp_path / "run_state.json").read_text())
    assert set(after["stages"]) == {"round_000_caption_tts_rollout"}
    assert after["stages"]["round_000_caption_tts_rollout"] == before["stages"]["round_000_caption_tts_rollout"]
    backup = tmp_path / "hotfix_backup_20260910"
    assert json.loads((backup / "run_state.json").read_text()) == before
    assert (backup / "round_000/checkpoints/caption_after_grpo/weights.bin").exists()
    assert not (tmp_path / "round_000/checkpoints/caption_after_grpo").exists()
    from dual_isl_train.stages import StageManager
    assert StageManager(tmp_path, cfg).state == after  # Normal hash guard still applies.
    assert hotfix.audit(tmp_path)[1]["already_applied"]


@pytest.mark.parametrize("change", ["config", "implementation", "committed", "artifact", "running"])
def test_hotfix_refuses_unreviewed_or_live_state(tmp_path, monkeypatch, change):
    cfg, state = stopped_run(tmp_path)
    if change == "config":
        cfg["version"] = 6
        dump_yaml(tmp_path / "resolved_config.yaml", cfg)
    elif change == "implementation":
        monkeypatch.setattr(hotfix, "implementation_hash", lambda: "unreviewed")
    elif change == "committed":
        state["current"] = {"round": 0}
        atomic_json(tmp_path / "run_state.json", state)
    elif change == "artifact":
        from pathlib import Path
        Path(state["stages"]["round_000_caption_tts_rollout"]["output_path"]).write_text("changed")
    else:
        atomic_json(tmp_path / "round_000/running.stage.json", {"status": "running"})
    with pytest.raises(RuntimeError):
        hotfix.audit(tmp_path)
    assert not (tmp_path / "hotfix_backup_20260910").exists()
