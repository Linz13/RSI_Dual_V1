from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dual_isl_train.config import load_config, public_config
from dual_isl_train.constants import FRAMEWORK_VERSION, RUN_STATE_VERSION, TRAJECTORY_VERSION
from dual_isl_train.io import atomic_json, dump_yaml, hash_path, stable_hash


LEGACY_IMPLEMENTATION_HASH = "8f5baca196947609a0b73986bffaab8586bba437b267d5c51d9076f219deb1a7"


def test_retired_parser_hotfix_migration_rejects_later_implementations(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    run_dir = tmp_path / "run"
    monkeypatch.setenv("DUALISL_RUN_DIR", str(run_dir))
    config_path = project_root / "configs" / "train_8gpu_h100.yaml"
    config = load_config(config_path)
    public = public_config(config)
    dump_yaml(run_dir / "resolved_config.yaml", public)

    caption_checkpoint = run_dir / "round_000" / "checkpoints" / "caption_final"
    tts_checkpoint = run_dir / "round_000" / "checkpoints" / "tts_final"
    caption_checkpoint.mkdir(parents=True)
    tts_checkpoint.mkdir(parents=True)
    (caption_checkpoint / "adapter.bin").write_bytes(b"caption")
    (tts_checkpoint / "adapter.bin").write_bytes(b"tts")
    commit = {
        "framework_version": FRAMEWORK_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "round": 0,
        "same_round_start": True,
        "input_captioner": None,
        "input_tts": None,
        "captioner": {"path": str(caption_checkpoint), "sha256": hash_path(caption_checkpoint)},
        "tts": {"path": str(tts_checkpoint), "sha256": hash_path(tts_checkpoint)},
    }
    atomic_json(run_dir / "round_000" / "commit.json", commit)
    atomic_json(run_dir / "latest.json", commit)

    legacy_config_hash = stable_hash({
        "config": public,
        "implementation_hash": LEGACY_IMPLEMENTATION_HASH,
    })
    state = {
        "version": RUN_STATE_VERSION,
        "framework_version": FRAMEWORK_VERSION,
        "config_hash": legacy_config_hash,
        "stages": {
            "round_001_audio_caption_rollout": {
                "status": "failed",
                "error": "CalledProcessError('caption rollout failed')",
            },
        },
        "current": {
            "round": 0,
            "caption_checkpoint": str(caption_checkpoint),
            "caption_checkpoint_sha256": hash_path(caption_checkpoint),
            "tts_checkpoint": str(tts_checkpoint),
            "tts_checkpoint_sha256": hash_path(tts_checkpoint),
        },
    }
    atomic_json(run_dir / "run_state.json", state)
    metadata = run_dir / "round_001" / "collections" / "round_001_audio_caption_rollout.stage.json"
    atomic_json(metadata, {"status": "failed"})
    log = run_dir / "logs" / "round_001_audio_caption_rollout.log"
    log.parent.mkdir(parents=True)
    log.write_text("TypeError: unhashable type: 'list'\n", encoding="utf-8")

    command = [
        sys.executable,
        str(project_root / "scripts" / "migrate_qwen_caption_parser_hotfix.py"),
        "--run-dir",
        str(run_dir),
        "--config",
        str(config_path),
    ]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    refused = subprocess.run(
        command, cwd=project_root, env=environment, check=False, capture_output=True, text=True,
    )
    assert refused.returncode != 0
    assert "not the exact reviewed parser hotfix" in refused.stderr
    assert json.loads((run_dir / "run_state.json").read_text()) == state
    assert not (run_dir / "run_state.pre_caption_parser_hotfix.json").exists()
    assert not (run_dir / "implementation_migration.caption_parser_hotfix.json").exists()
