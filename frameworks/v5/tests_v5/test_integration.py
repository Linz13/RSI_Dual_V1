from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest

from dual_isl_train.labeling import LabelService, MODEL_FIELDS
from dual_isl_train.schema import get_path, set_path
from dual_isl_train.dual_space import project_synth_caption
from dual_isl_train.workers.mock import mock_caption
from dual_isl_train.attribute_reward import EvaluationPending

ROOT = Path(__file__).resolve().parents[1]


def service(tmp_path, remote):
    # Same service logic with an explicitly stubbed transport and local experts.
    svc = LabelService.__new__(LabelService)
    svc.cfg = {"attempts": 2}
    svc.cache = tmp_path
    svc.events = tmp_path / "events.jsonl"
    svc.lock = threading.Lock()
    svc.identity = "test-only"
    svc.pool = ThreadPoolExecutor(2)
    svc.futures = {}
    svc.pipeline = SimpleNamespace(api_request=remote)
    svc._request_with_metadata = lambda model,path,prompt: (remote(model,path,prompt,svc.backend_config), {})
    svc.backend_config = {"api": {"gemini_model": "gemini-3.1-pro-preview", "qwen_model": "qwen3.5-omni-plus"}}
    def local(rows):
        value = project_synth_caption(mock_caption(0))
        fields = ["semantic_content.transcript", "paralinguistic.volume_level", "paralinguistic.emotion",
                  "speaker_profile.accent", "paralinguistic.speaking_rate"]
        return {r["id"]: {f: get_path(value, f) for f in fields} for r in rows}
    svc._local = local
    return svc


def remote_caption(model):
    target, source = {}, project_synth_caption(mock_caption(0))
    for field in MODEL_FIELDS[model]:
        set_path(target, field, get_path(source, field))
    return target


def test_single_source_labeling_cache_and_audio_identity(tmp_path):
    calls = []
    def remote(model, *args):
        calls.append(model)
        value = remote_caption(model)
        if model == "qwen35":
            set_path(value, "speaker_profile.age", "unknown")
        return json.dumps(value)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fixture-one")
    rows = [{"id": "a", "audio_path": str(wav)}]
    svc = service(tmp_path, remote)
    try:
        first = svc.labels(rows)
        second = svc.labels(rows)
        assert first == second
        assert get_path(first["a"], "speaker_profile.age") == "unknown"
        assert sorted(calls) == ["gemini", "qwen35"]
        wav.write_bytes(b"fixture-two")
        svc.labels(rows)
        assert len(calls) == 4
    finally:
        svc.close()


def test_api_error_is_pending_not_unknown_and_resume_retries(tmp_path):
    def broken(*args):
        raise TimeoutError("fixture transport unavailable")
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fixture")
    svc = service(tmp_path, broken)
    try:
        with pytest.raises(EvaluationPending):
            svc.labels([{"id": "a", "audio_path": str(wav)}])
        assert not list(tmp_path.glob("labels/*.json"))
    finally:
        svc.close()
    svc = service(tmp_path, lambda model, *args: json.dumps(remote_caption(model)))
    try:
        assert "a" in svc.labels([{"id": "a", "audio_path": str(wav)}])
    finally:
        svc.close()


def test_malformed_evaluator_json_is_retried(tmp_path):
    count = 0
    def remote(model, *args):
        nonlocal count
        count += 1
        return "{}" if count == 1 else json.dumps(remote_caption(model))
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"fixture")
    svc = service(tmp_path, remote)
    try:
        assert svc._remote(wav, "test", "gemini")
        assert count == 2
    finally:
        svc.close()


def test_evaluator_dotted_keys_are_unambiguous_and_normalized():
    from dual_isl_train.labeling import evaluator_payload
    fields = MODEL_FIELDS["qwen35"]
    source = remote_caption("qwen35")
    flat = {f: get_path(source, f) for f in fields}
    parsed = evaluator_payload(json.dumps(flat), fields)
    assert set(fields).issubset(parsed["valid_fields"])
    conflict = {**flat, "speaker_profile": {"gender": "male"}}
    with pytest.raises(ValueError, match="ambiguous"):
        evaluator_payload(json.dumps(conflict), fields)


def test_current_data_has_one_psyn_anchor_each(tmp_path, monkeypatch):
    from dual_isl_train.config import load_config
    from dual_isl_train.io import read_jsonl
    from dual_isl_train.orchestrator import DualISLOrchestrator
    monkeypatch.setenv("DUALISL_RUN_DIR", str(tmp_path / "run"))
    cfg = load_config(ROOT / "configs/v5_midasheng.yaml")
    paired = list(read_jsonl(cfg["data"]["paired_path"]))
    orch = DualISLOrchestrator(cfg)
    codes = {str(r["id"]): {"codec_path": "unused"} for r in paired}
    assert len(paired) == 39
    assert len(orch._caption_anchors(paired)) == len(orch._tts_anchors(paired, codes)) == 39
    orch.config["training"]["paired_anchor_enabled"] = False
    assert orch._caption_anchors(paired) == orch._tts_anchors(paired, codes) == []


def test_mock_dashboard_verifier_and_partial_resume(tmp_path, monkeypatch):
    from dual_isl_train.config import load_config
    from dual_isl_train.orchestrator import DualISLOrchestrator
    from scripts.export_training_dashboard import scalars
    from scripts.verify_v5 import verify
    monkeypatch.setenv("DUALISL_RUN_DIR", str(tmp_path / "run"))
    cfg = load_config(ROOT / "configs/v5_mock.yaml")
    orch = DualISLOrchestrator(cfg)
    real_update = orch._update_tts
    def fail_once(*args, **kwargs):
        raise RuntimeError("injected interruption before TTS update")
    orch._update_tts = fail_once
    with pytest.raises(RuntimeError, match="injected interruption"):
        orch.train()
    orch._update_tts = real_update
    orch.train(resume_only=True)
    report = verify(tmp_path / "run")
    assert report["ok"], report["errors"]
    values = {tag: value for tag, step, value in scalars(tmp_path / "run")}
    assert "rounds/Captioner_reward/format_reward_mean" in values
    assert "rounds/Captioner_reward/combined_reward_mean" in values
    assert "rounds/Captioner_reward/semantic_reward_mean" in values


def test_resume_ignores_round_and_anchor_shell_changes(tmp_path, monkeypatch):
    from dual_isl_train.config import load_config, public_config
    from dual_isl_train.io import dump_yaml
    from scripts.v5_launcher import runtime_config
    monkeypatch.setenv("DUALISL_RUN_DIR", str(tmp_path))
    cfg = load_config(ROOT / "configs/v5_midasheng_4gpu.yaml")
    dump_yaml(tmp_path / "resolved_config.yaml", public_config(cfg))
    monkeypatch.setenv("DUALISL_ROUNDS", "100")
    monkeypatch.setenv("DUALISL_PAIRED_ANCHOR", "0")
    resumed = runtime_config("resume", tmp_path)
    assert resumed["training"]["rounds"] == cfg["training"]["rounds"]
    assert resumed["training"]["paired_anchor_enabled"] is True
