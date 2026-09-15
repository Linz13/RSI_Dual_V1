from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fill_missing_transcriptions_whisper.py"
SPEC = importlib.util.spec_from_file_location("fill_missing_transcriptions_whisper", SCRIPT)
assert SPEC and SPEC.loader
WHISPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WHISPER)


def _row(sample_id: str, transcription: str, language: str):
    return {
        "sample_id": sample_id,
        "Target_JSON_Schema": {"semantic_content": {
            "transcription": transcription, "language": language,
        }},
    }


def test_all_scope_includes_audio_only_and_preserves_existing_text():
    rows = [
        _row("paired", "", "English"),
        _row("caption", "Original.", "other"),
        _row("audio", "", "Chinese"),
        _row("complete", "Keep me.", "English"),
    ]
    roles = {"paired": "paired", "caption": "caption_only", "audio": "audio_only", "complete": "audio_only"}
    all_tasks = WHISPER.task_rows(rows, roles, "all")
    training_tasks = WHISPER.task_rows(rows, roles, "training-required")
    assert [task[0]["sample_id"] for task in all_tasks] == ["paired", "caption", "audio"]
    assert [task[0]["sample_id"] for task in training_tasks] == ["paired", "caption"]
    assert WHISPER.semantic(rows[-1])["transcription"] == "Keep me."

