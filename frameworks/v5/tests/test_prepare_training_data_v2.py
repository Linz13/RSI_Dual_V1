from __future__ import annotations

import importlib.util
import json
import wave
from copy import deepcopy
from pathlib import Path

from dual_isl_train.data import load_records


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_training_data_v2.py"
SPEC = importlib.util.spec_from_file_location("prepare_training_data_v2", SCRIPT)
assert SPEC and SPEC.loader
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def _wav(path: Path, seconds: float = 2.0) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))


def test_v2_builder_merges_transcripts_repairs_labels_and_keeps_relative_paths(tmp_path, sample_caption):
    source = tmp_path / "training_data"
    audio = source / "audio"
    audio.mkdir(parents=True)
    _wav(audio / "paired.wav")
    _wav(audio / "audio.wav", seconds=2.1)
    paired_caption = deepcopy(sample_caption)
    paired_caption["semantic_content"].pop("transcript")
    caption_only_caption = deepcopy(paired_caption)
    caption_only_caption["paralinguistic"].update(emotion="neutral", emotion_intensity="high")
    PREPARE.write_jsonl(source / "paired.jsonl", [{
        "id": "paired", "group_id": "paired", "modality": "paired", "split": "train",
        "audio_path": "old/server/paired.wav", "caption": paired_caption,
    }])
    PREPARE.write_jsonl(source / "audio_only.jsonl", [{
        "id": "audio", "group_id": "audio", "modality": "audio_only", "split": "train",
        "audio_path": "old/server/audio.wav",
    }])
    PREPARE.write_jsonl(source / "caption_only.jsonl", [{
        "id": "caption", "group_id": "caption", "modality": "caption_only", "split": "train",
        "caption": caption_only_caption,
    }])
    transcripts = tmp_path / "original.jsonl"
    PREPARE.write_jsonl(transcripts, [
        {"id": "paired", "transcript": "Original paired words."},
        {"id": "caption", "transcript": "Original caption words."},
    ])
    output = tmp_path / "training_data_v2"
    report = PREPARE.build(source, output, [transcripts], force=False)
    assert report["output_counts"] == {"paired": 1, "audio_only": 1, "caption_only": 1}
    assert report["repairs"]["neutral_intensity"] == 1
    paired = load_records(output / "paired.jsonl")[0]
    caption_only = load_records(output / "caption_only.jsonl")[0]
    smoke_paired = load_records(output / "smoke/paired.jsonl")[0]
    smoke_audio_only = load_records(output / "smoke/audio_only.jsonl")[0]
    assert paired["audio_path"] == str((output / "audio/paired.wav").resolve())
    assert smoke_paired["audio_path"] == str((output / "audio/paired.wav").resolve())
    assert smoke_audio_only["audio_path"] == str((output / "audio/audio.wav").resolve())
    assert Path(smoke_paired["audio_path"]).is_file()
    assert Path(smoke_audio_only["audio_path"]).is_file()
    assert paired["caption"]["semantic_content"]["transcript"] == "Original paired words."
    assert caption_only["caption"]["semantic_content"]["transcript"] == "Original caption words."
    assert caption_only["caption"]["paralinguistic"]["emotion_intensity"] == "low"
    assert "environment" not in paired["caption"]
    assert report["smoke_counts"] == {"paired": 1, "audio_only": 1, "caption_only": 1}
    assert report["smoke_cross_role_validation"]["ok"] is True


def test_transcript_loader_accepts_nested_transcription_and_preserves_first_source(tmp_path):
    original = tmp_path / "original.jsonl"
    fallback = tmp_path / "fallback.jsonl"
    PREPARE.write_jsonl(original, [{
        "sample_id": "sample", "Target_JSON_Schema": {
            "semantic_content": {"transcription": "Original words.", "language": "English"},
        },
    }])
    PREPARE.write_jsonl(fallback, [{
        "sample_id": "sample", "Target_JSON_Schema": {
            "semantic_content": {"transcription": "Whisper words.", "language": "English"},
        },
    }])
    lookup, conflicts, language_fallbacks = PREPARE.load_transcripts([original, fallback])
    assert lookup["sample"]["transcript"] == "Original words."
    assert lookup["sample"]["language"] == "English"
    assert conflicts == [{"key": "sample", "kept": str(original.resolve()), "ignored": str(fallback.resolve())}]
    assert language_fallbacks == []


def test_transcript_loader_uses_later_source_only_for_missing_language(tmp_path):
    qwen = tmp_path / "qwen.jsonl"
    whisper = tmp_path / "whisper.jsonl"
    PREPARE.write_jsonl(qwen, [{
        "sample_id": "sample", "Target_JSON_Schema": {
            "semantic_content": {"transcription": "Qwen text。", "language": "other"},
        },
    }])
    PREPARE.write_jsonl(whisper, [{
        "sample_id": "sample", "Target_JSON_Schema": {
            "semantic_content": {"transcription": "Whisper text", "language": "Chinese"},
        },
    }])
    lookup, conflicts, language_fallbacks = PREPARE.load_transcripts([qwen, whisper])
    assert lookup["sample"]["transcript"] == "Qwen text。"
    assert lookup["sample"]["language"] == "Chinese"
    assert conflicts == [{"key": "sample", "kept": str(qwen.resolve()), "ignored": str(whisper.resolve())}]
    assert language_fallbacks == [{
        "key": "sample", "language": "Chinese", "source": str(whisper.resolve()),
    }]
