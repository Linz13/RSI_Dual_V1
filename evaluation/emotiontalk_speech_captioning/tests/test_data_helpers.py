from __future__ import annotations

import csv
import io
import json
import tarfile

import pytest

from scripts.prepare_benchmark import (
    CAPTION_KEYS,
    load_text_metadata,
    load_transcripts,
    parse_caption_content,
    resolve_speaker,
)
from validate_benchmark import split_matches


def test_caption_parser_preserves_all_original_strings() -> None:
    source = {key: f" 原始 {key}。 " for key in CAPTION_KEYS}
    parsed = parse_caption_content(repr(source))
    assert parsed == source
    assert [parsed[f"caption_{index}"] for index in range(1, 6)] == [source[f"caption_{index}"] for index in range(1, 6)]


def test_caption_parser_rejects_code_and_missing_fields() -> None:
    with pytest.raises((ValueError, SyntaxError)):
        parse_caption_content("__import__('os').system('false')")
    with pytest.raises(ValueError):
        parse_caption_content("{'spe_cap': 'x'}")


def test_transcript_join_uses_utterance_stem(tmp_path) -> None:
    path = tmp_path / "transcription.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "emotion", "chinese"])
        writer.writeheader()
        writer.writerow({"name": "G00003/a/b/G00003_01_13_002", "emotion": "neutral", "chinese": "仅供分析"})
    assert load_transcripts(path) == {"G00003_01_13_002": "仅供分析"}


def test_text_tar_official_speaker_and_speaker_resolution(tmp_path) -> None:
    path = tmp_path / "Text.tar"
    payload = json.dumps({"speaker_id": "13", "content": "文本"}).encode()
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("Text/json/G00003/G00003_01/G00003_01_13/G00003_01_13_002.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    metadata = load_text_metadata(path, {"G00003_01_13_002"})
    assert resolve_speaker("G00003_01_13_002", metadata["G00003_01_13_002"]) == ("13", "Text.tar:speaker_id")
    with pytest.raises(ValueError):
        resolve_speaker("G00003_01_13_002", {"speaker_id": "20"})


def test_split_equality_requires_official_order_and_set() -> None:
    official = ["a", "b", "c"]
    assert split_matches(["a", "b", "c"], official)
    assert not split_matches(["b", "a", "c"], official)
    assert not split_matches(["a", "b", "d"], official)
