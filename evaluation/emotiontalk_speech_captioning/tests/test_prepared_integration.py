from __future__ import annotations

import json

import pytest

from common import ROOT, TASKS, read_jsonl


@pytest.mark.integration
def test_prepared_manifests_exact_counts_and_isolation() -> None:
    references_path = ROOT / "data/test_references.jsonl"
    inference_path = ROOT / "data/test_inference.jsonl"
    if not references_path.exists() or not inference_path.exists():
        pytest.skip("prepared data not present")
    references = read_jsonl(references_path)
    inference = read_jsonl(inference_path)
    assert len(references) == 1929
    assert len(inference) == 1929 * 4
    assert len({row["id"] for row in references}) == 1929
    assert all(len(row["overall_captions"]) == 5 for row in references)
    forbidden = {"transcript", "speaker_caption", "style_caption", "emotion_caption", "overall_captions"}
    assert all(not (set(row) & forbidden) for row in inference)
    assert all({item["task"] for item in inference[index:index + 4]} == set(TASKS) for index in range(0, len(inference), 4))
