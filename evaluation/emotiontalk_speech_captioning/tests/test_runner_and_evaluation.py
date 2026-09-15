from __future__ import annotations

import json

import pytest

from backends.base import CaptionBackend, CaptionRequest
from common import ROOT, TASKS
from evaluate import (
    ensure_unique_predictions,
    normalize_metric_text,
    prepare_singleton_inputs,
    trim_bertscore_sentences,
)
from run_inference import (
    generate_batch_resilient,
    select_requests,
    validate_resume_identity,
)
from summarize_eight_models import metric_value


def requests() -> list[dict]:
    return [
        {"id": uid, "task": task, "audio_path": f"audio/{uid}.wav", "prompt": f"prompt-{task}"}
        for uid in ("u1", "u2") for task in TASKS
    ]


def test_max_samples_keeps_four_tasks_for_one_utterance() -> None:
    selected = select_requests(requests(), TASKS, 1)
    assert {row["id"] for row in selected} == {"u1"}
    assert [row["task"] for row in selected] == list(TASKS)
    assert all(set(row) == {"id", "task", "audio_path", "prompt"} for row in selected)


def test_inference_request_has_no_gt_or_transcript() -> None:
    forbidden = {"transcript", "caption", "captions", "reference", "references"}
    assert all(not (set(row) & forbidden) for row in requests())


def test_duplicate_predictions_are_rejected() -> None:
    row = {"id": "u1", "task": "style", "prediction": "x"}
    with pytest.raises(ValueError, match="Duplicate"):
        ensure_unique_predictions([row, dict(row)])


def test_metric_text_replaces_newlines_only() -> None:
    assert normalize_metric_text("  温和\n\n语气\t平稳 ") == "  温和  语气\t平稳 "


def test_singleton_bertscore_inputs_are_batched_and_trimmed() -> None:
    candidates = ["中文描述"]
    references = [["参考一", "参考二"]]
    batched_candidates, batched_references, duplicated = prepare_singleton_inputs(candidates, references)
    assert duplicated
    assert batched_candidates == ["中文描述", "中文描述"]
    assert batched_references == [references[0], references[0]]
    assert trim_bertscore_sentences({"f1": [0.8, 0.8]}, duplicated) == {"f1": [0.8]}


def test_prompts_are_chinese_audio_only_instructions() -> None:
    prompts = json.loads((ROOT / "config/prompts.json").read_text(encoding="utf-8"))
    assert set(prompts) == set(TASKS)
    for prompt in prompts.values():
        assert "音频" in prompt
        assert "转写" in prompt
        assert "解释" in prompt
        assert any("\u3400" <= char <= "\u9fff" for char in prompt)


def test_resume_identity_must_match() -> None:
    validate_resume_identity([{"run_id": "abc"}], "abc")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identity([{"run_id": "different"}], "abc")


class _SplittingBackend(CaptionBackend):
    def generate(self, request: CaptionRequest) -> str:
        raise AssertionError("the batch implementation should be used")

    def generate_batch(self, values: list[CaptionRequest]) -> list[str]:
        if any(value.id == "bad" for value in values):
            raise RuntimeError("sample failure")
        return [f"caption-{value.id}" for value in values]

    def identity(self) -> dict[str, object]:
        return {"backend": "test"}


def test_batch_failure_isolated_without_losing_good_samples() -> None:
    values = [
        CaptionRequest(id=value, task="speaker", audio_path="x.wav", prompt="p")
        for value in ("good-1", "bad", "good-2")
    ]
    results = generate_batch_resilient(_SplittingBackend(), values)
    assert results[0] == ("caption-good-1", None)
    assert results[1][0] is None
    assert results[1][1]["error_type"] == "RuntimeError"
    assert results[2] == ("caption-good-2", None)


def test_eight_model_summary_uses_public_metric_keys() -> None:
    assert metric_value(
        "bertscore", {"status": "ok", "corpus": {"bert_score.f1": 0.75}}
    ) == 0.75
    with pytest.raises(ValueError, match="unavailable"):
        metric_value("fense", {"status": "unavailable", "error": "missing"})
