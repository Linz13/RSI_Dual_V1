from __future__ import annotations

from copy import deepcopy

from dual_isl_train.dual_space import (
    SYNTHESIZABLE_FIELDS,
    project_synth_caption,
    synth_caption_json,
)
from dual_isl_train.workers.qwen3_captioner import Qwen3CaptionerWorker


def _reverse_object_order(value: dict) -> dict:
    return {key: deepcopy(value[key]) for key in reversed(value)}


def test_psyn_value_spans_are_order_independent_and_cover_all_16_fields(sample_caption):
    caption = project_synth_caption(sample_caption)
    caption["paralinguistic"]["emphasis"] = {
        "emphasized_text": ["paired anchor"],
        "level": "emphasized",
    }
    caption["semantic_content"] = _reverse_object_order(caption["semantic_content"])
    caption["speaker_profile"] = _reverse_object_order(caption["speaker_profile"])
    caption["paralinguistic"] = _reverse_object_order(caption["paralinguistic"])

    target = synth_caption_json(caption)
    spans = Qwen3CaptionerWorker._value_content_spans(target, caption)

    assert target.index('"accent":') < target.index('"gender":')
    assert set(spans) == set(SYNTHESIZABLE_FIELDS)
    assert len(spans) == 16
    assert all(begin < end <= len(target) for field_spans in spans.values() for begin, end in field_spans)


def test_psyn_value_spans_exclude_unknown_and_empty_values(sample_caption):
    caption = project_synth_caption(sample_caption)
    caption["semantic_content"]["language"] = "unknown"
    caption["speaker_profile"]["age"] = "unknown"
    caption["paralinguistic"]["nonverbal_vocalization"] = ["unknown"]

    target = synth_caption_json(caption)
    spans = Qwen3CaptionerWorker._value_content_spans(target, caption)

    assert "semantic_content.language" not in spans
    assert "speaker_profile.age" not in spans
    assert "paralinguistic.emphasis.emphasized_text" not in spans
    assert "paralinguistic.nonverbal_vocalization" not in spans
    assert "semantic_content.transcript" in spans
