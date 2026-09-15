import copy
import json

import pytest

from dual_isl_train.attribute_reward import EvaluationPending, reconstruction, reference_mask, EVENTS, EMPHASIS
from dual_isl_train.labeling import MODEL_FIELDS
from dual_isl_train.label_normalization import VERSION, parse_fields, finish_fields
from dual_isl_train.partial_caption import admit
from dual_isl_train.schema import get_path, set_path
from tests_v5.test_integration import service, remote_caption


def payload(field, value):
    data = remote_caption("qwen35")
    set_path(data, field, value)
    return json.dumps(data)


@pytest.mark.parametrize("original,expected", [
    ([" SIGH ", "audible breath", "kissing sound"], ["sigh", "audible breath", "other"]),
    (["laughing", "laughter", "Coughing"], ["laughter", "cough"]),
    (["gasps", "groan", "kissing sounds"], ["other"]),
    (["UNKNOWN"], ["unknown"]), ([" None "], ["none"]), ([], ["none"]),
])
def test_event_normalization(original, expected):
    parsed = parse_fields(payload(EVENTS, original), MODEL_FIELDS["qwen35"])
    assert parsed["field_errors"] == {}
    assert get_path(parsed["attributes"], EVENTS) == expected
    assert parsed["normalizations"][EVENTS] == {"from": original, "to": expected}


@pytest.mark.parametrize("value", [["asdf"], ["sigh", "unknown"], ["none", "sigh"], "laughter", [42]])
def test_no_blanket_other_or_partial_list_acceptance(value):
    parsed = parse_fields(payload(EVENTS, value), MODEL_FIELDS["qwen35"])
    assert EVENTS in parsed["field_errors"]
    final = finish_fields(parsed)
    assert get_path(final, EVENTS) == ["unknown"]
    assert get_path(final, "speaker_profile.gender") != "unknown"


@pytest.mark.parametrize("raw", [
    "{}", "{", "[]", '{"speaker_profile":{"gender":"male","gender":"female"}}',
    '{"speaker_profile": [], "speaker_profile.gender":"male"}',
    '{"speaker_profile":{"gender":"male"},"speaker_profile.gender":"female"}',
    '{"speaker_profile":{"gender":NaN}}',
])
def test_structure_failures_never_become_field_unknown(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_fields(raw, MODEL_FIELDS["qwen35"])


def test_enum_normalization_and_captioner_scope():
    fields = ["speaker_profile.gender", "paralinguistic.emotion"]
    parsed = parse_fields('{"speaker_profile":{"gender":" Female "},"paralinguistic":{"emotion":"Happy"}}', fields)
    assert parsed["attributes"] == {"speaker_profile": {"gender": "female"}, "paralinguistic": {"emotion": "happy"}}
    raw = payload(EVENTS, ["kissing sound"])
    assert EVENTS in admit(raw)["field_errors"]  # Evaluator tolerance does not modify sampled-caption admission.


def test_invalid_field_gets_feedback_then_model_correction(tmp_path):
    calls = []
    def remote(model, path, prompt, cfg):
        calls.append(prompt)
        return payload("speaker_profile.age", "ageless" if len(calls) == 1 else "adult")
    svc = service(tmp_path, remote)
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    try:
        result = svc._remote(audio, "fixture", "qwen35")
        assert get_path(result, "speaker_profile.age") == "adult"
        assert "Validation feedback" not in calls[0]
        assert "speaker_profile.age" in calls[1] and "Validation feedback" in calls[1]
        cached = json.loads((tmp_path / "remote/fixture.qwen35.json").read_text())
        assert cached["unknown_fallback_fields"] == []
        assert cached["output_policy_version"] == VERSION
    finally:
        svc.close()


def test_partial_last_response_preserved_after_exhausting_field_retries(tmp_path):
    calls = []
    def remote(model, path, prompt, cfg):
        calls.append(prompt)
        data = json.loads(payload("speaker_profile.age", "ageless"))
        set_path(data, "speaker_profile.gender", "male" if len(calls) == 1 else "female")
        set_path(data, EMPHASIS, {"invalid": "container"})
        return json.dumps(data)
    svc = service(tmp_path, remote)
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    try:
        result = svc._remote(audio, "fixture", "qwen35")
        assert len(calls) == 2
        assert get_path(result, "speaker_profile.age") == "unknown"
        assert get_path(result, "speaker_profile.gender") == "female"
        assert get_path(result, EMPHASIS) == ["unknown"]  # Unknown is not an empty/absent emphasis list.
        cached = json.loads((tmp_path / "remote/fixture.qwen35.json").read_text())
        assert set(cached["unknown_fallback_fields"]) == {"speaker_profile.age", EMPHASIS}
        assert svc._remote(audio, "fixture", "qwen35") == result and len(calls) == 2
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert events[-2]["unknown_fallback_fields"] == cached["unknown_fallback_fields"]
    finally:
        svc.close()


@pytest.mark.parametrize("last", ["timeout", "metadata", "structure"])
def test_last_request_failure_does_not_reuse_earlier_partial_as_unknown(tmp_path, last):
    calls = []
    def remote(*args):
        calls.append(1)
        if len(calls) == 1:
            return payload(EVENTS, ["unrecognized_string"])
        if last == "timeout":
            raise TimeoutError("fixture")
        if last == "metadata":
            raise ValueError("wrong model or incomplete evaluator response")
        return "{"
    svc = service(tmp_path, remote)
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    try:
        with pytest.raises(EvaluationPending):
            svc._remote(audio, "fixture", "qwen35")
        assert not (tmp_path / "remote/fixture.qwen35.json").exists()
    finally:
        svc.close()


def test_unknown_fallback_keeps_generated_reward_denominator():
    reference = {"speaker_profile": {"age": "adult", "gender": "male"}}
    parsed = parse_fields('{"speaker_profile":{"age":"ageless","gender":"male"}}',
                          ["speaker_profile.age", "speaker_profile.gender"])
    generated = finish_fields(parsed)
    score = reconstruction(reference, generated)
    assert score["denominator"] == 2 and score["score"] == 0.5
    assert "speaker_profile.age" not in reference_mask(generated)


def test_explicit_unknown_is_accepted_without_retries(tmp_path):
    calls = []
    def remote(*args):
        calls.append(1)
        return payload("speaker_profile.age", "unknown")
    svc = service(tmp_path, remote)
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    try:
        assert get_path(svc._remote(audio, "fixture", "qwen35"), "speaker_profile.age") == "unknown"
        assert len(calls) == 1
    finally:
        svc.close()
