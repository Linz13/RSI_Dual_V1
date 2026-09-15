from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Experiment.labeling2.consensus import list_vote, required_votes, scalar_vote
from Experiment.labeling2.expert_worker import LOW_NORMAL, NORMAL_HIGH, normalize_accent_label, rate_label, textrol_label
from Experiment.labeling2.manifest import load_manifest
from Experiment.labeling2.parsing import extract_json, normalize_payload, validate_subset
from Experiment.labeling2.prompts import build_prompt
from Experiment.labeling2.target_schema import empty_target, set_path, validate_target


class CoreTests(unittest.TestCase):
    def test_required_votes(self):
        self.assertEqual(required_votes(2), 2)
        self.assertEqual(required_votes(3), 2)
        self.assertEqual(required_votes(4), 3)

    def test_scalar_consensus(self):
        self.assertTrue(scalar_vote(["male", "male"]).passed)
        self.assertTrue(scalar_vote(["male", "male", "female"]).passed)
        self.assertFalse(scalar_vote(["male", "female"]).passed)
        self.assertEqual(scalar_vote(["male", "female"]).value, "unknown")
        self.assertFalse(scalar_vote(["unknown", None, None]).passed)

    def test_event_vote(self):
        result = list_vote([["laughter"], ["laughter", "sigh"], ["laughter"], ["unknown"]])
        self.assertEqual(result.value, ["laughter"])
        self.assertTrue(result.passed)
        self.assertEqual(list_vote([["none"], ["laughter"], ["sigh"], ["unknown"]]).value, ["unknown"])

    def test_json_extraction(self):
        payload, status = extract_json('reasoning\n```json\n{"speaker_profile":{"gender":"male"}}\n```')
        self.assertEqual(status, "json")
        self.assertEqual(payload["speaker_profile"]["gender"], "male")

    def test_prompt_does_not_embed_dataset_text(self):
        prompt = build_prompt("gemini", ["semantic_content.language"])
        self.assertNotIn("secret dataset transcript", prompt)
        self.assertNotIn("private metadata payload", prompt)

    def test_kimi_prompt_requires_short_non_repeating_json(self):
        prompt = build_prompt("kimi_audio", [
            "paralinguistic.emphasis.level",
            "paralinguistic.emphasis.emphasized_text",
            "paralinguistic.nonverbal_vocalization",
        ])
        self.assertIn("under 200 text tokens", prompt)
        self.assertIn("at most 5 distinct items", prompt)
        self.assertIn("sobbing", prompt)
        self.assertIn("NEVER emphasis levels", prompt)
        self.assertIn("Stop immediately after the final closing brace", prompt)

    def test_subset_validation_and_normalization_shape(self):
        payload = {"environment": {"background_sound_events": ["keyboard typing"]}}
        self.assertFalse(validate_subset(payload, ["environment.background_sound_events"]))
        self.assertTrue(validate_subset({"environment": {"background_sound_events": ["none", "rain"]}}, ["environment.background_sound_events"]))

    def test_model_encoded_lists_are_recovered(self):
        fields = [
            "paralinguistic.emphasis.emphasized_text",
            "paralinguistic.nonverbal_vocalization",
        ]
        payload = {
            "paralinguistic": {
                "emphasis": {"emphasized_text": "[]"},
                "nonverbal_vocalization": '["sigh"]',
            }
        }
        self.assertEqual(validate_subset(payload, fields), {})
        normalized = normalize_payload(payload, fields)
        self.assertEqual(normalized["paralinguistic"]["emphasis"]["emphasized_text"], [])
        self.assertEqual(normalized["paralinguistic"]["nonverbal_vocalization"], ["sigh"])

        scalar = {"paralinguistic": {"emphasis": {"emphasized_text": "hello"}}}
        self.assertEqual(validate_subset(scalar, [fields[0]]), {})
        self.assertEqual(
            normalize_payload(scalar, [fields[0]])["paralinguistic"]["emphasis"]["emphasized_text"],
            ["hello"],
        )

    def test_dotted_model_keys_are_recovered_as_nested_fields(self):
        fields = ["speaker_profile.gender", "paralinguistic.emphasis.emphasized_text"]
        payload = {
            "speaker_profile.gender": "male",
            "paralinguistic.emphasis.emphasized_text": ["这个名字好好听"],
        }
        self.assertEqual(validate_subset(payload, fields), {})
        normalized = normalize_payload(payload, fields)
        self.assertEqual(normalized["speaker_profile"]["gender"], "male")
        self.assertEqual(
            normalized["paralinguistic"]["emphasis"]["emphasized_text"],
            ["这个名字好好听"],
        )

    def test_repeated_or_excessive_emphasized_text_is_rejected(self):
        field = "paralinguistic.emphasis.emphasized_text"
        duplicate = {"paralinguistic": {"emphasis": {"emphasized_text": ["这个", "这个"]}}}
        excessive = {"paralinguistic": {"emphasis": {"emphasized_text": [str(i) for i in range(6)]}}}
        self.assertEqual(validate_subset(duplicate, [field])[field], "duplicate values")
        self.assertEqual(validate_subset(excessive, [field])[field], "must contain at most 5 items")

    def test_unknown_nonverbal_event_uses_other_catchall(self):
        field = "paralinguistic.nonverbal_vocalization"
        payload = {"paralinguistic": {"nonverbal_vocalization": ["question mark"]}}
        self.assertEqual(validate_subset(payload, [field]), {})
        self.assertEqual(normalize_payload(payload, [field])["paralinguistic"]["nonverbal_vocalization"], ["other"])
        mixed = {"paralinguistic": {"nonverbal_vocalization": ["none", "question mark"]}}
        self.assertEqual(validate_subset(mixed, [field])[field], "none/unknown must be exclusive")

    def test_empty_event_lists_mean_none_but_empty_emphasis_stays_empty(self):
        fields = [
            "paralinguistic.nonverbal_vocalization",
            "environment.background_sound_events",
            "paralinguistic.emphasis.emphasized_text",
        ]
        payload = {
            "paralinguistic": {
                "nonverbal_vocalization": [],
                "emphasis": {"emphasized_text": []},
            },
            "environment": {"background_sound_events": []},
        }
        self.assertEqual(validate_subset(payload, fields), {})
        normalized = normalize_payload(payload, fields)
        self.assertEqual(normalized["paralinguistic"]["nonverbal_vocalization"], ["none"])
        self.assertEqual(normalized["environment"]["background_sound_events"], ["none"])
        self.assertEqual(normalized["paralinguistic"]["emphasis"]["emphasized_text"], [])

    def test_target_defaults_are_valid(self):
        target = empty_target()
        self.assertEqual(validate_target(target), [])
        target["semantic_content"]["topic"] = 123
        self.assertTrue(validate_target(target))
        target["semantic_content"]["topic"] = "unknown"
        set_path(target, "paralinguistic.nonverbal_vocalization", ["none", "laughter"])
        self.assertTrue(validate_target(target))

    def test_rate_boundaries(self):
        self.assertEqual(rate_label(3.999999, 4.0, 10.0), "slow")
        self.assertEqual(rate_label(4.0, 4.0, 10.0), "moderate")
        self.assertEqual(rate_label(10.0, 4.0, 10.0), "moderate")
        self.assertEqual(rate_label(10.000001, 4.0, 10.0), "fast")

    def test_textrol_boundaries(self):
        self.assertEqual(textrol_label(LOW_NORMAL - 1e-9), "low")
        self.assertEqual(textrol_label(LOW_NORMAL), "medium")
        self.assertEqual(textrol_label(NORMAL_HIGH), "medium")
        self.assertEqual(textrol_label(NORMAL_HIGH + 1e-9), "high")

    def test_released_xlsr_accent_labels_are_normalized_exactly(self):
        expected = {
            "us": "US English", "england": "England English",
            "australia": "Australian English", "indian": "Indian English",
            "canada": "Canadian English", "bermuda": "Bermudian English",
            "scotland": "Scottish English", "african": "African English",
            "ireland": "Irish English", "newzealand": "New Zealand English",
            "wales": "Welsh English", "malaysia": "Malaysian English",
            "philippines": "Philippine English", "singapore": "Singapore English",
            "hongkong": "Hong Kong English", "southatlandtic": "South Atlantic English",
        }
        for raw, canonical in expected.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_accent_label(raw), canonical)
        self.assertEqual(normalize_accent_label("unrecognized"), "other")

    def test_manifest_duplicate_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "a.wav"
            audio.write_bytes(b"RIFF")
            manifest = root / "data.jsonl"
            row = {"sample_id": "x", "audio_path": str(audio)}
            manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(manifest)

            manifest.write_text(json.dumps({"sample_id": "relative", "audio_path": "a.wav"}) + "\n", encoding="utf-8")
            self.assertEqual(load_manifest(manifest)[0].audio_path, str(audio.resolve()))


if __name__ == "__main__":
    unittest.main()
