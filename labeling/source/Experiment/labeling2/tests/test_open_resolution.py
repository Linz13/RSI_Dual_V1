from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from Experiment.labeling2.open_resolution import (
    finalize_open_labels,
    load_legacy_gpt_settings,
    resolve_open_labels,
    validate_field_decision,
)
from Experiment.labeling2.target_schema import empty_target


def request_field(kind: str, count: int, values: list[object]) -> dict[str, object]:
    return {
        "kind": kind,
        "required_support": count if count == 2 else count - 1,
        "expected_model_count": count,
        "candidates": [
            {"id": chr(65 + index), "available": value != "unknown", "value": value if value != "unknown" else None}
            for index, value in enumerate(values)
        ],
    }


class OpenResolutionUnitTests(unittest.TestCase):
    def test_legacy_settings_are_read_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt_text.py"
            path.write_text(
                'url="https://example.test/v1"\napi_key="secret-value"\n'
                'data={"model":"judge-model"}\nraise RuntimeError("must not execute")\n',
                encoding="utf-8",
            )
            settings = load_legacy_gpt_settings(path)
            self.assertEqual(settings.model, "judge-model")
            self.assertNotIn("secret-value", repr(settings))

    def test_two_model_gate_requires_two_votes(self):
        decision, error = validate_field_decision(
            "environment.acoustic_scene",
            {"core_consistent_ids": ["A"], "conflicts": ["different scene"], "supported_items": [], "final_value": "unknown"},
            request_field("scene", 2, ["street", "office"]),
        )
        self.assertIsNone(error)
        self.assertEqual(decision["status"], "conflict")
        self.assertEqual(decision["required"], 2)

    def test_three_model_gate_and_supported_claims(self):
        decision, error = validate_field_decision(
            "paralinguistic.prosody",
            {
                "core_consistent_ids": ["A", "B"], "conflicts": [],
                "supported_items": [{"value": "rising clause-final intonation", "supporting_ids": ["A", "B"]}],
                "final_value": "rising clause-final intonation with regular phrasing",
            },
            request_field("prosody", 3, ["rising intonation", "rising clause endings", "flat delivery"]),
        )
        self.assertIsNone(error)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["required"], 2)

    def test_unknown_is_abstention_and_does_not_lower_threshold(self):
        decision, error = validate_field_decision(
            "environment.background_sound_events",
            {"core_consistent_ids": ["A"], "conflicts": [], "supported_items": [], "final_value": ["unknown"]},
            request_field("events", 3, [["music"], "unknown", ["traffic"]]),
        )
        self.assertIsNone(error)
        self.assertEqual(decision["status"], "conflict")
        self.assertEqual(decision["required"], 2)

    def test_prosody_rejects_excluded_attributes(self):
        decision, error = validate_field_decision(
            "paralinguistic.prosody",
            {
                "core_consistent_ids": ["A", "B"], "conflicts": [],
                "supported_items": [{"value": "even rhythm", "supporting_ids": ["A", "B"]}],
                "final_value": "even rhythm with a slow pace",
            },
            request_field("prosody", 3, ["even rhythm", "even rhythm", "varied"]),
        )
        self.assertIsNone(decision)
        self.assertEqual(error, "prosody_repeats_excluded_attribute")

    def test_emphasis_span_must_overlap_each_supporter(self):
        decision, error = validate_field_decision(
            "paralinguistic.emphasis.emphasized_text",
            {
                "core_consistent_ids": ["A", "B"], "conflicts": [],
                "supported_items": [{"value": "really", "supporting_ids": ["A", "B"]}],
                "final_value": ["really"],
            },
            request_field("spans", 3, [["really would defer"], ["really"], []]),
        )
        self.assertIsNone(error)
        self.assertEqual(decision["value"], ["really"])


class OpenResolutionGoldenTests(unittest.TestCase):
    def test_mock_end_to_end_resume_and_stable_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            final_dir = run_dir / "final"
            final_dir.mkdir(parents=True)
            helper = Path(directory) / "gpt_text.py"
            helper.write_text(
                'url="https://example.test/v1"\napi_key="golden-secret"\n'
                'data={"model":"judge-model"}\n', encoding="utf-8",
            )
            sample_id = "sample-001"
            target = empty_target()
            target["paralinguistic"]["emphasis"]["level"] = "emphasized"
            label = {"schema_version": "labeling2.v1", "sample_id": sample_id, "audio_path": "/tmp/a.wav", "dataset": "test", "Target_JSON_Schema": target}
            candidates = {
                "schema_version": "labeling2.v1", "sample_id": sample_id,
                "environment.acoustic_scene": [
                    {"model": "m1", "status": "success", "value": "meeting room"},
                    {"model": "m2", "status": "success", "value": "meeting room"},
                ],
                "environment.background_sound_events": [
                    {"model": "m1", "status": "success", "value": ["none"]},
                    {"model": "m2", "status": "success", "value": ["none"]},
                    {"model": "m3", "status": "success", "value": ["air hum"]},
                ],
                "paralinguistic.prosody": [
                    {"model": "m1", "status": "success", "value": "rising intonation"},
                    {"model": "m2", "status": "success", "value": "rising clause endings"},
                    {"model": "m3", "status": "success", "value": "flat"},
                ],
                "paralinguistic.emphasis.emphasized_text": [
                    {"model": "m1", "status": "success", "value": ["really important"]},
                    {"model": "m2", "status": "success", "value": ["really"]},
                    {"model": "m3", "status": "success", "value": []},
                ],
            }
            provenance = {"schema_version": "labeling2.v1", "sample_id": sample_id, "diarization": "not_run", "fields": {}}
            review = {"schema_version": "labeling2.v1", "sample_id": sample_id, "issues": [
                {"field": field, "reason": "open_resolution_pending"}
                for field in candidates if field not in {"schema_version", "sample_id"}
            ]}
            for name, row in (("labels.jsonl", label), ("open_candidates.jsonl", candidates), ("provenance.jsonl", provenance), ("review_queue.jsonl", review)):
                (final_dir / name).write_text(json.dumps(row) + "\n", encoding="utf-8")
            original_hash = hashlib.sha256((final_dir / "labels.jsonl").read_bytes()).hexdigest()
            calls = []

            def fake_request(_settings, messages, *, timeout):
                calls.append((messages, timeout))
                fields = json.loads(messages[1]["content"])["fields"]
                result = {}
                for field, spec in fields.items():
                    available = [row["id"] for row in spec["candidates"] if row["available"]]
                    if spec["kind"] == "scene":
                        supporters = available[:spec["required_support"]]
                        item, final_value = "meeting room", "meeting room"
                    elif spec["kind"] == "events":
                        supporters = [row["id"] for row in spec["candidates"] if row.get("value") == ["none"]]
                        item, final_value = "none", ["none"]
                    elif spec["kind"] == "prosody":
                        supporters = available[:spec["required_support"]]
                        item, final_value = "rising clause-final intonation", "rising clause-final intonation"
                    else:
                        supporters = [
                            row["id"] for row in spec["candidates"]
                            if any("really" in value for value in (row.get("value") or []))
                        ]
                        item, final_value = "really", ["really"]
                    result[field] = {
                        "core_consistent_ids": supporters, "conflicts": [],
                        "supported_items": [{"value": item, "supporting_ids": supporters}],
                        "final_value": final_value,
                    }
                return json.dumps({"fields": result})

            summary = resolve_open_labels(run_dir, helper, resume=True, workers=1, requester=fake_request)
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(hashlib.sha256((final_dir / "labels.jsonl").read_bytes()).hexdigest(), original_hash)
            resolved = json.loads((final_dir / "labels_open_resolved.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(resolved["Target_JSON_Schema"]["environment"]["acoustic_scene"], "meeting room")
            self.assertEqual(resolved["Target_JSON_Schema"]["paralinguistic"]["emphasis"]["emphasized_text"], ["really"])
            before = {path.name: path.read_bytes() for path in final_dir.glob("*open_resolved.jsonl")}
            resolve_open_labels(run_dir, helper, resume=True, workers=1, requester=fake_request)
            self.assertEqual(len(calls), 1)
            finalize_open_labels(run_dir)
            self.assertEqual(before, {path.name: path.read_bytes() for path in final_dir.glob("*open_resolved.jsonl")})
            for path in run_dir.rglob("*"):
                if path.is_file():
                    self.assertNotIn(b"golden-secret", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
