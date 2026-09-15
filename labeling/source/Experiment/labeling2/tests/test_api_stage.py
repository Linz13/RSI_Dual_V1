from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Experiment.labeling2.api_stage import finalize_api_stage
from Experiment.labeling2.manifest import Sample
from Experiment.labeling2.pipeline import make_run_dirs


class ApiStageTests(unittest.TestCase):
    def test_preserves_api_values_and_marks_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run_dirs(run_dir)
            sample = Sample("api-001", "/tmp/api.wav", dataset="test")
            gemini = {
                "semantic_content": {"language": "English", "topic": "weather"},
                "paralinguistic": {"pitch_level": "high", "prosody": "rising", "pause": "short"},
                "environment": {"background_sound_events": ["rain"], "acoustic_scene": "street"},
            }
            qwen = {
                "semantic_content": {"intent": "statement"},
                "speaker_profile": {"gender": "female", "age": "adult", "timbre": "bright"},
                "paralinguistic": {
                    "emotion_intensity": "medium",
                    "emphasis": {"level": "emphasized", "emphasized_text": ["today"]},
                    "prosody": "rising",
                    "nonverbal_vocalization": ["none"],
                },
                "environment": {
                    "background_sound_events": ["rain"],
                    "recording_quality": "good",
                    "acoustic_scene": "street",
                },
            }
            for model, parsed in (("gemini", gemini), ("qwen35", qwen)):
                (run_dir / "raw_predictions" / f"{model}.jsonl").write_text(
                    json.dumps({"sample_id": sample.sample_id, "status": "success", "parsed": parsed, "field_errors": {}}) + "\n",
                    encoding="utf-8",
                )
            decisions = {
                "sample_id": sample.sample_id,
                "fields": {
                    field: {"status": "resolved", "passed": True, "value": value}
                    for field, value in {
                        "paralinguistic.prosody": "rising",
                        "environment.background_sound_events": ["rain"],
                        "environment.acoustic_scene": "street",
                    }.items()
                },
            }
            (run_dir / "open_resolution").mkdir()
            (run_dir / "open_resolution" / "decisions.jsonl").write_text(json.dumps(decisions) + "\n", encoding="utf-8")

            finalize_api_stage([sample], run_dir)
            label = json.loads((run_dir / "final" / "labels_api_stage.jsonl").read_text(encoding="utf-8"))
            target = label["Target_JSON_Schema"]
            self.assertEqual(target["speaker_profile"]["gender"], "female")
            self.assertIn("speaker_profile.gender", label["pending_gpu_fields"])
            self.assertIsNone(target["paralinguistic"]["emotion"])
            self.assertIn("paralinguistic.emotion", label["not_run_fields"])
            self.assertEqual(target["environment"]["acoustic_scene"], "street")
            provenance = json.loads((run_dir / "final" / "provenance_api_stage.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(provenance["fields"]["speaker_profile.gender"]["status"], "api_provisional")
            self.assertEqual(provenance["fields"]["environment.acoustic_scene"]["status"], "api_consensus")


if __name__ == "__main__":
    unittest.main()
