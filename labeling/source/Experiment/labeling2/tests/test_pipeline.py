from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Experiment.labeling2.manifest import Sample
from Experiment.labeling2.pipeline import finalize, latest_by_id, make_run_dirs, run_api_model, run_expert, run_experts
from Experiment.labeling2.target_schema import validate_target


class PipelineGoldenTests(unittest.TestCase):
    def test_expert_resume_skips_deterministic_data_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run_dirs(run_dir)
            sample = Sample("silent-001", "/tmp/silent.wav", language_hint="English")
            (run_dir / "expert_predictions/rate_en.jsonl").write_text(
                json.dumps({
                    "sample_id": sample.sample_id,
                    "status": "error",
                    "error": "ValueError('vad_no_speech')",
                }) + "\n",
                encoding="utf-8",
            )
            with patch("Experiment.labeling2.pipeline.subprocess.run") as subprocess_run:
                run_expert(
                    [sample],
                    "rate_en",
                    run_dir,
                    {},
                    resume=True,
                    gemini_rows={},
                )
            subprocess_run.assert_not_called()

    def test_experts_can_run_concurrently_with_cached_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run_dirs(run_dir)
            samples = [
                Sample("en-001", "/tmp/en.wav", language_hint="English"),
                Sample("zh-001", "/tmp/zh.wav", language_hint="Chinese"),
            ]
            gemini_rows = [
                {"sample_id": "en-001", "parsed": {"semantic_content": {"language": "English"}}},
                {"sample_id": "zh-001", "parsed": {"semantic_content": {"language": "Chinese"}}},
            ]
            (run_dir / "raw_predictions/gemini.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in gemini_rows), encoding="utf-8"
            )
            seen = []

            def fake_run_expert(_samples, expert, _run_dir, _config, *, resume, gemini_rows):
                seen.append((expert, resume, sorted(gemini_rows)))

            with patch("Experiment.labeling2.pipeline.run_expert", side_effect=fake_run_expert):
                run_experts(
                    samples,
                    run_dir,
                    {},
                    resume=True,
                    experts=("volume", "accent_en", "rate_zh"),
                    workers=3,
                )

            self.assertEqual({row[0] for row in seen}, {"volume", "accent_en", "rate_zh"})
            self.assertTrue(all(row[1] for row in seen))
            self.assertTrue(all(row[2] == ["en-001", "zh-001"] for row in seen))

    def test_api_resume_after_transport_error_uses_clean_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run_dirs(run_dir)
            sample = Sample("resume-transport-001", "/tmp/transport.wav")
            old_error = "SSLError(UNEXPECTED_EOF_WHILE_READING)"
            old = {
                "sample_id": sample.sample_id,
                "status": "error",
                "response_text": "",
                "parsed": {},
                "field_errors": {"semantic_content.language": old_error},
            }
            (run_dir / "raw_predictions" / "gemini.jsonl").write_text(
                json.dumps(old) + "\n", encoding="utf-8"
            )
            response = '{"semantic_content":{"language":"English","topic":"weather"},"paralinguistic":{"pitch_level":"high","prosody":"varied","pause":"short"},"environment":{"background_sound_events":["rain"],"acoustic_scene":"street"}}'
            prompts = []

            def request(_model, _sample, prompt, _config):
                prompts.append(prompt)
                return response

            config = {"run": {"max_attempts": 2}, "api": {}, "backends": {"gemini": {}}}
            with patch("Experiment.labeling2.pipeline.api_request", side_effect=request):
                run_api_model([sample], "gemini", run_dir, config, resume=True)

            self.assertEqual(len(prompts), 1)
            self.assertNotIn("Repair these fields", prompts[0])
            self.assertNotIn(old_error, prompts[0])

    def test_api_transport_retry_uses_clean_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run_dirs(run_dir)
            sample = Sample("transport-001", "/tmp/transport.wav")
            response = '{"semantic_content":{"language":"English","topic":"weather"},"paralinguistic":{"pitch_level":"high","prosody":"varied","pause":"short"},"environment":{"background_sound_events":["rain"],"acoustic_scene":"street"}}'
            prompts = []

            def request(_model, _sample, prompt, _config):
                prompts.append(prompt)
                if len(prompts) == 1:
                    raise RuntimeError("SSL transport failure")
                return response

            config = {"run": {"max_attempts": 2}, "api": {}, "backends": {"gemini": {}}}
            with patch("Experiment.labeling2.pipeline.api_request", side_effect=request):
                run_api_model([sample], "gemini", run_dir, config, resume=False)

            self.assertEqual(len(prompts), 2)
            self.assertNotIn("Repair these fields", prompts[0])
            self.assertNotIn("Repair these fields", prompts[1])
            self.assertNotIn("SSL transport failure", prompts[1])

    def test_api_validation_retry_uses_repair_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run_dirs(run_dir)
            sample = Sample("validation-001", "/tmp/validation.wav")
            responses = [
                '{"semantic_content":{"language":"English"}}',
                '{"semantic_content":{"topic":"weather"},"paralinguistic":{"pitch_level":"high","prosody":"varied","pause":"short"},"environment":{"background_sound_events":["rain"],"acoustic_scene":"street"}}',
            ]
            prompts = []

            def request(_model, _sample, prompt, _config):
                prompts.append(prompt)
                return responses[len(prompts) - 1]

            config = {"run": {"max_attempts": 2}, "api": {}, "backends": {"gemini": {}}}
            with patch("Experiment.labeling2.pipeline.api_request", side_effect=request):
                run_api_model([sample], "gemini", run_dir, config, resume=False)

            self.assertEqual(len(prompts), 2)
            self.assertNotIn("Repair these fields", prompts[0])
            self.assertIn("Repair the invalid field paths", prompts[1])
            self.assertIn("never use dotted keys", prompts[1])

    def test_api_retry_merges_valid_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run_dirs(run_dir)
            sample = Sample("retry-001", "/tmp/retry.wav")
            responses = [
                '{"semantic_content":{"language":"English"}}',
                '{"semantic_content":{"topic":"weather"},"paralinguistic":{"pitch_level":"high","prosody":"varied","pause":"short"},"environment":{"background_sound_events":["rain"],"acoustic_scene":"street"}}',
            ]
            config = {"run": {"max_attempts": 2}, "api": {}, "backends": {"gemini": {}}}
            with patch("Experiment.labeling2.pipeline.api_request", side_effect=responses):
                run_api_model([sample], "gemini", run_dir, config, resume=False)
            latest = latest_by_id(run_dir / "raw_predictions" / "gemini.jsonl")[sample.sample_id]
            self.assertEqual(latest["status"], "success")
            self.assertEqual(latest["parsed"]["semantic_content"]["language"], "English")
            self.assertEqual(latest["parsed"]["semantic_content"]["topic"], "weather")

    def test_finalize_is_stable_and_schema_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            make_run_dirs(run_dir)
            (run_dir / "state" / "run_metadata.json").write_text('{"created_utc":"fixed"}\n', encoding="utf-8")

            sample = Sample("golden-001", "/tmp/golden.wav", dataset="test", transcript="hello world", language_hint="English")
            predictions = {
                "gemini": {"semantic_content": {"language": "English", "topic": "weather"}, "paralinguistic": {"pitch_level": "high", "prosody": "varied", "pause": "short"}, "environment": {"background_sound_events": ["rain"], "acoustic_scene": "street"}},
                "qwen35": {"semantic_content": {"intent": "question"}, "speaker_profile": {"gender": "male", "age": "adult", "timbre": "bright"}, "paralinguistic": {"emotion_intensity": "medium", "emphasis": {"level": "emphasized", "emphasized_text": ["hello"]}, "prosody": "varied", "nonverbal_vocalization": ["laughter"]}, "environment": {"background_sound_events": ["rain"], "recording_quality": "good", "acoustic_scene": "street"}},
                "qwen3_captioner": {"speaker_profile": {"gender": "male", "age": "adult"}, "paralinguistic": {"emotion_intensity": "medium", "emphasis": {"level": "emphasized", "emphasized_text": ["hello"]}, "prosody": "varied", "nonverbal_vocalization": ["laughter"]}, "environment": {"recording_quality": "good"}},
                "kimi_audio": {"paralinguistic": {"emphasis": {"level": "emphasized", "emphasized_text": ["hello"]}, "nonverbal_vocalization": ["laughter"]}},
                "step_audio_r1_1": {"paralinguistic": {"nonverbal_vocalization": ["laughter"]}, "environment": {"background_sound_events": ["rain"]}},
            }
            for model, parsed in predictions.items():
                (run_dir / "raw_predictions" / f"{model}.jsonl").write_text(json.dumps({"sample_id": sample.sample_id, "status": "success", "parsed": parsed, "field_errors": {}}) + "\n", encoding="utf-8")
            for expert, prediction in (("volume", "medium"), ("emotion", "neutral"), ("accent_en", "US English"), ("rate_en", "moderate")):
                (run_dir / "expert_predictions" / f"{expert}.jsonl").write_text(json.dumps({"sample_id": sample.sample_id, "status": "success", "prediction": prediction, "evidence": {}}) + "\n", encoding="utf-8")

            finalize([sample], run_dir)
            paths = [run_dir / "final" / name for name in ("labels.jsonl", "provenance.jsonl", "open_candidates.jsonl", "review_queue.jsonl",)] + [run_dir / "run_summary.json"]
            before = {path.name: path.read_bytes() for path in paths}
            finalize([sample], run_dir)
            self.assertEqual(before, {path.name: path.read_bytes() for path in paths})
            label = json.loads((run_dir / "final" / "labels.jsonl").read_text(encoding="utf-8"))
            target = label["Target_JSON_Schema"]
            self.assertEqual(validate_target(target), [])
            self.assertEqual(target["speaker_profile"]["gender"], "male")
            self.assertEqual(target["semantic_content"]["transcript"], "hello world")
            self.assertLess(
                list(target["semantic_content"]).index("transcript"),
                list(target["semantic_content"]).index("intent"),
            )
            self.assertEqual(target["paralinguistic"]["nonverbal_vocalization"], ["laughter"])


if __name__ == "__main__":
    unittest.main()
