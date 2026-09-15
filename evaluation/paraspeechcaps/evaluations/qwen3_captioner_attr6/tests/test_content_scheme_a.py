from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import ALL_FIELDS, load_schema  # noqa: E402
from content_scheme_a import (  # noqa: E402
    PROTOCOL,
    build_all_field_prompts,
    option_table,
    parse_field_response,
)
from score_content_scheme_a import score_samples  # noqa: E402
import run_content_scheme_a as scheme_runner  # noqa: E402
import score_content_scheme_a as scheme_scorer  # noqa: E402


class ContentSchemeATests(unittest.TestCase):
    def test_protocol_has_six_independent_prompts_and_stable_option_ids(self):
        prompts = build_all_field_prompts()
        self.assertEqual(tuple(prompts), ALL_FIELDS)
        self.assertEqual(len(set(prompts.values())), 6)
        self.assertEqual(option_table("gender"), {"G1": "female", "G2": "male", "G0": "unknown"})
        self.assertEqual(option_table("pitch")["P3"], "high-pitched")
        self.assertEqual(option_table("intrinsic_traits")["I23"], "vocal-fry")
        self.assertEqual(option_table("situational_traits")["S21"], "whispered")
        for field, prompt in prompts.items():
            self.assertIn("Return only", prompt)
            self.assertIn("one closed-set speech-attribute question", prompt)
            self.assertIn(field.split("_")[0], prompt.casefold())

    def test_single_choice_parser_prioritizes_one_unambiguous_id(self):
        self.assertEqual(parse_field_response("gender", "G1")["prediction"], "female")
        self.assertEqual(parse_field_response("gender", "answer: g 02")["prediction"], "male")
        self.assertEqual(parse_field_response("gender", "G0")["prediction"], "unknown")
        ambiguous = parse_field_response("gender", "G1 or G2; probably female")
        self.assertEqual(ambiguous["status"], "unparsed")
        self.assertEqual(ambiguous["parse_mode"], "multiple_options")

    def test_canonical_label_fallback_avoids_substring_collisions(self):
        female = parse_field_response("gender", "female")
        self.assertEqual(female["prediction"], "female")
        self.assertEqual(female["parse_mode"], "canonical_label_fallback")
        indian_american = parse_field_response("accent", "indian-american")
        self.assertEqual(indian_american["prediction"], "indian-american")
        self.assertEqual(indian_american["status"], "success")

    def test_multi_choice_ids_labels_none_and_invalid_output(self):
        ids = parse_field_response("intrinsic_traits", "I01, I23, I01")
        self.assertEqual(ids["prediction"], ["authoritative", "vocal-fry"])
        labels = parse_field_response("situational_traits", "calm, happy")
        self.assertEqual(labels["prediction"], ["calm", "happy"])
        self.assertEqual(parse_field_response("situational_traits", "NONE")["prediction"], [])
        self.assertEqual(parse_field_response("situational_traits", "unusable output")["status"], "unparsed")

    def test_parse_failure_zeros_only_its_field_on_fixed_denominator(self):
        sample = {
            "sample_id": "sample-1",
            "benchmark_indices": [1],
            "source": "unit-test",
            "audio_path": "/tmp/not-read.wav",
            "gt": {
                "gender": "female",
                "pitch": "high-pitched",
                "speaking_rate": "measured speed",
                "accent": "american",
                "intrinsic_traits": [],
                "situational_traits": [],
            },
        }
        fields = {
            "gender": {"status": "success", "prediction": "female", "parse_mode": "option_id"},
            "pitch": {"status": "unparsed", "prediction": None, "parse_mode": "no_option"},
            "speaking_rate": {"status": "success", "prediction": "measured speed", "parse_mode": "option_id"},
            "accent": {"status": "success", "prediction": "american", "parse_mode": "option_id"},
            "intrinsic_traits": {"status": "success", "prediction": [], "parse_mode": "explicit_none"},
            "situational_traits": {"status": "success", "prediction": [], "parse_mode": "explicit_none"},
        }
        predictions = {
            "sample-1": {
                "protocol": PROTOCOL,
                "sample_id": "sample-1",
                "fields": fields,
            }
        }
        row = score_samples([sample], predictions)[0]
        self.assertEqual(row["score_pitch"], 0.0)
        self.assertEqual(row["score_gender"], 1.0)
        self.assertEqual(row["parsed_fields"], 5)
        self.assertEqual(row["all_six_fields_parsed"], 0)
        self.assertAlmostEqual(row["sample_score"], 5 / 6)

    def test_mock_end_to_end_run_resume_and_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            manifest_path = root / "manifest.jsonl"
            sample = {
                "sample_id": "sample-1",
                "benchmark_indices": [1],
                "source": "unit-test",
                "audio_path": "/tmp/not-read.wav",
                "duration_seconds": 1.0,
                "gt": {
                    "gender": "female",
                    "pitch": "high-pitched",
                    "speaking_rate": "measured speed",
                    "accent": "american",
                    "intrinsic_traits": [],
                    "situational_traits": [],
                },
            }
            manifest_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            output_dir = root / "outputs"
            responses = iter(("G1", "P3", "R2", "A1", "NONE", "NONE"))
            run_argv = [
                "run_content_scheme_a.py",
                "--manifest", str(manifest_path),
                "--output-dir", str(output_dir),
                "--candidate-name", "mock-candidate",
                "--backend", "qwen3",
                "--model-dir", str(model_dir),
            ]
            with (
                patch.object(sys, "argv", run_argv),
                patch.object(
                    scheme_runner,
                    "load_captioner",
                    return_value={"backend": "qwen3", "adapter": None},
                ) as load_model,
                patch.object(
                    scheme_runner,
                    "infer_one",
                    side_effect=lambda *_args: next(responses),
                ) as infer,
            ):
                self.assertEqual(scheme_runner.main(), 0)
            load_model.assert_called_once()
            self.assertEqual(infer.call_count, 6)

            resume_argv = run_argv + ["--resume"]
            with (
                patch.object(sys, "argv", resume_argv),
                patch.object(scheme_runner, "load_captioner") as resumed_load,
                patch.object(scheme_runner, "infer_one") as resumed_infer,
            ):
                self.assertEqual(scheme_runner.main(), 0)
            resumed_load.assert_not_called()
            resumed_infer.assert_not_called()

            report_dir = root / "reports"
            with patch.object(
                sys,
                "argv",
                [
                    "score_content_scheme_a.py",
                    "--manifest", str(output_dir / "selected_manifest.jsonl"),
                    "--predictions", str(output_dir / "predictions.jsonl"),
                    "--output-dir", str(report_dir),
                ],
            ):
                scheme_scorer.main()
            summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_name"], "mock-candidate")
            self.assertEqual(summary["final_score"], 1.0)
            self.assertEqual(summary["all_six_fields_parse_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
