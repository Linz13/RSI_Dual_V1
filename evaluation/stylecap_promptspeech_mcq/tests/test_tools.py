from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    TASK_ORDER,
    TASK_SPECS,
    answer_for_label,
    make_question_records,
    normalize_metadata_row,
)
from evaluate import normalize_prediction, score_predictions  # noqa: E402
import run_midasheng  # noqa: E402


def sample_audio() -> dict:
    return {
        "audio_id": "1089_134686_000003_000000",
        "speaker_id": "1089",
        "audio_path": "/tmp/example.wav",
        "relative_audio_path": "audio/LibriTTS/test-clean/example.wav",
        "attributes": {
            "gender": "male",
            "pitch": "high",
            "speaking_speed": "fast",
            "volume": "low",
        },
    }


class BenchmarkToolTests(unittest.TestCase):
    def test_captioner_backend_dispatch(self):
        args = Namespace(backend="qwen3")
        with mock.patch.object(run_midasheng, "infer_qwen3", return_value="A") as call:
            self.assertEqual(
                run_midasheng.infer({"backend": "qwen3"}, "x.wav", "p", args),
                "A",
            )
            call.assert_called_once()
        args.backend = "midasheng"
        with mock.patch.object(run_midasheng, "infer_midasheng", return_value="B") as call:
            self.assertEqual(
                run_midasheng.infer({"backend": "midasheng"}, "x.wav", "p", args),
                "B",
            )
            call.assert_called_once()

    def test_batch_backend_dispatch(self):
        questions = make_question_records(sample_audio())[:2]
        args = Namespace()
        with mock.patch.object(
            run_midasheng, "infer_qwen3_batch", return_value=["A", "B"]
        ) as call:
            self.assertEqual(
                run_midasheng.infer_batch(
                    {"backend": "qwen3"}, questions, args
                ),
                ["A", "B"],
            )
            call.assert_called_once()

    def test_failed_batch_is_split_and_order_is_preserved(self):
        questions = make_question_records(sample_audio())[:2]
        cuda = SimpleNamespace(is_available=lambda: True, empty_cache=mock.Mock())
        bundle = {"backend": "qwen3", "torch": SimpleNamespace(cuda=cuda)}

        def fake_infer(_bundle, subset, _args):
            if len(subset) > 1:
                raise RuntimeError("simulated OOM")
            return [subset[0]["answer"]]

        with mock.patch.object(run_midasheng, "infer_batch", side_effect=fake_infer):
            results = run_midasheng.infer_batch_resilient(
                bundle, questions, Namespace()
            )
        self.assertEqual(results, [("A", ""), ("C", "")])
        cuda.empty_cache.assert_called_once()

    def test_metadata_normalization_and_required_fields(self):
        row = {
            "item_name": "x",
            "spk_id": "1",
            "gender": "F",
            "pitch": "normal",
            "speaking_speed": "slow",
            "energy": "high",
        }
        self.assertEqual(
            normalize_metadata_row(row),
            {
                "gender": "female",
                "pitch": "normal",
                "speaking_speed": "slow",
                "volume": "high",
            },
        )
        incomplete = dict(row)
        incomplete["energy"] = ""
        with self.assertRaises(ValueError):
            normalize_metadata_row(incomplete)

    def test_fixed_task_order_choices_and_answers(self):
        questions = make_question_records(sample_audio())
        self.assertEqual([row["task"] for row in questions], list(TASK_ORDER))
        self.assertEqual([row["answer"] for row in questions], ["A", "C", "C", "A"])
        self.assertEqual(questions[0]["choices"], {"A": "Male", "B": "Female"})
        self.assertEqual(answer_for_label("pitch", "normal"), "B")

    def test_prediction_accepts_exact_letter_or_label_only(self):
        question = make_question_records(sample_audio())[3]
        self.assertEqual(normalize_prediction(" a ", question), "A")
        self.assertEqual(normalize_prediction("Low", question), "A")
        self.assertEqual(normalize_prediction("low", question), "A")
        high_question = make_question_records(sample_audio())[1]
        self.assertEqual(normalize_prediction("high", high_question), "C")
        self.assertIsNone(normalize_prediction("The answer is A", question))
        self.assertIsNone(normalize_prediction("", question))

    def test_scoring_wrong_answer_and_macro(self):
        benchmark = make_question_records(sample_audio())
        predictions = [
            {"question_id": row["question_id"], "prediction": row["answer"]}
            for row in benchmark
        ]
        predictions[1]["prediction"] = "A"
        summary = score_predictions(benchmark, predictions)
        self.assertEqual(summary["tasks"]["gender"]["accuracy"], 1.0)
        self.assertEqual(summary["tasks"]["pitch"]["accuracy"], 0.0)
        self.assertEqual(summary["macro_average_accuracy"], 0.75)

    def test_missing_extra_and_duplicate_predictions_fail(self):
        benchmark = make_question_records(sample_audio())
        predictions = [
            {"question_id": row["question_id"], "prediction": row["answer"]}
            for row in benchmark
        ]
        with self.assertRaises(ValueError):
            score_predictions(benchmark, predictions[:-1])
        with self.assertRaises(ValueError):
            score_predictions(benchmark, predictions + [dict(predictions[0])])
        extra = list(predictions) + [{"question_id": "extra", "prediction": "A"}]
        with self.assertRaises(ValueError):
            score_predictions(benchmark, extra)

    def test_answer_and_label_corruption_is_detectable(self):
        question = make_question_records(sample_audio())[0]
        self.assertEqual(
            question["choices"][question["answer"]].casefold(), question["label"]
        )
        corrupt = dict(question)
        corrupt["answer"] = "B"
        self.assertNotEqual(
            corrupt["choices"][corrupt["answer"]].casefold(), corrupt["label"]
        )


if __name__ == "__main__":
    unittest.main()
