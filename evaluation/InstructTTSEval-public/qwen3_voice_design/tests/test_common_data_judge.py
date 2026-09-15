from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

import common  # noqa: E402
import judge  # noqa: E402
import prepare_data  # noqa: E402
import score  # noqa: E402


def sample(language: str = "en", sample_id: str = "en_0") -> dict:
    return {
        "id": sample_id,
        "language": language,
        "text": "Hello.",
        "APS": "A precise voice.",
        "DSD": "Speak precisely.",
        "RP": "Imagine explaining something precisely.",
    }


class CommonTests(unittest.TestCase):
    def test_sample_and_safe_audio_path(self):
        common.validate_sample(sample(), "en")
        self.assertEqual(
            common.audio_relative_path("en", "APS", "en_0"),
            Path("audios/en/APS/en_0.wav"),
        )
        with self.assertRaises(ValueError):
            common.validate_sample(sample(sample_id="../escape"))

    def test_selection_is_first_per_language(self):
        rows = [sample("en", "en_0"), sample("en", "en_1")]
        rows += [sample("zh", "zh_0"), sample("zh", "zh_1")]
        selected = common.selected_samples(rows, 1)
        self.assertEqual([row["id"] for row in selected], ["en_0", "zh_0"])

    def test_checkpoint_last_record_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            common.atomic_write_jsonl(
                path,
                [
                    {"language": "en", "id": "en_0", "task": "APS", "status": "failed"},
                    {"language": "en", "id": "en_0", "task": "APS", "status": "success"},
                ],
            )
            latest = common.load_latest_records(path)
            self.assertEqual(latest[("en", "en_0", "APS")]["status"], "success")


class DataTests(unittest.TestCase):
    def test_parquet_conversion_excludes_reference_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "en.parquet"
            row = sample()
            table = pa.Table.from_pylist(
                [
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "APS": row["APS"],
                        "DSD": row["DSD"],
                        "RP": row["RP"],
                        "reference_audio": {"bytes": b"not-read", "path": None},
                    }
                ]
            )
            pq.write_table(table, path)
            with mock.patch.dict(prepare_data.EXPECTED_ROWS, {"en": 1}), mock.patch.dict(
                prepare_data.EXPECTED_BYTES, {"en": path.stat().st_size}
            ):
                rows = prepare_data.convert_split(path, "en")
            self.assertEqual(rows, [row])
            self.assertNotIn("reference_audio", rows[0])


class JudgeTests(unittest.TestCase):
    def test_prompt_and_response(self):
        template = f"Instruction: {judge.PLACEHOLDER}"
        prompt = judge.build_prompt(template, "Speak softly")
        self.assertEqual(prompt, "Instruction: Speak softly")
        self.assertTrue(judge.extract_result('```json\n{"一致性": true}\n```')["一致性"])
        with self.assertRaises(ValueError):
            judge.extract_result('{"一致性": "true"}')

    def test_incomplete_exit_mode(self):
        self.assertEqual(judge.result_exit_code(9, 10, False), 1)
        self.assertEqual(judge.result_exit_code(9, 10, True), 0)
        self.assertEqual(judge.result_exit_code(10, 10, False), 0)

    def test_recorded_failure_is_skipped_unless_retry_is_requested(self):
        job = {"language": "en", "id": "en_0", "task": "APS"}
        latest = {
            ("en", "en_0", "APS"): {
                **job,
                "status": "failed",
                "dry_run": False,
            }
        }
        pending, completed, failed = judge.checkpoint_progress(
            [job], latest, False, True, False
        )
        self.assertEqual(pending, [])
        self.assertEqual(completed, set())
        self.assertEqual(failed, {("en", "en_0", "APS")})
        pending, _, _ = judge.checkpoint_progress(
            [job], latest, False, True, True
        )
        self.assertEqual(pending, [job])

    def test_dry_run_never_constructs_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            sf.write(audio, np.zeros(240, dtype=np.float32), 24000)
            generation = root / "generation_manifest.json"
            common.atomic_write_json(
                generation,
                {
                    "complete": True,
                    "expected_audios": 1,
                    "generation_identity_sha256": "identity",
                    "items": [
                        {
                            **sample(),
                            "task": "APS",
                            "instruction": "A precise voice.",
                            "audio_path": str(audio),
                            "status": "generated",
                        }
                    ],
                },
            )
            prompt = root / "prompt.txt"
            prompt.write_text(f"Judge this: {judge.PLACEHOLDER}", encoding="utf-8")
            output = root / "judge"
            argv = [
                "judge.py",
                "--generation-manifest",
                str(generation),
                "--output-dir",
                str(output),
                "--prompt-file",
                str(prompt),
                "--dry-run",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                judge, "create_client", side_effect=AssertionError("network client constructed")
            ):
                self.assertEqual(judge.main(), 0)
            result = common.read_jsonl(output / "judge_results.jsonl")[0]
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["status"], "success")


class ScoreTests(unittest.TestCase):
    def make_fixture(self, root: Path, include_result: bool) -> tuple[Path, Path, Path]:
        generation = root / "generation_manifest.json"
        common.atomic_write_json(
            generation,
            {
                "expected_audios": 1,
                "items": [
                    {
                        "language": "en",
                        "id": "en_0",
                        "task": "APS",
                        "status": "generated",
                    }
                ],
            },
        )
        results = root / "judge_results.jsonl"
        if include_result:
            common.atomic_write_jsonl(
                results,
                [
                    {
                        "language": "en",
                        "id": "en_0",
                        "task": "APS",
                        "status": "success",
                        "gemini_score": True,
                        "dry_run": True,
                        "usage": None,
                    }
                ],
            )
        else:
            results.write_text("", encoding="utf-8")
        return generation, results, root / "summary.json"

    def test_complete_dry_run_is_labeled_non_official(self):
        with tempfile.TemporaryDirectory() as directory:
            generation, results, output = self.make_fixture(Path(directory), True)
            argv = [
                "score.py",
                "--generation-manifest",
                str(generation),
                "--judge-results",
                str(results),
                "--output",
                str(output),
                "--allow-dry-run",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(score.main(), 0)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["official"])

    def test_paid_proxy_result_is_not_labeled_official(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, results, output = self.make_fixture(root, True)
            rows = common.read_jsonl(results)
            rows[0].update(
                {
                    "dry_run": False,
                    "judge_model": score.OFFICIAL_JUDGE_MODEL,
                    "backend": "inline",
                }
            )
            common.atomic_write_jsonl(results, rows)
            common.atomic_write_json(
                root / "judge_metadata.json",
                {
                    "judge_model": score.OFFICIAL_JUDGE_MODEL,
                    "prompt_sha256": score.OFFICIAL_PROMPT_SHA256,
                    "backend": "inline",
                    "endpoint": "https://proxy.invalid",
                },
            )
            argv = [
                "score.py",
                "--generation-manifest",
                str(generation),
                "--judge-results",
                str(results),
                "--output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(score.main(), 0)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["official"])

    def test_exact_paper_judge_metadata_is_labeled_official(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, results, output = self.make_fixture(root, True)
            rows = common.read_jsonl(results)
            rows[0].update(
                {
                    "dry_run": False,
                    "judge_model": score.OFFICIAL_JUDGE_MODEL,
                    "backend": "files",
                }
            )
            common.atomic_write_jsonl(results, rows)
            common.atomic_write_json(
                root / "judge_metadata.json",
                {
                    "judge_model": score.OFFICIAL_JUDGE_MODEL,
                    "prompt_sha256": score.OFFICIAL_PROMPT_SHA256,
                    "backend": "files",
                    "endpoint": None,
                },
            )
            argv = [
                "score.py",
                "--generation-manifest",
                str(generation),
                "--judge-results",
                str(results),
                "--output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(score.main(), 0)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(summary["official"])

    def test_incomplete_results_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            generation, results, output = self.make_fixture(Path(directory), False)
            argv = [
                "score.py",
                "--generation-manifest",
                str(generation),
                "--judge-results",
                str(results),
                "--output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(score.main(), 1)

    def test_incomplete_results_record_scored_coverage_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation_manifest.json"
            common.atomic_write_json(
                generation,
                {
                    "expected_audios": 2,
                    "items": [
                        {
                            "language": "en",
                            "id": "en_0",
                            "task": "APS",
                            "status": "generated",
                        },
                        {
                            "language": "en",
                            "id": "en_1",
                            "task": "APS",
                            "status": "generated",
                        },
                    ],
                },
            )
            results = root / "judge_results.jsonl"
            common.atomic_write_jsonl(
                results,
                [
                    {
                        "language": "en",
                        "id": "en_0",
                        "task": "APS",
                        "status": "success",
                        "gemini_score": True,
                        "dry_run": False,
                        "usage": None,
                    },
                    {
                        "language": "en",
                        "id": "en_1",
                        "task": "APS",
                        "status": "failed",
                        "error_type": "ReadTimeout",
                        "error": "timed out",
                        "elapsed_seconds": 10.0,
                    },
                ],
            )
            output = root / "summary.json"
            argv = [
                "score.py",
                "--generation-manifest",
                str(generation),
                "--judge-results",
                str(results),
                "--output",
                str(output),
                "--allow-incomplete",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(score.main(), 0)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["expected"], 2)
            self.assertEqual(summary["scored"], 1)
            self.assertEqual(summary["coverage_percentage"], 50.0)
            self.assertEqual(
                summary["score_basis"]["policy"],
                "successful_judge_results_only",
            )
            self.assertEqual(summary["metrics"]["en"]["APS"]["scored"], 1)
            self.assertEqual(
                summary["failed_items"],
                [
                    {
                        "language": "en",
                        "id": "en_1",
                        "task": "APS",
                        "status": "failed",
                        "error_type": "ReadTimeout",
                        "error": "timed out",
                        "elapsed_seconds": 10.0,
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
