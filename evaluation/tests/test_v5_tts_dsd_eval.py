import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v5_tts_dsd_eval as dsd
import score


class DsdTests(unittest.TestCase):
    def fixture(self, root):
        samples = [{"language": lang, "id": f"{lang}_{i}", "text": "hello", "DSD": "calm"}
                   for lang in ("en", "zh") for i in range(2)]
        plan = {"num_shards": 4, "batch_size": 1, "seed_strategy": "stable",
                "adapter": {"weights_sha256": "trained"}, "manifest_hashes": {"en": "en", "zh": "zh"},
                "selected_ids": [r["id"] for r in samples],
                "decoding": {"seed": 42, "temperature": 1.0, "top_p": .9,
                             "max_new_tokens": 8192, "attention": "sdpa"}}
        for i, sample in enumerate(samples):
            folder = root / "shards" / f"{i:02d}"
            folder.mkdir(parents=True)
            audio = folder / "audio.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                handle.writeframes(b"\0\0" * 240)
            item = {**sample, "task": "DSD", "instruction": sample["DSD"],
                    "audio_path": str(audio), "audio_sha256": dsd.sha256_file(audio),
                    "status": "generated"}
            identity = {"tasks": ["DSD"], "selected_ids": [sample["id"]],
                        "adapter": plan["adapter"], "batch_size": 1,
                        "seed_strategy": plan["seed_strategy"], **plan["decoding"],
                        "data_files": {lang: {"sha256": lang} for lang in ("en", "zh")},
                        "shard": {"count": 4, "index": i}}
            dsd.atomic_write_json(folder / "generation_manifest.json", {
                "complete": True, "expected_audios": 1, "generation_identity": identity,
                "generation_identity_sha256": dsd.sha256_json(identity), "items": [item]})
        return plan, samples

    def test_merge_resume_and_dry_judge_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, samples = self.fixture(root)
            merged = dsd.merge(root, plan, samples)
            self.assertEqual(merged["expected_audios"], 4)
            self.assertEqual(dsd.merge(root, plan, samples), merged)
            dsd.judge(root, 2, dry_run=True)
            dsd.judge(root, 2, dry_run=True)
            results = (root / "dry_run/judge_results.jsonl").read_text().splitlines()
            self.assertEqual(len(results), 4)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(dsd.score(root, dry_run=True), 0)
            report = dsd.read_json(root / "dry_run/dsd_summary.json")
            self.assertTrue(report["complete"])
            self.assertTrue(report["dry_run"])
            self.assertIsNone(report["delta_percentage_points"])
            self.assertEqual(report["usage"]["estimated_cost_usd"], 0)

    def test_merge_rejects_duplicate_and_missing_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, samples = self.fixture(root)
            p = root / "shards/03/generation_manifest.json"
            child = dsd.read_json(p)
            child["items"] = dsd.read_json(root / "shards/02/generation_manifest.json")["items"]
            dsd.atomic_write_json(p, child)
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                dsd.merge(root, plan, samples)

    def test_merge_rejects_modified_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, samples = self.fixture(root)
            (root / "shards/00/audio.wav").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "audio mismatch"):
                dsd.merge(root, plan, samples)

    def test_merge_rejects_wrong_adapter_or_incomplete_shard(self):
        for field, value in (("adapter", {"weights_sha256": "base"}), ("complete", False)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                plan, samples = self.fixture(root)
                path = root / "shards/00/generation_manifest.json"
                child = dsd.read_json(path)
                if field == "complete":
                    child[field] = value
                else:
                    child["generation_identity"][field] = value
                    child["generation_identity_sha256"] = dsd.sha256_json(child["generation_identity"])
                dsd.atomic_write_json(path, child)
                with self.assertRaisesRegex(RuntimeError, "mismatched"):
                    dsd.merge(root, plan, samples)

    def test_plan_mismatch_preserves_previous_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dsd.lock_plan(root, {"batch_size": 1})
            with self.assertRaises(RuntimeError):
                dsd.lock_plan(root, {"batch_size": 4})
            self.assertEqual(dsd.read_json(root / "dsd_plan.json"), {"batch_size": 1})

    def test_memory_check_refuses_insufficient_free_gpu(self):
        with mock.patch.object(dsd.subprocess, "check_output", return_value="4, 12000, 81559, H100\n"):
            with self.assertRaisesRegex(RuntimeError, "at least 18"):
                dsd.check_gpu_memory(["4"], 16)

    def test_command_only_generates_dsd_on_assigned_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            plan, _ = self.fixture(Path(folder))
            plan["adapter"]["path"] = "adapter"
            command = dsd.generation_command(Path(folder), plan, 2, None, 16)
            self.assertEqual(command[command.index("--tasks") + 1], "DSD")
            self.assertEqual(command[command.index("--shard-index") + 1], "2")
            self.assertIn("--seed-per-batch", command)

    def test_scoring_includes_gemini_thinking_without_double_counting_completion(self):
        for usage, expected in (
            ({"prompt_token_count": 1000, "candidates_token_count": 100,
              "thoughts_token_count": 900}, 1000),
            ({"prompt_tokens": 1000, "completion_tokens": 1000,
              "thoughts_token_count": 900}, 1000),
        ):
            with self.subTest(usage=usage), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                item = {"language": "en", "id": "en_0", "task": "DSD", "status": "generated"}
                manifest = root / "generation_manifest.json"
                dsd.atomic_write_json(manifest, {"expected_audios": 1, "items": [item]})
                results = root / "judge_results.jsonl"
                results.write_text(json.dumps({**item, "status": "success", "gemini_score": True,
                                                "dry_run": False, "usage": usage}) + "\n")
                output = root / "summary.json"
                argv = ["score", "--generation-manifest", str(manifest), "--judge-results", str(results),
                        "--output", str(output)]
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(score.main(), 0)
                result = dsd.read_json(output)
                self.assertEqual(result["usage"]["billable_output_tokens"], expected)
                self.assertEqual(result["usage"]["estimated_cost_usd"], .01125)
                self.assertEqual(result["score_basis"]["macro_cells_expected"], 1)


if __name__ == "__main__":
    unittest.main()
