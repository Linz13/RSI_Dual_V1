from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

import common  # noqa: E402
import generate  # noqa: E402


class GenerationTests(unittest.TestCase):
    def test_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation_manifest.json"
            common.atomic_write_json(path, {"generation_identity_sha256": "old"})
            with self.assertRaises(RuntimeError):
                generate.ensure_generation_identity(path, "new")

    def test_valid_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            sf.write(path, np.zeros(240, dtype=np.float32), 24000)
            self.assertTrue(generate.valid_wav(path))

    def test_dsd_shards_cover_each_sample_exactly_once(self):
        samples = [{"id": f"{lang}_{i}", "language": lang, "text": "hello",
                    "DSD": "calm", "APS": "happy", "RP": "actor"}
                   for lang in ("en", "zh") for i in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            jobs = [job for index in range(4) for job in generate.make_jobs(
                generate.shard_samples(samples, 4, index), Path(directory), ("DSD",))]
        self.assertEqual(len(jobs), 20)
        self.assertEqual(len({job["id"] for job in jobs}), 20)
        self.assertEqual({job["task"] for job in jobs}, {"DSD"})
        with self.assertRaises(ValueError):
            generate.shard_samples(samples, 4, 4)

    def test_sample_seed_is_independent_of_other_samples_and_sharding(self):
        jobs = [{"id": str(i), "language": "en", "task": "DSD"} for i in range(8)]
        before = {j["id"]: generate.batch_seed(42, [j]) for j in jobs}
        resumed = {j["id"]: generate.batch_seed(42, [j])
                   for j in generate.shard_samples(jobs, 4, 2)}
        self.assertEqual(resumed, {k: before[k] for k in resumed})
        self.assertNotEqual(before["0"], before["1"])

    def test_partial_batch_resume_preserves_original_batch_membership(self):
        jobs = [{"text": "x" * (i + 1), "audio_path": str(i)} for i in range(16)]
        batches = generate.pending_batches(jobs, jobs[3:], 8, fixed=True)
        self.assertEqual(batches, [jobs[:8], jobs[8:]])
        self.assertEqual(generate.pending_batches(jobs, jobs[8:], 8, fixed=True), [jobs[8:]])


if __name__ == "__main__":
    unittest.main()
