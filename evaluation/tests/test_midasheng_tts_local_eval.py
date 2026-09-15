import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "midasheng_tts_local_eval.py"
sys.path.insert(0, str(PATH.parent))
spec = importlib.util.spec_from_file_location("midasheng_tts_local_eval", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class LocalEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.data = self.root / "data.jsonl"
        self.data.write_text(''.join(json.dumps({"text_to_synthesize": "test", "category": "test", "language": "en"}) + '\n' for _ in range(2)))
        self.mos = self.root / "mos.ckpt"
        self.mos.write_bytes(b"mos")
        self.patches = [patch.object(m, "SOURCE_RUN", self.root / "training"),
                        patch.object(m, "MODEL", self.model), patch.object(m, "DATA", self.data),
                        patch.object(m, "MOS", self.mos)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def checkpoint(self, rounds=6):
        folder = m.SOURCE_RUN / f"round_{rounds-1:03d}"
        adapter = folder / "checkpoints/tts_final"
        adapter.mkdir(parents=True)
        m.write_json(adapter / "adapter_config.json", {"base_model_name_or_path": str(self.model)})
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        m.write_json(folder / "commit.json", {"round": rounds-1, "same_round_start": True,
                     "tts": {"path": str(adapter), "sha256": m.checkpoint_hash(adapter)}})
        return folder, adapter

    def completed_case(self):
        self.checkpoint()
        identity = m.checkpoint_identity(6)
        folder = self.root / "eval/full/rounds_06"
        generation = m.expected_generation(identity, "full")
        m.write_json(folder / "local_run_identity.json", {**identity, "generation": generation, "wvmos_sha256": m.sha256_file(self.mos)})
        m.write_json(folder / "generation_manifest.json", {"generation_identity": generation,
                     "generation_identity_sha256": m.sha256_json(generation), "expected_samples": 2})
        for i in range(2):
            p = folder / "audios" / f"{i}.wav"
            p.parent.mkdir(exist_ok=True)
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\0" * 480)
        stage = folder / "staged_evaluation"
        m.write_json(stage / "local_metadata.json", {"stage": "local_metrics", "format_version": 1,
                     "dataset_sha256": m.sha256_file(self.data), "selected_ids_sha256": m.sha256_json([0, 1]),
                     "selected_samples": 2, "seed": 42, "num_samples": None,
                     "audio_dir": str(folder / "audios"), "whisper_model": "openai/whisper-large-v3",
                     "wvmos_checkpoint": str(self.mos), "wvmos_checkpoint_sha256": m.sha256_file(self.mos)})
        records = [{"stage": "local_metrics", "format_version": 1, "status": "success", "unique_id_eval": i,
                    "audio_out_path": str(folder / "audios" / f"{i}.wav"), "wer": wer, "mos_score": mos}
                   for i, wer, mos in [(0, 10, 3), (1, 40, 5)]]
        (stage / "local_metrics.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in records))
        return folder, records

    def test_round_mapping_and_committed_hash(self):
        for rounds, index in [(6, 5), (9, 8)]:
            self.checkpoint(rounds)
            self.assertEqual(m.checkpoint_identity(rounds)["code_round"], index)

    def test_rejects_changed_weights(self):
        _, adapter = self.checkpoint()
        (adapter / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            m.checkpoint_identity(6)

    def test_training_metrics_excluded_like_original_commit(self):
        _, adapter = self.checkpoint()
        m.write_json(adapter / "training_metrics/rank0.json", {"loss": 1})
        m.checkpoint_identity(6)

    def test_local_mean_and_no_win_rate(self):
        folder, _ = self.completed_case()
        result = m.summarize_case(folder)
        self.assertEqual((result["wer_percent"], result["wvmos"]), (25, 4))
        self.assertIsNone(result["win_rate"])
        self.assertEqual(result["samples"], 2)

    def test_rejects_failed_latest_sample_and_stale_summary(self):
        folder, records = self.completed_case()
        m.summarize_case(folder)
        records[0]["status"] = "failed"
        with (folder / "staged_evaluation/local_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(records[0]) + '\n')
        self.assertEqual(m.write_summary(self.root / "eval", "full"), 1)
        self.assertEqual(json.loads((folder / "local_summary.json").read_text())["status"], "incomplete")

    def test_rejects_nan_missing_ids_and_wrong_audio(self):
        folder, records = self.completed_case()
        path = folder / "staged_evaluation/local_metrics.jsonl"
        for mutation in ("nan", "missing", "wrong_audio"):
            changed = [dict(r) for r in records]
            if mutation == "nan": changed[0]["mos_score"] = float("nan")
            if mutation == "missing": changed.pop()
            if mutation == "wrong_audio": changed[0]["audio_out_path"] = str(self.root / "wrong.wav")
            path.write_text(''.join(json.dumps(r) + '\n' for r in changed))
            with self.assertRaises(ValueError):
                m.summarize_case(folder)

    def test_rejects_generation_identity_or_scoring_protocol_change(self):
        folder, _ = self.completed_case()
        path = folder / "staged_evaluation/local_metadata.json"
        meta = json.loads(path.read_text()); meta["whisper_model"] = "wrong"
        m.write_json(path, meta)
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            m.summarize_case(folder)
        genpath = folder / "generation_manifest.json"
        gen = json.loads(genpath.read_text()); gen["generation_identity_sha256"] = "wrong"
        m.write_json(genpath, gen)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            m.summarize_case(folder)

    def test_commands_use_existing_workers_without_judge(self):
        identity = {"adapter": {"path": "/checkpoint/tts_final"}}
        gen, local = m.commands(self.root / "output", identity, "full")
        self.assertIn("run_inference.sh", gen[1])
        self.assertIn("run_staged_local_metrics.sh", local[1])
        self.assertEqual(gen[gen.index("--batch-size") + 1], "8")
        self.assertEqual(local[local.index("--workers") + 1], "4")
        self.assertNotIn("--num-samples", gen)
        self.assertFalse(any("judge" in x or "run_evaluation" in x or "staged_score" in x for x in gen + local))


if __name__ == "__main__":
    unittest.main()
