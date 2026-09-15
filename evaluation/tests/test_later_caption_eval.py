from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "later_caption_eval.py"
SPEC = importlib.util.spec_from_file_location("later_caption_eval", MODULE_PATH)
assert SPEC and SPEC.loader
later = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = later
SPEC.loader.exec_module(later)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_adapter(path: Path, model: Path, payload: bytes = b"weights") -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "adapter_config.json", {"base_model_name_or_path": str(model.resolve())})
    (path / "adapter_model.safetensors").write_bytes(payload)


class LaterCaptionEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.caption = Path(self.temp.name) / "Caption"
        self.qwen_model = self.caption / "models/Qwen3-Omni-30B-A3B-Captioner"
        self.mida_model = self.caption / "models/MiDashengLM-7B-1021-BF16"
        self.qwen_model.mkdir(parents=True)
        self.mida_model.mkdir(parents=True)
        qwen_run = self.caption / "DualISL_Train/runs/dual_recursive_8gpu_h100_qwen_from_r2_20260831_run01"
        mida_run = self.caption / "DualISL_Train/runs/dual_recursive_8gpu_h100_midasheng_from_r2_20260831_run01"
        for number in range(3, 5):
            root = qwen_run / f"round_{number:03d}"
            make_adapter(root / "checkpoints/caption_final", self.qwen_model, str(number).encode())
            write_json(root / "commit.json", {"round": number})
            write_json(root / "summary.json", {"round": number})
        for number in range(3, 13):
            root = mida_run / f"round_{number:03d}"
            make_adapter(root / "checkpoints/caption_final", self.mida_model, str(number).encode())
            write_json(root / "commit.json", {"round": number})
            write_json(root / "summary.json", {"round": number})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def commit_reward_round(self, number: int, *, bad_hash: bool = False) -> Path:
        root = (
            self.caption
            / "DualISL_Train_RewardV2/runs/dual_recursive_8gpu_h100_midasheng_reward_v2_20260901_run01"
            / f"round_{number:03d}"
        )
        adapter = root / "checkpoints/caption_final"
        make_adapter(adapter, self.mida_model, f"reward-{number}".encode())
        digest = "bad" if bad_hash else later.hash_path(adapter)
        write_json(root / "commit.json", {"round": number, "captioner": {"path": str(adapter.resolve()), "sha256": digest}})
        return adapter

    def test_candidate_discovery_gates_reward_commits(self) -> None:
        for number in range(4):
            self.commit_reward_round(number)
        candidates = later.discover_candidates(self.caption)
        counts = {status: sum(item.status == status for item in candidates) for status in ("ready", "pending", "invalid")}
        self.assertEqual(counts, {"ready": 16, "pending": 6, "invalid": 0})
        self.assertEqual([item.candidate_id for item in candidates if item.trajectory == "qwen_v1"], ["qwen_v1_r3", "qwen_v1_r4"])
        self.assertEqual(len([item for item in candidates if item.trajectory == "midasheng_v1"]), 10)

    def test_reward_hash_mismatch_is_invalid(self) -> None:
        self.commit_reward_round(0, bad_hash=True)
        candidate = next(item for item in later.discover_candidates(self.caption) if item.candidate_id == "midasheng_rewardv2_r0")
        self.assertEqual(candidate.status, "invalid")
        self.assertIn("hash mismatch", candidate.detail)

    @patch.dict(later.os.environ, {"LATER_CAPTION_PROFILE": "midasheng_rewardv2_v3"})
    def test_v2_continuation_and_v3_discovery(self) -> None:
        self.commit_reward_round(9)
        runs = [
            ("DualISL_Train_RewardV2/runs/dual_recursive_8gpu_h100_midasheng_reward_v2_from_r9_20260904_run01", 10),
            ("DualISL_Train_RewardV3/runs/midasheng_7b_reward_v3_10rounds_20260905_run01", 0),
        ]
        for run, number in runs:
            root = self.caption / run / f"round_{number:03d}"
            adapter = root / "checkpoints/caption_final"
            make_adapter(adapter, self.mida_model)
            write_json(root / "commit.json", {
                "round": number,
                "captioner": {"path": str(adapter.resolve()), "sha256": later.hash_path(adapter)},
            })
        candidates = later.discover_candidates(self.caption)
        self.assertEqual(len(candidates), 30)
        self.assertEqual({c.candidate_id for c in candidates if c.status == "ready"}, {
            "midasheng_rewardv2_r9", "midasheng_rewardv2_r10", "midasheng_rewardv3_r0",
        })
        self.assertEqual(sum(c.status == "pending" for c in candidates), 27)
        summary = later.write_summaries(self.caption, Path(self.temp.name) / "results")
        self.assertEqual(summary.name, "summary_midasheng_rewardv2_v3")
        self.assertIn("## midasheng_rewardv3", (summary / "results.md").read_text())
        self.assertNotIn("qwen_v1", (summary / "results.md").read_text())
        adapter.joinpath("adapter_model.safetensors").write_bytes(b"changed")
        candidates = later.discover_candidates(self.caption)
        self.assertEqual(next(c.status for c in candidates if c.candidate_id == "midasheng_rewardv3_r0"), "invalid")

    @patch.dict(later.os.environ, {"LATER_CAPTION_PROFILE": "midasheng_rewardv3"})
    def test_v3_only_scope_commit_gate_and_summary(self) -> None:
        self.commit_reward_round(0, bad_hash=True)
        run = self.caption / "DualISL_Train_RewardV3/runs/midasheng_7b_reward_v3_10rounds_20260905_run01"
        for number in range(10):
            root = run / f"round_{number:03d}"
            adapter = root / "checkpoints/caption_final"
            make_adapter(adapter, self.mida_model, str(number).encode())
            write_json(root / "commit.json", {
                "round": number,
                "captioner": {"path": str(adapter.resolve()), "sha256": later.hash_path(adapter)},
            })
        candidates = later.discover_candidates(self.caption)
        self.assertEqual(len(candidates), 10)
        self.assertTrue(all(c.status == "ready" and c.trajectory == "midasheng_rewardv3" for c in candidates))
        output = Path(self.temp.name) / "outputs"
        for size in ("smoke", "full"):
            rows = later.task_rows(self.caption, output, "all", size)
            self.assertEqual(len(rows), 30)
            self.assertEqual({r["candidate_id"] for r in rows}, {f"midasheng_rewardv3_r{n}" for n in range(10)})
        summary = later.write_summaries(self.caption, output)
        self.assertEqual(summary.name, "summary_midasheng_rewardv3")
        entries = later.read_json(summary / "results.json")["entries"]
        self.assertEqual(len(entries), 30)
        self.assertTrue(all(e["trajectory"] == "midasheng_rewardv3" for e in entries))
        adapter.joinpath("adapter_model.safetensors").write_bytes(b"changed")
        candidates = later.discover_candidates(self.caption)
        self.assertEqual(candidates[-1].status, "invalid")
        self.assertIn("hash mismatch", candidates[-1].detail)

    def test_empty_output_builds_48_longest_first_tasks(self) -> None:
        for number in range(4):
            self.commit_reward_round(number)
        rows = later.task_rows(self.caption, Path(self.temp.name) / "outputs", "all", "smoke")
        self.assertEqual(len(rows), 48)
        self.assertEqual(rows[0]["suite"], "emotiontalk")
        self.assertEqual(rows[0]["family"], "qwen")
        self.assertTrue(all(rows[index]["priority"] >= rows[index + 1]["priority"] for index in range(len(rows) - 1)))

    @patch.dict(later.os.environ, {"LATER_CAPTION_PROFILE": "qwen25_v1_v2", "QWEN25_PY": "/custom/python"})
    def test_qwen25_snapshot_requires_commit_and_checks_identity(self) -> None:
        model = self.caption / "models/Qwen2.5-Omni-3B"
        run = self.caption / "DualISL_Train/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_20260903_run01"
        for number in range(2):
            adapter = run / f"round_{number:03d}/checkpoints/caption_final"
            make_adapter(adapter, model)
            if number == 0:
                write_json(run / "round_000/commit.json", {
                    "round": 0, "captioner": {"path": str(adapter), "sha256": later.hash_path(adapter)},
                })
        candidates = later.discover_candidates(self.caption)
        self.assertEqual(len(candidates), 20)
        ready = [c for c in candidates if c.status == "ready"]
        self.assertEqual([c.candidate_id for c in ready], ["qwen25_v1_r0"])
        self.assertEqual(ready[0].python, "/custom/python")
        self.assertEqual(candidates[1].status, "pending")
        out = Path(self.temp.name) / "out"
        rows = later.task_rows(self.caption, out, "all", "full", candidates)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["family"] == "qwen25" for row in rows))
        c = ready[0]
        result = later.output_dir(out, "stylecap", "smoke", c.candidate_id)
        result.mkdir(parents=True)
        (result / "predictions.jsonl").write_text('{}\n' * 4)
        identity = {
            "model_dir": c.model, "backend": "qwen25", "attn_backend": c.attn,
            "protocol": "stylecap-promptspeech-speaker-open-mcq-v1",
            "adapter": {"path": c.adapter, "weights_sha256": later.sha256_file(Path(c.adapter) / "adapter_model.safetensors")},
        }
        write_json(result / "run_metadata.json", {"prediction_rows": 4, "identity": identity})
        self.assertTrue(later.completion_status(out, "stylecap", "smoke", c.candidate_id, c)[0])
        identity["backend"] = "qwen3"
        write_json(result / "run_metadata.json", {"prediction_rows": 4, "identity": identity})
        self.assertFalse(later.completion_status(out, "stylecap", "smoke", c.candidate_id, c)[0])
        summary = later.write_summaries(self.caption, out)
        self.assertEqual(summary.name, "summary_qwen25_v1_v2")
        Path(c.adapter, "adapter_model.safetensors").write_bytes(b"tampered")
        self.assertEqual(later.discover_candidates(self.caption)[0].status, "invalid")

    def test_frozen_inventory_does_not_gain_a_new_commit(self) -> None:
        for number in range(4):
            self.commit_reward_round(number)
        frozen = later.discover_candidates(self.caption)
        self.commit_reward_round(4)
        rows = later.task_rows(
            self.caption, Path(self.temp.name) / "outputs", "all", "smoke", frozen
        )
        self.assertEqual(len(rows), 48)
        self.assertNotIn("midasheng_rewardv2_r4", {row["candidate_id"] for row in rows})

    def test_completion_requires_adapter_identity(self) -> None:
        candidate = next(item for item in later.discover_candidates(self.caption) if item.candidate_id == "qwen_v1_r3")
        output_root = Path(self.temp.name) / "outputs"
        root = later.output_dir(output_root, "stylecap", "smoke", candidate.candidate_id)
        root.mkdir(parents=True)
        rows = [{"id": str(number)} for number in range(4)]
        (root / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        adapter_path = Path(candidate.adapter)
        adapter = {"path": str(adapter_path), "weights_sha256": later.sha256_file(adapter_path / "adapter_model.safetensors")}
        identity = {
            "adapter": adapter, "model_dir": candidate.model, "backend": "qwen3",
            "attn_backend": "flash_attention_2",
            "protocol": "stylecap-promptspeech-speaker-open-mcq-v1",
        }
        write_json(root / "run_metadata.json", {"prediction_rows": 4, "identity": identity})
        self.assertTrue(later.completion_status(output_root, "stylecap", "smoke", candidate.candidate_id, candidate)[0])
        adapter["weights_sha256"] = "wrong"
        write_json(root / "run_metadata.json", {"prediction_rows": 4, "identity": identity})
        ok, detail = later.completion_status(output_root, "stylecap", "smoke", candidate.candidate_id, candidate)
        self.assertFalse(ok)
        self.assertIn("identity mismatch", detail)

    def test_scheduler_source_uses_wait_n_p_and_global_queue(self) -> None:
        source = (MODULE_PATH.parent / "run_later_caption_benchmarks_8gpu.sh").read_text(encoding="utf-8")
        self.assertIn("wait -n -p done_pid", source)
        self.assertIn("if wait -n; then :; else :; fi", source)
        self.assertIn("find_eligible_task", source)
        self.assertIn("QWEN_RUNNING >= QWEN_LIMIT", source)
        self.assertIn("FREE_GPUS+=(\"${gpu}\")", source)
        self.assertNotIn("wait_for_batch", source)

    def test_metric_scalar_prefers_named_composite_score(self) -> None:
        self.assertEqual(
            later._metric_scalar(
                {"status": "ok", "corpus": {"cider_d": 0.01, "spice": 0.2, "spider": 0.105}},
                "spider",
            ),
            0.105,
        )
        self.assertEqual(
            later._metric_scalar(
                {"status": "ok", "corpus": {"sbert_sim": 0.95, "fer": 0.1, "fense": 0.87}},
                "fense",
            ),
            0.87,
        )
        self.assertEqual(
            later._metric_scalar(
                {
                    "status": "ok",
                    "corpus": {
                        "bert_score.precision": 0.72,
                        "bert_score.recall": 0.74,
                        "bert_score.f1": 0.73,
                    },
                },
                "bertscore",
            ),
            0.73,
        )


if __name__ == "__main__":
    unittest.main()
