import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_budget_exec as guard
import v5_caption_round_eval as caption
import v5_dsd_audio_api_judge as api
import v5_tts_dsd_eval as dsd


class RoundEvaluationTests(unittest.TestCase):
    def test_caption_gpu_command_preserves_adapter_and_bounds_batch(self):
        c = argparse.Namespace(python="python", model="base", adapter="round_001/adapter")
        with tempfile.TemporaryDirectory() as folder:
            runner = caption.BudgetRunner(Path(folder), 1, 26)
            command = caption.legacy.commands(c, "emotiontalk", "full", Path(folder))[0]
            actual = runner.prepare_command(command)
            self.assertEqual(actual[1], str(caption.GUARD))
            self.assertEqual(actual[actual.index("--batch-size") + 1], "1")
            self.assertEqual(actual[actual.index("--gpu-memory-gib") + 1], "26")
            self.assertEqual(actual[actual.index("--adapter-dir") + 1], "round_001/adapter")
            self.assertEqual(command[command.index("--batch-size") + 1], "4")

    def test_gpu_budget_rejects_shared_visible_devices_and_low_memory(self):
        torch = mock.Mock()
        torch.cuda.device_count.return_value = 2
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            guard.configure_budget(torch, 26)
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_properties.return_value.total_memory = 80 * 1024 ** 3
        torch.cuda.mem_get_info.return_value = (32 * 1024 ** 3, 80 * 1024 ** 3)
        guard.configure_budget(torch, 26)
        torch.cuda.set_per_process_memory_fraction.assert_called_once_with(26 / 80, 0)
        torch.cuda.mem_get_info.return_value = (20 * 1024 ** 3, 80 * 1024 ** 3)
        with self.assertRaisesRegex(RuntimeError, "free memory"):
            guard.configure_budget(torch, 26)

    def test_gpu_wrapper_forwards_script_arguments_and_import_path(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "child.py"
            script.write_text("import sys\nassert sys.argv[1:] == ['--batch-size', '1']\n")
            old_path = sys.path[:]
            try:
                with mock.patch.object(sys, "argv", ["guard", "--gpu-memory-gib", "26", str(script),
                                                     "--batch-size", "1"]), \
                     mock.patch.dict(sys.modules, {"torch": mock.Mock()}), \
                     mock.patch.object(guard, "configure_budget") as configure:
                    guard.main()
                    self.assertEqual(sys.path[0], folder)
                    configure.assert_called_once()
            finally:
                sys.path[:] = old_path

    def test_api_round_is_from_plan_and_rejects_first_round_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "dsd_plan.json").write_text(json.dumps({"round": 1, "adapter": {"hash": "second"}}))
            manifest = {"generation_identity": {"adapter": {"hash": "second"}}}
            self.assertEqual(api.selected_round(root, manifest, "v5", 1), 1)
            self.assertEqual(api.selected_round(root, manifest, "v5", None), 1)
            with self.assertRaisesRegex(ValueError, "round differs"):
                api.selected_round(root, manifest, "v5", 0)
            manifest["generation_identity"]["adapter"] = {"hash": "first"}
            with self.assertRaisesRegex(ValueError, "adapter differs"):
                api.selected_round(root, manifest, "v5", 1)
            self.assertIsNone(api.selected_round(root, manifest, "base", None))

    def test_base_comparison_uses_current_api_results(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(api, "BASE_RUN", Path(folder)):
            self.assertIsNone(api.same_api_base_score())
            target = Path(folder) / api.TAG / "summary.json"
            target.parent.mkdir()
            target.write_text(json.dumps({"complete": True, "expected": 2000,
                                          "bilingual_macro_average": 80.9}))
            self.assertEqual(api.same_api_base_score(), 80.9)

    def test_dsd_round_one_selects_second_round_commit_before_other_io(self):
        args = argparse.Namespace(batch_size=8, workers=32, gpu_memory_gib=26,
                                  gpus="3,4,5,6,7", round_index=1, mode="check")
        with mock.patch.object(dsd, "read_json", side_effect=FileNotFoundError("sentinel")) as read:
            with self.assertRaises(FileNotFoundError):
                dsd.make_plan(args)
            read.assert_called_once_with(dsd.ROUND.parent / "round_001/commit.json")


if __name__ == "__main__":
    unittest.main()
