import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v6_round_benchmarks as v6
from test_v5_tts_dsd_eval import DsdTests


class V6BenchmarkTests(unittest.TestCase):
    def test_round_selection_is_explicit_and_does_not_follow_latest(self):
        self.assertEqual(v6.rounds_arg('0,1,2'), [0, 1, 2])
        for value in ('0,0', '-1', ''):
            with self.assertRaises(ValueError):
                v6.rounds_arg(value)

    def test_dsd_uses_v6_path_and_restores_legacy_module(self):
        args = argparse.Namespace(train_run=Path('/v6/run'), tts_batch=8, workers=32,
                                  gpu_memory_gib=26, gpus='0,1,2,3,4,5,6,7')
        old = v6.dsd.ROUND
        with patch.object(v6.dsd, 'read_json', side_effect=FileNotFoundError('fixture')) as read:
            with self.assertRaises(FileNotFoundError):
                v6.tts_plan(args, 2)
            read.assert_called_once_with(Path('/v6/run/round_002/commit.json'))
        self.assertEqual(v6.dsd.ROUND, old)

    def test_caption_batch_budget_and_checkpoint_preserved(self):
        c = argparse.Namespace(python='python', model='base', adapter='v6/round_002/adapter')
        with tempfile.TemporaryDirectory() as d:
            runner = v6.CaptionRunner(Path(d), 4, 26)
            command = v6.caption.commands(c, 'emotiontalk', 'full', Path(d))[0]
            actual = runner.prepare_command(command)
            self.assertEqual(actual[actual.index('--batch-size')+1], '4')
            self.assertEqual(actual[actual.index('--gpu-memory-gib')+1], '26')
            self.assertEqual(actual[actual.index('--adapter-dir')+1], c.adapter)

    def test_caption_schedules_eight_workers_and_skips_completed_tasks(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as d:
            runner = v6.CaptionRunner(Path(d), 4, 26)
            c = argparse.Namespace(candidate_id='v6_r0')
            with patch.object(v6, 'ThreadPoolExecutor', wraps=ThreadPoolExecutor) as pool, \
                 patch.object(v6.caption, 'task_status', return_value=(True, 'done')), \
                 patch.object(runner, 'run_command') as command:
                runner.stage([c], 'full', list(map(str, range(8))))
                self.assertEqual(pool.call_args.kwargs['max_workers'], 8)
                command.assert_not_called()

    def test_dsd_dry_run_scoring_and_resume_cannot_appear_as_real_score(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            plan, samples = DsdTests().fixture(out)
            plan.update(expected_audios=4, round=2, checkpoint_commit_sha256='v6fixture')
            v6.dsd.merge(out, plan, samples)
            with patch.object(v6.api, 'load_config', side_effect=AssertionError('No credentials/network for smoke')):
                v6.judge_round(out, plan, 2, dry_run=True)
                v6.judge_round(out, plan, 2, dry_run=True)
            directory = out / (v6.api.TAG + '_dry_run')
            result = json.loads((directory / 'dsd_summary.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(result['candidate'], 'midasheng_v6')
            self.assertEqual(result['round'], 2)
            self.assertIsNone(result['bilingual_percentage'])
            self.assertEqual(len((directory / 'judge_results.jsonl').read_text().splitlines()), 4)

    def test_judge_rejects_audio_from_another_adapter_before_api(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            plan, samples = DsdTests().fixture(out)
            v6.dsd.merge(out, plan, samples)
            plan.update(expected_audios=4, adapter={'weights_sha256': 'wrong'})
            with patch.object(v6.api, 'load_config') as config:
                with self.assertRaisesRegex(ValueError, 'identity/count'):
                    v6.judge_round(out, plan, 32)
                config.assert_not_called()


if __name__ == '__main__':
    unittest.main()
