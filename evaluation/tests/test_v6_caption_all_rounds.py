import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v6_caption_all_rounds as v6


def candidate(number):
    return v6.legacy.existing.Candidate(
        f'midasheng_v6_r{number}', 'midasheng_v6', f'r{number}', 'midasheng', number,
        '/base', f'/train/round_{number:03}/caption_final', 'python', 'sdpa', 'ready', f'hash{number}')


class CaptionOnlyTests(unittest.TestCase):
    def test_reuse_rejects_changed_checkpoint_protocol_code_or_batch(self):
        c = candidate(0)
        old = {'candidates': [asdict(c)], 'caption_batch': 4, 'resources': {'data': 'original'}}
        self.assertTrue(v6.reuse_compatible(old, c, {'data': 'original'}, 4))
        self.assertFalse(v6.reuse_compatible(old, c, {'data': 'changed'}, 4))
        self.assertFalse(v6.reuse_compatible(old, c, {'data': 'original'}, 8))
        self.assertFalse(v6.reuse_compatible(old, candidate(1), {'data': 'original'}, 4))
        old['candidates'][0]['detail'] = 'different checkpoint digest'
        self.assertFalse(v6.reuse_compatible(old, c, {'data': 'original'}, 4))

    def test_reused_and_locally_completed_tasks_never_scheduled_again(self):
        args = argparse.Namespace(output_root=Path('/new'))
        candidates = [candidate(i) for i in range(10)]
        record = {'reused': {v6.task_key(c, s): {} for c in candidates[:4] for s in v6.SUITES}}
        with patch.object(v6.legacy, 'task_status', side_effect=lambda root, suite, size, c:
                          (c.round == 4 and suite == 'emotiontalk', 'fixture')):
            pending = v6.pending_tasks(args, record, candidates, 'full')
        self.assertEqual(len(pending), 17)
        self.assertTrue(all(c.round >= 4 for c, _ in pending))
        self.assertEqual(pending[0][1], 'emotiontalk')
        self.assertEqual(len({v6.task_key(c, s) for c, s in pending}), 17)

    def test_gpu_guard_keeps_adapter_and_cpu_scorer_untouched(self):
        args = argparse.Namespace(output_root=Path('/out'), batch_size=4, gpu_memory_gib=26)
        runner = v6.CaptionRunner(args)
        c = candidate(4)
        for suite in v6.SUITES:
            for original in v6.legacy.commands(c, suite, 'full', Path('/result')):
                cmd = runner.prepare_command(original)
                if '--adapter-dir' in original:
                    self.assertEqual(cmd[cmd.index('--adapter-dir')+1], c.adapter)
                    self.assertEqual(cmd[cmd.index('--gpu-memory-gib')+1], '26')
                if '--batch-size' in cmd:
                    self.assertEqual(cmd[cmd.index('--batch-size')+1], '4')
                self.assertNotIn('tts', ' '.join(cmd).lower())
        cpu = v6.legacy.commands(c, 'paraspeechcaps', 'full', Path('/out'))[1]
        self.assertEqual(runner.prepare_command(cpu), cpu)

    def test_frozen_reused_artifact_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = argparse.Namespace(train_run=Path('/train'), output_root=root,
                                      reuse_roots=[], batch_size=4)
            with patch.object(v6.legacy, 'resource_identity', return_value={}):
                record = v6.inventory(args, [candidate(0)])
                artifact = root/'old_score.json'; artifact.write_text('original')
                record['reused']['fixture'] = {'artifacts': {str(artifact): v6.legacy.existing.sha256_file(artifact)}}
                v6.legacy.write_json(root/'inventory.json', record)
                v6.inventory(args, [candidate(0)])
                artifact.write_text('modified')
                with self.assertRaisesRegex(ValueError, 'Reused result changed'):
                    v6.inventory(args, [candidate(0)])

    def test_no_gpu_process_started_when_every_task_is_reused(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(output_root=Path(d), batch_size=4, gpu_memory_gib=26,
                                      gpus=[str(i) for i in range(8)])
            cs = [candidate(0)]
            record = {'reused': {v6.task_key(cs[0], s): {} for s in v6.SUITES}}
            runner = v6.CaptionRunner(args)
            with patch.object(runner, 'run_command') as run, \
                 patch.object(v6, 'ThreadPoolExecutor', wraps=ThreadPoolExecutor) as pool:
                runner.stage(record, cs, 'full')
                run.assert_not_called()
                self.assertEqual(pool.call_args.kwargs['max_workers'], 8)

    def test_invalid_rounds_and_gpu_indices_rejected(self):
        self.assertEqual(v6.round_numbers('9,0,4'), [0,4,9])
        for value in ('', '-1', '1,1', 'x'):
            with self.assertRaises(argparse.ArgumentTypeError):
                v6.round_numbers(value)
        for value in ('', '-1', '0,0', 'GPU0'):
            with self.assertRaises(argparse.ArgumentTypeError):
                v6.gpu_numbers(value)


if __name__ == '__main__':
    unittest.main()
