import collections
import json
import os
import tempfile
import threading
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runtime_api_workers_v6 import parallel_attributes, worker_limits, stop_labeling
from runtime_api_workers_v6 import ApiPacer, retry_settings, remote_until_complete, install_transport_pacing
from concurrent.futures import CancelledError
from dual_isl_train.reward_v6 import EvaluationPending


class RuntimeConcurrencyTests(unittest.TestCase):
    def service(self, path):
        return SimpleNamespace(cfg={'api_workers': 4}, cache=Path(path),
                               event=lambda **kw: None,
                               _local=lambda rows, phase: {r['id']: {'emotion': 'happy'} for r in rows})

    def test_models_have_independent_concurrency_and_same_labels(self):
        with tempfile.TemporaryDirectory() as d:
            service = self.service(d)
            gate = threading.Barrier(4, timeout=5)
            lock = threading.Lock()
            active = collections.Counter()
            peak = collections.Counter()
            calls = collections.Counter()
            def remote(row, model):
                with lock:
                    active[model] += 1
                    peak[model] = max(peak[model], active[model])
                    calls[model] += 1
                try:
                    if int(row['id']) < 2:
                        gate.wait()
                    return {'pitch_level': 'medium'} if model == 'gemini' else {'gender': 'female', 'emotion_intensity': 'low'}
                finally:
                    with lock:
                        active[model] -= 1
            service._remote = remote
            rows = [{'id': str(i)} for i in range(4)]
            with patch.dict(os.environ, {'DUALISL_QWEN_API_WORKERS': '2', 'DUALISL_GEMINI_API_WORKERS': '2'}):
                result = parallel_attributes(service, rows)
            self.assertEqual(peak, {'qwen35': 2, 'gemini': 2})
            self.assertEqual(calls, {'qwen35': 4, 'gemini': 4})
            self.assertEqual(result, {str(i): {'gender': 'female', 'emotion_intensity': 'low', 'pitch_level': 'medium', 'emotion': 'happy'} for i in range(4)})
            audit = json.loads((Path(d) / 'runtime_api_concurrency.json').read_text())
            self.assertEqual(audit['total_api_workers'], 4)
            self.assertEqual(service.cfg, {'api_workers': 4})

    def test_failed_request_stays_pending_other_jobs_finish(self):
        with tempfile.TemporaryDirectory() as d:
            service = self.service(d)
            calls = []
            def remote(row, model):
                calls.append((row['id'], model))
                if row['id'] == '0' and model == 'gemini':
                    raise TimeoutError('mock failure')
                return {'gender': 'unknown'} if model == 'qwen35' else {'pitch_level': 'high'}
            service._remote = remote
            with self.assertRaises(EvaluationPending):
                parallel_attributes(service, [{'id': str(i)} for i in range(4)])
            self.assertEqual(len(calls), 8)

    def test_invalid_limits_rejected(self):
        with patch.dict(os.environ, {'DUALISL_QWEN_API_WORKERS': '0'}):
            with self.assertRaises(ValueError):
                worker_limits()

    def test_stop_unlocked_run_never_signals(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / '.driver.lock').touch()
            with patch('runtime_api_workers_v6.os.kill') as kill:
                stop_labeling(Path(d))
                kill.assert_not_called()

    def test_stop_only_matching_labeling_driver_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'status.json').write_text(json.dumps({'phase': 'attribute_labels'}))
            code = ('import fcntl,sys,time; '
                    'f=open(sys.argv[-1]+"/.driver.lock","a"); '
                    'fcntl.flock(f,fcntl.LOCK_EX); print("ready",flush=True); time.sleep(30)')
            proc = subprocess.Popen([sys.executable, '-c', code, '-m', 'scripts.v6_launcher', 'resume', str(root)],
                                    stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(proc.stdout.readline().strip(), 'ready')
                stop_labeling(root)
                self.assertEqual(proc.wait(timeout=5), -15)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
                proc.stdout.close()

    def test_stop_refuses_training_phase(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / '.driver.lock').touch()
            (Path(d) / 'status.json').write_text(json.dumps({'phase': 'caption_grpo'}))
            with patch('fcntl.flock', side_effect=BlockingIOError), patch('runtime_api_workers_v6.os.kill') as kill:
                with self.assertRaises(RuntimeError):
                    stop_labeling(Path(d))
                kill.assert_not_called()


class PersistentRetryTests(unittest.TestCase):
    def test_exhausted_task_keeps_retrying_until_success_without_fake_label(self):
        waits = []
        cancel = SimpleNamespace(is_set=lambda: False, wait=lambda delay: waits.append(delay) or False)
        events = []
        from unittest.mock import Mock
        remote = Mock(side_effect=[EvaluationPending('failed')] * 6 + [{'gender': 'female'}])
        service = SimpleNamespace(_remote=remote, event=lambda **kw: events.append(kw))
        settings = retry_settings()
        result = remote_until_complete(service, {'id': 'a'}, 'qwen35', cancel, settings)
        self.assertEqual(result, {'gender': 'female'})
        self.assertEqual(remote.call_count, 7)
        self.assertEqual(waits, [60, 120, 240, 480, 900, 900])
        self.assertTrue(all(x['kind'] == 'api_task_retry_wait' for x in events))

    def test_cancellation_stops_retry_wait(self):
        from unittest.mock import Mock
        cancel = SimpleNamespace(is_set=lambda: False, wait=lambda delay: True)
        remote = Mock(side_effect=EvaluationPending('failed'))
        with self.assertRaises(CancelledError):
            remote_until_complete(SimpleNamespace(_remote=remote, event=lambda **kw: None),
                                  {'id': 'a'}, 'qwen35', cancel, retry_settings())
        self.assertEqual(remote.call_count, 1)

    def test_global_pacing_and_429_cooldown_apply_to_other_threads(self):
        now = [0.0]
        waits = []
        def wait(seconds):
            waits.append(seconds)
            now[0] += seconds
            return False
        cancel = SimpleNamespace(is_set=lambda: False, wait=wait)
        pacer = ApiPacer(30, retry_settings(), clock=lambda: now[0])
        pacer.acquire(cancel)
        pacer.acquire(cancel)
        self.assertEqual(waits, [2.0])
        limited = pacer.rate_limited()
        self.assertEqual(limited, {'wait_seconds': 60.0, 'effective_requests_per_minute': 15.0})
        self.assertIsNone(pacer.rate_limited())
        pacer.acquire(cancel)
        self.assertEqual(waits[-1], 60.0)
        pacer.acquire(cancel)
        self.assertEqual(waits[-1], 4.0)
        self.assertEqual(pacer.rate_limited()['wait_seconds'], 120.0)

    def test_429_detection_does_not_log_response_body_and_keeps_original_exception(self):
        from dual_isl_train import api_transport
        from unittest.mock import Mock
        failure = RuntimeError('HTTP 429: secret provider response')
        events = []
        pacer = Mock()
        pacer.rate_limited.return_value = {'wait_seconds': 60, 'effective_requests_per_minute': 15}
        service = SimpleNamespace(_runtime_api_pacers={'qwen35': pacer},
                                  _runtime_api_cancel=threading.Event(), event=lambda **kw: events.append(kw))
        with patch.object(api_transport, 'request_with_metadata', side_effect=failure) as original:
            original._runtime_pacing = False
            install_transport_pacing()
            with self.assertRaises(RuntimeError) as caught:
                api_transport.request_with_metadata(service, 'qwen35', 'audio', 'prompt')
            self.assertIs(caught.exception, failure)
        self.assertNotIn('secret', json.dumps(events))
        self.assertEqual(events[1]['http_status'], 429)
        pacer.acquire.assert_called_once()
        pacer.rate_limited.assert_called_once()

    def test_parallel_stage_completes_after_pending_api_recovers(self):
        with tempfile.TemporaryDirectory() as d:
            calls = collections.Counter()
            def remote(row, model):
                calls[(row['id'], model)] += 1
                if model == 'qwen35' and calls[(row['id'], model)] == 1:
                    raise EvaluationPending('temporary')
                return {'gender': 'female'} if model == 'qwen35' else {'pitch_level': 'high'}
            service = SimpleNamespace(cfg={'api_workers': 4}, cache=Path(d), event=lambda **kw: None,
                                      _remote=remote, _local=lambda rows, phase: {'a': {'emotion': 'happy'}})
            with patch.dict(os.environ, {'DUALISL_API_RETRY_SECONDS': '0.001'}):
                result = parallel_attributes(service, [{'id': 'a'}])
            self.assertEqual(result, {'a': {'gender': 'female', 'pitch_level': 'high', 'emotion': 'happy'}})
            self.assertEqual(calls[('a', 'qwen35')], 2)
            self.assertEqual(calls[('a', 'gemini')], 1)

    def test_nonfinite_rate_rejected(self):
        with patch.dict(os.environ, {'DUALISL_QWEN_API_RPM': 'nan'}):
            with self.assertRaises(ValueError):
                retry_settings()


if __name__ == '__main__':
    unittest.main()
