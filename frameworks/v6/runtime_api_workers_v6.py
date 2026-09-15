"""Scheduling-only entry point; frozen training code/config/cache identity stay intact.

Keep this extension at the project root, outside the frozen package/scripts trees.
The original LabelService methods, prompts, parsing and cache keys are unchanged.
"""
import json
import math
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path


def worker_limits():
    limits = {
        'qwen35': int(os.environ.get('DUALISL_QWEN_API_WORKERS', '2')),
        'gemini': int(os.environ.get('DUALISL_GEMINI_API_WORKERS', '64')),
    }
    if any(v < 1 for v in limits.values()):
        raise ValueError('API worker counts must be positive integers')
    return limits


def retry_settings():
    settings = {
        'qwen_requests_per_minute': float(os.environ.get('DUALISL_QWEN_API_RPM', '30')),
        'retry_initial_seconds': float(os.environ.get('DUALISL_API_RETRY_SECONDS', '60')),
        'retry_max_seconds': float(os.environ.get('DUALISL_API_RETRY_MAX_SECONDS', '900')),
        'rate_limit_initial_seconds': 60.0,
        'rate_limit_max_seconds': 300.0,
    }
    if any(not math.isfinite(v) or v <= 0 for v in settings.values()):
        raise ValueError('Rate and retry settings must be finite positive numbers')
    if settings['retry_max_seconds'] < settings['retry_initial_seconds']:
        raise ValueError('Maximum retry wait must not be less than initial wait')
    return settings


class ApiPacer:
    """Shared per-model start pacing and cooldown, including original retries."""
    def __init__(self, rpm, settings, clock=time.monotonic):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.settings = settings
        self.clock = clock
        self.lock = threading.Lock()
        self.next_start = 0.0
        self.cooldown_until = 0.0
        self.rate_limit_rounds = 0

    def acquire(self, cancel):
        while True:
            if cancel.is_set():
                raise CancelledError('API scheduling interrupted')
            with self.lock:
                now = self.clock()
                delay = max(self.next_start, self.cooldown_until) - now
                if delay <= 0:
                    self.next_start = now + self.interval
                    return
            if cancel.wait(delay):
                raise CancelledError('API scheduling interrupted')

    def rate_limited(self):
        with self.lock:
            now = self.clock()
            # Several already-in-flight 429s count as one overload episode.
            if now < self.cooldown_until:
                return None
            delay = min(self.settings['rate_limit_max_seconds'],
                        self.settings['rate_limit_initial_seconds'] * 2 ** min(self.rate_limit_rounds, 10))
            self.rate_limit_rounds += 1
            self.cooldown_until = now + delay
            if self.interval:
                self.interval = max(self.interval, min(10.0, self.interval * 2))
            return {'wait_seconds': delay,
                    'effective_requests_per_minute': 60.0 / self.interval if self.interval else None}


def install_transport_pacing():
    from dual_isl_train import api_transport
    if getattr(api_transport.request_with_metadata, '_runtime_pacing', False):
        return
    original = api_transport.request_with_metadata

    def paced(service, model, path, prompt):
        pacers = getattr(service, '_runtime_api_pacers', None)
        if pacers is None:
            return original(service, model, path, prompt)
        pacer = pacers[model]
        pacer.acquire(service._runtime_api_cancel)
        service.event(kind='paced_request_start', model=model)
        try:
            return original(service, model, path, prompt)
        except CancelledError:
            raise
        except Exception as exc:
            # Never log response bodies, URLs or credentials. Preserve original exception.
            match = re.search(r'\bHTTP\s+(\d{3})\b', str(exc))
            status = int(match.group(1)) if match else None
            service.event(kind='transport_failure', model=model,
                          http_status=status, error_type=type(exc).__name__)
            if status == 429:
                cooldown = pacer.rate_limited()
                if cooldown:
                    service.event(kind='api_rate_limit_wait', model=model, **cooldown)
                    print(json.dumps({'phase': 'api_rate_limit_wait', 'model': model, **cooldown}), flush=True)
            raise

    paced._runtime_pacing = True
    api_transport.request_with_metadata = paced


def remote_until_complete(service, row, model, cancel, settings):
    from dual_isl_train.reward_v6 import EvaluationPending
    retries = 0
    while not cancel.is_set():
        try:
            return service._remote(row, model)
        except EvaluationPending:
            # The frozen evaluator exhausted its short retry cycle. Keep this
            # task pending; never fabricate a label/reward or abort the round.
            if cancel.is_set():
                raise CancelledError('API scheduling interrupted')
            delay = min(settings['retry_max_seconds'],
                        settings['retry_initial_seconds'] * 2 ** min(retries, 20))
            retries += 1
            service.event(kind='api_task_retry_wait', model=model, sample_id=row['id'],
                          retry_cycle=retries, wait_seconds=delay)
            print(json.dumps({'phase': 'api_task_retry_wait', 'model': model,
                              'sample_id': row['id'], 'retry_cycle': retries, 'wait_seconds': delay}), flush=True)
            if cancel.wait(delay):
                raise CancelledError('API scheduling interrupted')
    raise CancelledError('API scheduling interrupted')


def parallel_attributes(self, rows):
    from dual_isl_train.io import atomic_json, sha256_file
    from dual_isl_train.labeling_v6 import MODELS
    from dual_isl_train.reward_v6 import EvaluationPending

    limits = worker_limits()
    settings = retry_settings()
    cancel = threading.Event()
    self._runtime_api_cancel = cancel
    if not hasattr(self, '_runtime_api_pacers'):
        self._runtime_api_pacers = {
            m: ApiPacer(settings['qwen_requests_per_minute'] if m == 'qwen35' else None, settings)
            for m in MODELS
        }
    audit = {
        'api_workers_per_model': limits,
        'total_api_workers': sum(limits.values()),
        'frozen_shared_api_workers': self.cfg['api_workers'],
        'runtime_extension_sha256': sha256_file(__file__),
        'persistent_api_retry': True,
        **settings,
    }
    self.event(kind='runtime_api_concurrency', **audit)
    atomic_json(self.cache / 'runtime_api_concurrency.json', audit)
    print(json.dumps({'phase': 'api_concurrency', **audit}), flush=True)
    values = {r['id']: {} for r in rows}
    errors = []
    pools = {m: ThreadPoolExecutor(max_workers=limits[m], thread_name_prefix='label-' + m)
             for m in MODELS}
    try:
        futures = {pools[m].submit(remote_until_complete, self, row, m, cancel, settings): (row['id'], m)
                   for row in rows for m in MODELS}
        # Preserve the original local-emotion scheduling and result semantics.
        try:
            local = self._local(rows, 'emotion')
        except Exception as exc:
            local = {}
            errors.append(exc)
        for future in as_completed(futures):
            sid, _ = futures[future]
            try:
                values[sid].update(future.result())
            except Exception as exc:
                errors.append(exc)
    except BaseException:
        # Ctrl+C should not drain thousands of not-yet-started API jobs.
        # In-flight calls retain the original timeout and write successful caches.
        cancel.set()
        for pool in pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        for pool in pools.values():
            pool.shutdown(wait=True)
    if errors:
        raise EvaluationPending(f'{len(errors)} evaluator tasks failed; resume cached run. First: {errors[0]}')
    for sid in values:
        values[sid]['emotion'] = local[sid]['emotion']
    return values


def stop_labeling(root):
    """User-invoked SIGTERM of this run's driver, only during API labeling.

The legacy executor drains its entire queue on Ctrl+C. This command avoids that
wait. It refuses GPU-child phases and never sends SIGKILL or broad pkill signals.
"""
    import fcntl
    root = root.resolve()
    with (root / '.driver.lock').open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            print('No driver holds this run lock; no process was stopped.')
            return
    status = json.loads((root / 'status.json').read_text())
    if status['phase'] not in ('attribute_labels', 'reference_labels'):
        raise RuntimeError('Refusing API-only stop during phase ' + status['phase'] + '; use Ctrl+C in the training terminal.')
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            i = args.index('-m')
            if (args[i + 1] in ('scripts.v6_launcher', 'runtime_api_workers_v6')
                    and args[i + 2] in ('train', 'resume', 'gpu-smoke', 'prepare')
                    and args[i + 3] == str(root)):
                matches.append(int(proc.name))
        except (OSError, ValueError, IndexError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f'Expected one local driver, found {matches}. Run this command on the training server.')
    pid = matches[0]
    children_path = Path(f'/proc/{pid}/task/{pid}/children')
    if children_path.read_text().strip():
        raise RuntimeError('Driver still has subprocesses; wait for local evaluation to finish before stopping.')
    print(f'Sending SIGTERM to labeling driver PID {pid} for {root}', flush=True)
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        with (root / '.driver.lock').open('r+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(.2)
            else:
                print('Driver exited and run lock released. Completed stages and caches are retained.')
                return
    raise RuntimeError('Driver lock still held; do not start a second driver yet.')


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: runtime_api_workers_v6 MODE ABS_RUN_DIR')
    if sys.argv[1] == 'stop-labeling':
        stop_labeling(Path(sys.argv[2]))
        return
    limits = worker_limits()
    retry_settings()
    install_transport_pacing()
    from dual_isl_train.labeling_v6 import LabelService
    LabelService.attributes = parallel_attributes
    print(json.dumps({'runtime_api_workers_per_model': limits,
                      'note': 'Only API scheduling overrides frozen config; model/reward/cache contracts are unchanged.'}), flush=True)
    from scripts.v6_launcher import main as launch
    launch()


if __name__ == '__main__':
    main()
