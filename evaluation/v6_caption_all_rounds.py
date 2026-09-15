#!/usr/bin/env python3
"""Captioner-only evaluation of committed V6 rounds; reuse audited full results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import time

import midasheng_v4_v5_eval as legacy

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / 'DualRSI_Train_V6_AudioOnly4Attr/runs/midasheng_v6_8gpu_run01'
OUTPUT = ROOT / 'v6_caption_all_rounds_run01'
REUSE = [ROOT / 'v6_round000_002_benchmarks_run01', ROOT / 'v6_round003_benchmarks_run01']
GUARD = ROOT / 'gpu_budget_exec.py'
SUITES = ('emotiontalk', 'paraspeechcaps', 'stylecap')
VERSION = 'v6-caption-only-all-rounds-v1'


def round_numbers(value):
    try:
        numbers = [int(x) for x in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated zero-based rounds') from exc
    if not numbers or min(numbers) < 0 or len(set(numbers)) != len(numbers):
        raise argparse.ArgumentTypeError('Rounds must be distinct and nonnegative')
    return sorted(numbers)


def gpu_numbers(value):
    ids = value.split(',')
    if not ids or any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError('Use distinct physical GPU indices')
    return ids


def discover(train, numbers):
    candidates = legacy.existing.discover_reward_candidates(
        ROOT.parent, [(train, 'midasheng_v6', numbers)], True, True)
    for c in candidates:
        if c.status != 'ready':
            raise RuntimeError(f'{c.candidate_id}: {c.detail}')
        commit = legacy.existing.read_json(train / f'round_{c.round:03d}/commit.json')
        if commit.get('framework_version') != 'DualRSI-Train-V6-AudioOnly4Attr-0.6':
            raise ValueError('Not a committed V6 round: ' + c.candidate_id)
    # Only Captioner checkpoint contents are read/hashed. No TTS dependency.
    legacy.validate_candidates(candidates)
    return candidates


def artifacts(root, suite, candidate):
    out = legacy.existing.output_dir(root, suite, 'full', candidate.candidate_id)
    names = {
        'emotiontalk': ['run_metadata.json', 'predictions.jsonl', 'metrics_standard_public/scores.json'],
        'paraspeechcaps': ['outputs/run_metadata.json', 'outputs/predictions.jsonl',
                           'outputs/field_records.jsonl', 'reports/summary.json'],
        'stylecap': ['run_metadata.json', 'predictions.jsonl', 'evaluation_summary.json'],
    }[suite]
    return {str(out / name): legacy.existing.sha256_file(out / name) for name in names}


def reuse_compatible(old, c, resources, batch):
    return (old.get('caption_batch') == batch
            and any(v == asdict(c) for v in old.get('candidates', []))
            and all(old.get('resources', {}).get(k) == v for k, v in resources.items()))


def task_key(c, suite):
    return f'{c.candidate_id}/{suite}'


def inventory(args, candidates):
    shared = legacy.resource_identity()
    resources = dict(shared)
    for p in (Path(__file__), ROOT / 'run_v6_caption_all_rounds.sh', GUARD):
        resources[str(p)] = legacy.existing.sha256_file(p)
    identity = {'version': VERSION, 'train_run': str(args.train_run),
                'candidates': [asdict(c) for c in candidates], 'resources': resources,
                'batch_size': args.batch_size, 'suites': list(SUITES),
                'reuse_roots': [str(p) for p in args.reuse_roots]}
    target = args.output_root / 'inventory.json'
    if target.exists():
        old = legacy.existing.read_json(target)
        if old['identity'] != identity:
            raise ValueError('Checkpoint/code/data/batch changed; use a new output directory')
        for spec in old['reused'].values():
            for path, digest in spec['artifacts'].items():
                if legacy.existing.sha256_file(Path(path)) != digest:
                    raise ValueError('Reused result changed: ' + path)
        return old
    reused, rejected = {}, []
    for source in args.reuse_roots:
        if not (source / 'inventory.json').exists():
            continue
        prior = legacy.existing.read_json(source / 'inventory.json')
        for c in candidates:
            if not any(v.get('candidate_id') == c.candidate_id for v in prior.get('candidates', [])):
                continue
            if not reuse_compatible(prior, c, shared, args.batch_size):
                rejected.append({'source': str(source), 'candidate': c.candidate_id,
                                 'reason': 'checkpoint/code/data/batch identity differs; will evaluate locally'})
                continue
            for suite in SUITES:
                key = task_key(c, suite)
                if key in reused:
                    continue
                ok, _ = legacy.task_status(source / 'captioner', suite, 'full', c)
                if ok:
                    reused[key] = {'root': str(source / 'captioner'),
                                   'artifacts': artifacts(source / 'captioner', suite, c)}
    return {'identity': identity, 'reused': reused, 'reuse_rejections': rejected}


def result_root(args, record, c, suite):
    spec = record['reused'].get(task_key(c, suite))
    return Path(spec['root']) if spec else args.output_root / 'captioner'


def pending_tasks(args, record, candidates, size):
    tasks = []
    # Run the long EmotionTalk jobs first; remaining GPUs consume shorter tasks.
    for suite in SUITES:
        for c in candidates:
            if task_key(c, suite) in record['reused']:
                continue
            if not legacy.task_status(args.output_root / 'captioner', suite, size, c)[0]:
                tasks.append((c, suite))
    return tasks


def summarize(args, record, candidates):
    entries, table = [], []
    for c in candidates:
        values = {}
        for suite in SUITES:
            root = result_root(args, record, c, suite)
            ok, detail = legacy.task_status(root, suite, 'full', c)
            out = legacy.existing.output_dir(root, suite, 'full', c.candidate_id)
            result = legacy.existing.load_suite_result(suite, out) if ok else {}
            entries.append({'round_index': c.round, 'training_round': c.round + 1, 'suite': suite,
                            'complete': ok, 'detail': detail, 'result_path': str(out),
                            'reused': task_key(c, suite) in record['reused'], 'result': result})
            values[suite] = result
        et = values['emotiontalk'].get('tasks', {}).get('overall', {}).get('metrics', {})
        table.append({'round': c.round + 1, 'ET_SPIDEr': et.get('spider', {}).get('value'),
                      'ET_FENSE': et.get('fense', {}).get('value'),
                      'ParaSpeechCaps': values['paraspeechcaps'].get('final_score'),
                      'StyleCap': values['stylecap'].get('macro_average_accuracy')})
    # The requested main table omits StyleCap; complete results remain in JSON/CSV.
    columns = list(table[0])[:-1]
    lines = ['# V6 Captioner 全轮次评测', '',
             'EmotionTalk 使用 overall 的 SPIDEr/FENSE；缺失结果记为 —，不记为零。', '',
             '| 训练轮次 | ET SPIDEr | ET FENSE | ParaSpeechCaps |', '|---|---:|---:|---:|']
    for row in table:
        lines.append('| ' + ' | '.join(str(row[k]) if k == 'round' else
                     ('—' if row[k] is None else f'{row[k]:.6f}') for k in columns) + ' |')
    lines += ['', 'StyleCap 完整结果见 summary.csv / summary.json。复用来源见 summary.json 的 result_path。',
              '本入口仅评测 Captioner，不生成 TTS 音频、不调用付费 API。']
    legacy.write_json(args.output_root / 'summary.json', {'entries': entries})
    (args.output_root / 'summary.md').write_text('\n'.join(lines) + '\n')
    with (args.output_root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    return entries


class CaptionRunner(legacy.Runner):
    def __init__(self, args):
        super().__init__(args.output_root / 'captioner')
        self.args = args

    def prepare_command(self, command):
        command = list(command)
        if '--batch-size' in command:
            command[command.index('--batch-size') + 1] = str(self.args.batch_size)
        gpu_scripts = {legacy.ET / 'run_inference.py', legacy.ET / 'evaluate.py',
                       legacy.PSC / 'run_content_scheme_a.py', legacy.SC / 'run_midasheng.py'}
        if Path(command[1]) in gpu_scripts:
            command = [command[0], str(GUARD), '--gpu-memory-gib', str(self.args.gpu_memory_gib), *command[1:]]
        return command

    def stage(self, record, candidates, size):
        tasks = queue.Queue()
        for item in pending_tasks(self.args, record, candidates, size):
            tasks.put(item)
        failures = []
        def worker(gpu):
            while not self.stop.is_set():
                try:
                    c, suite = tasks.get_nowait()
                except queue.Empty:
                    return
                name = f'{suite}_{c.candidate_id}'
                out = legacy.existing.output_dir(self.output, suite, size, c.candidate_id)
                logpath = self.output / 'logs' / (name + '.log')
                logpath.parent.mkdir(parents=True, exist_ok=True)
                env = legacy.environment(gpu, self.output, name)
                Path(env['AAC_METRICS_TMP_PATH']).mkdir(parents=True, exist_ok=True)
                start = time.monotonic()
                print(f'[START] GPU={gpu} {name}; log={logpath}', flush=True)
                try:
                    with logpath.open('a') as log:
                        for phase in (['smoke', 'full'] if size == 'full' else ['smoke']):
                            if legacy.task_status(self.output, suite, phase, c)[0]:
                                continue
                            out = legacy.existing.output_dir(self.output, suite, phase, c.candidate_id)
                            out.mkdir(parents=True, exist_ok=True)
                            for command in legacy.commands(c, suite, phase, out):
                                cmd = self.prepare_command([gpu if v == 'GPU' else v for v in command])
                                log.write('COMMAND ' + json.dumps(cmd) + '\n'); log.flush()
                                self.run_command(cmd, env, log)
                            ok, detail = legacy.task_status(self.output, suite, phase, c)
                            if not ok:
                                raise RuntimeError(detail)
                    legacy.write_json(out / 'launcher_timing.json', {'seconds_this_invocation': time.monotonic()-start})
                    print(f'[DONE] {name}', flush=True)
                    with self.lock:
                        summarize(self.args, record, candidates)
                except Exception as exc:
                    with self.lock:
                        failures.append({'task': name, 'error': str(exc), 'log': str(logpath)})
                    print(f'[FAILED] {name}: {exc}', flush=True)
        pool = ThreadPoolExecutor(max_workers=len(self.args.gpus))
        try:
            futures = [pool.submit(worker, gpu) for gpu in self.args.gpus]
            for future in futures:
                future.result()
        except BaseException:
            self.cancel()
            raise
        finally:
            pool.shutdown(wait=True)
        legacy.write_json(self.args.output_root / (size + '_execution.json'), {'failures': failures})
        if failures:
            raise RuntimeError('Some Captioner tasks failed; repeat the command to resume')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'smoke', 'run', 'status'])
    p.add_argument('--train-run', type=Path, default=TRAIN)
    p.add_argument('--output-root', type=Path, default=OUTPUT)
    p.add_argument('--rounds', type=round_numbers, default=list(range(10)), help='Zero-based; default 0 through 9')
    p.add_argument('--gpus', type=gpu_numbers, default=gpu_numbers('0,1,2,3,4,5,6,7'))
    p.add_argument('--gpu-memory-gib', type=float, default=26)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--reuse-root', type=Path, action='append', dest='reuse_roots')
    args = p.parse_args()
    if args.batch_size < 1 or args.gpu_memory_gib <= 0:
        p.error('Batch size and GPU memory budget must be positive')
    args.train_run = args.train_run.resolve(); args.output_root = args.output_root.resolve()
    args.reuse_roots = [v.resolve() for v in (args.reuse_roots if args.reuse_roots is not None else REUSE)]
    os.umask(0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.launcher.lock').open('a') as lock:
        # Status also takes the lock because it writes a summary.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidates = discover(args.train_run, args.rounds)
        record = inventory(args, candidates)
        legacy.write_json(args.output_root / 'inventory.json', record)
        entries = summarize(args, record, candidates)
        report = {'rounds': [c.round + 1 for c in candidates], 'total_tasks': len(entries),
                  'complete_tasks': sum(e['complete'] for e in entries), 'reused_tasks': len(record['reused']),
                  'pending_tasks': len(pending_tasks(args, record, candidates, 'full')),
                  'gpus': args.gpus, 'batch_size': args.batch_size, 'gpu_memory_gib': args.gpu_memory_gib,
                  'tts_evaluation': False, 'api_requests': 0, 'reuse_rejections': record['reuse_rejections']}
        if args.mode == 'check':
            report.update(legacy.cpu_check(candidates))
            legacy.write_json(args.output_root / 'cpu_preflight.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.mode in ('check', 'status'):
            return
        runner = CaptionRunner(args)
        def terminate(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            runner.stage(record, candidates, 'smoke' if args.mode == 'smoke' else 'full')
        finally:
            runner.cancel()
            summarize(args, record, candidates)
        print(f'[REPORT] {args.output_root / "summary.md"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped; completed results preserved. Repeat the same command to resume.', flush=True)
        raise SystemExit(130)
