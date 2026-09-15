"""Frozen V6 rounds: original three Captioner suites and InstructTTSEval DSD."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time

import midasheng_v4_v5_eval as caption
from v5_caption_round_eval import BudgetRunner
import v5_tts_dsd_eval as dsd
import v5_dsd_audio_api_judge as api

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / 'DualRSI_Train_V6_AudioOnly4Attr/runs/midasheng_v6_8gpu_run01'
OUTPUT = ROOT / 'v6_round000_002_benchmarks_run01'


def rounds_arg(value):
    numbers = [int(v) for v in value.split(',')]
    if not numbers or min(numbers) < 0 or len(set(numbers)) != len(numbers):
        raise ValueError('Select unique, nonnegative round indices')
    return numbers


def discover(run, numbers):
    candidates = caption.existing.discover_reward_candidates(
        ROOT.parent, [(run, 'midasheng_v6', numbers)], True, True)
    for c in candidates:
        if c.status != 'ready':
            raise RuntimeError(f'{c.candidate_id}: {c.detail}')
        commit = dsd.read_json(run / f'round_{c.round:03d}/commit.json')
        if commit.get('framework_version') != 'DualRSI-Train-V6-AudioOnly4Attr-0.6':
            raise ValueError('Expected a committed V6 checkpoint')
        path = run / f'round_{c.round:03d}/checkpoints/tts_final'
        if commit['tts'] != {'path': str(path), 'sha256': dsd.checkpoint_hash(path)}:
            raise ValueError(f'TTS commit mismatch for round {c.round}')
    caption.validate_candidates(candidates)
    return candidates


def tts_plan(args, number, smoke=False):
    # Reuse the exact previously evaluated DSD dataset/decoding and adapter audit.
    previous = dsd.ROUND
    try:
        dsd.ROUND = args.train_run / 'round_000'
        options = argparse.Namespace(batch_size=args.tts_batch, workers=args.workers,
            gpu_memory_gib=args.gpu_memory_gib, gpus=args.gpus, round_index=number,
            mode='smoke' if smoke else 'generate')
        plan, samples, gpus, limit = dsd.make_plan(options)
    finally:
        dsd.ROUND = previous
    plan['version'] = 'v6-dsd-committed-round-v1'
    plan['training_run'] = str(args.train_run)
    plan['judge_transport'] = 'api/gemini_audio.py; same protocol as rescored base/V5'
    for p in (Path(__file__), ROOT / 'v5_tts_dsd_eval.py', ROOT / 'v5_dsd_audio_api_judge.py',
              dsd.PIPELINE / 'score.py'):
        plan['code_hashes'][str(p.relative_to(ROOT))] = dsd.sha256_file(p)
    return plan, samples, gpus, limit


def freeze(args, candidates):
    resources = caption.resource_identity()
    for p in (Path(__file__), ROOT / 'run_v6_benchmarks.sh', ROOT / 'v5_caption_round_eval.py',
              ROOT / 'gpu_budget_exec.py', ROOT / 'v5_tts_dsd_eval.py', ROOT / 'v5_dsd_audio_api_judge.py'):
        resources[str(p)] = dsd.sha256_file(p)
    record = {'version': 'v6-three-round-benchmarks-v1', 'candidates': [asdict(c) for c in candidates],
              'resources': resources, 'caption_batch': args.caption_batch, 'tts_batch': args.tts_batch,
              'gpu_shards': len(dsd.parse_gpus(args.gpus)),
              'protocol': {'emotiontalk': 'standard_public', 'paraspeechcaps': 'Scheme A attr6',
                           'stylecap': 'speaker-open MCQ', 'tts': 'InstructTTSEval DSD en1000+zh1000',
                           'judge_model': api.MODEL, 'judge_transport': str(api.API_SCRIPT)},
              'tts_plans': [tts_plan(args, c.round)[0] for c in candidates]}
    target = args.output_root / 'inventory.json'
    if target.exists() and dsd.read_json(target) != record:
        raise ValueError('Checkpoint/code/data/batch/shard settings changed; use a new output directory')
    caption.write_json(target, record)
    return record


def summarize(args, candidates):
    entries = []
    lines = ['# V6 前三轮 benchmark', '',
             '| 训练轮次 | ET SPIDEr | ET FENSE | ParaSpeechCaps | StyleCap | DSD 中文 | DSD 英文 | DSD 平均 |',
             '|---|---|---|---|---|---|---|---|']
    for c in candidates:
        results = {}
        for suite in caption.existing.BENCHMARKS:
            ok, detail = caption.task_status(args.output_root / 'captioner', suite, 'full', c)
            out = caption.existing.output_dir(args.output_root / 'captioner', suite, 'full', c.candidate_id)
            result = caption.existing.load_suite_result(suite, out) if ok else {}
            results[suite] = result
            entries.append({'round': c.round, 'suite': suite, 'complete': ok, 'detail': detail, 'result': result})
        path = args.output_root / 'tts' / f'round_{c.round:03d}' / api.TAG / 'dsd_summary.json'
        report = dsd.read_json(path) if path.exists() else {}
        entries.append({'round': c.round, 'suite': 'DSD', 'complete': report.get('complete', False), 'result': report})
        et = results['emotiontalk'].get('tasks', {}).get('overall', {}).get('metrics', {})
        values = [et.get('spider', {}).get('value'), et.get('fense', {}).get('value'),
                  results['paraspeechcaps'].get('final_score'), results['stylecap'].get('macro_average_accuracy')]
        values += [report.get('metrics', {}).get(lang, {}).get('percentage') if report.get('complete') else None
                   for lang in ('zh', 'en')]
        values += [report.get('bilingual_percentage') if report.get('complete') else None]
        lines.append('| ' + str(c.round + 1) + ' | ' + ' | '.join('—' if v is None else f'{v:.4f}' for v in values) + ' |')
    lines += ['', '缺失/未完成结果不记为零。保留原 benchmark 的完整属性和提示，不按 V6 四属性裁剪。',
              'DSD 使用与 base 重评分相同的 API 和 Gemini-2.5-Pro；每轮中英文各 1,000 条。']
    caption.write_json(args.output_root / 'summary.json', {'entries': entries})
    (args.output_root / 'summary.md').write_text('\n'.join(lines) + '\n')
    return entries


class CaptionRunner(BudgetRunner):
    def stage(self, candidates, size, gpus):
        tasks = queue.Queue()
        # Long EmotionTalk jobs begin first; all selected GPUs can claim work.
        for suite in caption.existing.BENCHMARKS:
            for c in candidates:
                tasks.put((c, suite))
        failures = []
        def worker(gpu):
            while not self.stop.is_set():
                try:
                    c, suite = tasks.get_nowait()
                except queue.Empty:
                    return
                name = f'{suite}_{c.candidate_id}'
                logpath = self.output / 'logs' / (name + '.log')
                logpath.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with logpath.open('a') as log:
                        for phase in (['smoke', 'full'] if size == 'full' else ['smoke']):
                            if caption.task_status(self.output, suite, phase, c)[0]:
                                continue
                            out = caption.existing.output_dir(self.output, suite, phase, c.candidate_id)
                            out.mkdir(parents=True, exist_ok=True)
                            env = caption.environment(gpu, self.output, name)
                            Path(env['AAC_METRICS_TMP_PATH']).mkdir(parents=True, exist_ok=True)
                            started = time.monotonic()
                            print(f'[START] GPU={gpu} {name} {phase} log={logpath}', flush=True)
                            for cmd in caption.commands(c, suite, phase, out):
                                self.run_command([gpu if v == 'GPU' else v for v in cmd], env, log)
                            ok, detail = caption.task_status(self.output, suite, phase, c)
                            if not ok:
                                raise RuntimeError(detail)
                            caption.write_json(out / 'launcher_timing.json', {'seconds_this_invocation': time.monotonic()-started})
                            print(f'[DONE] {name} {phase}', flush=True)
                except Exception as exc:
                    with self.lock:
                        failures.append({'task': name, 'error': str(exc), 'log': str(logpath)})
                    print(f'[FAILED] {name}: {exc}', flush=True)
        pool = ThreadPoolExecutor(max_workers=len(gpus))
        try:
            futures = [pool.submit(worker, gpu) for gpu in gpus]
            for future in futures:
                future.result()
        except BaseException:
            self.cancel()
            raise
        finally:
            pool.shutdown(wait=True)
        caption.write_json(self.output / (size + '_execution.json'), {'failures': failures})
        if failures:
            raise RuntimeError('Some Captioner tasks failed; rerun same command to resume')


def judge_round(out, plan, workers, dry_run=False):
    manifest = out / 'generation_manifest.json'
    data, jobs = api.judge.generation_jobs(manifest)
    if (len(jobs) != plan['expected_audios'] or any(j['task'] != 'DSD' for j in jobs)
            or data['generation_identity']['adapter'] != plan['adapter']):
        raise ValueError('DSD generation identity/count mismatch')
    destination = out / (api.TAG + ('_dry_run' if dry_run else ''))
    destination.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        key, endpoint = api.load_config()
        # Capture defaults: do not retain credentials in the plan or command line.
        api.judge.configure_judger = lambda: (key, endpoint, str(api.API_SCRIPT))
        api.judge.create_client = api.RestClient
        api.judge.call_gemini = api.call_rest
    with (destination / 'judge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = api.invoke(api.judge.main, ['--generation-manifest', manifest, '--output-dir', destination,
            '--model', api.MODEL, '--backend', 'inline', '--workers', workers, '--attempts', 5,
            '--retry-failed', '--dry-run' if dry_run else '--confirm-paid'])
        scoring = api.invoke(api.score.main, ['--generation-manifest', manifest,
            '--judge-results', destination / 'judge_results.jsonl', '--output', destination / 'summary.json',
            *(['--allow-dry-run'] if dry_run else [])])
        report = dsd.read_json(destination / 'summary.json')
        summary = {'benchmark': 'InstructTTSEval DSD', 'candidate': 'midasheng_v6', 'round': plan['round'],
            'complete': report['complete'], 'dry_run': dry_run, 'expected': report['expected'], 'scored': report['scored'],
            'checkpoint_sha256': plan['checkpoint_commit_sha256'], 'judge_model': api.MODEL,
            'metrics': {lang: report['metrics'][lang]['DSD'] for lang in ('en', 'zh')},
            'bilingual_percentage': report['bilingual_macro_average'] if report['complete'] and not dry_run else None,
            'base_bilingual_percentage': api.same_api_base_score(),
            'base_summary_path': str(api.BASE_RUN / api.TAG / 'summary.json'), 'usage': report['usage']}
        caption.write_json(destination / 'dsd_summary.json', summary)
        if result or scoring:
            raise RuntimeError('DSD judge incomplete; rerun tts-judge to reuse successes and retry failures')


def run_tts(args, numbers, smoke=False):
    for number in numbers:
        out = args.output_root / 'tts' / f'round_{number:03d}'
        if smoke:
            out /= 'smoke'
        out.mkdir(parents=True, exist_ok=True)
        plan, samples, gpus, limit = tts_plan(args, number, smoke)
        with (out / 'dsd.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            dsd.lock_plan(out, plan)
            if args.mode != 'tts-judge':
                complete = all((out / 'shards' / f'{i:02d}/generation_manifest.json').exists()
                    and dsd.read_json(out / 'shards' / f'{i:02d}/generation_manifest.json').get('complete')
                    for i in range(len(gpus)))
                if not complete:
                    dsd.generate(out, plan, gpus, limit, args.gpu_memory_gib)
            dsd.merge(out, plan, samples)
            if args.mode != 'tts-generate':
                judge_round(out, plan, args.workers, dry_run=smoke)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'smoke', 'run', 'captioner', 'tts', 'tts-generate', 'tts-judge', 'status'])
    p.add_argument('--train-run', type=Path, default=TRAIN)
    p.add_argument('--output-root', type=Path, default=OUTPUT)
    p.add_argument('--rounds', type=rounds_arg, default=[0, 1, 2])
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--gpu-memory-gib', type=float, default=26)
    p.add_argument('--caption-batch', type=int, default=4)
    p.add_argument('--tts-batch', type=int, default=8)
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    args.train_run = args.train_run.resolve(); args.output_root = args.output_root.resolve()
    gpus = dsd.parse_gpus(args.gpus)
    if min(args.caption_batch, args.tts_batch, args.workers, args.gpu_memory_gib) <= 0:
        raise ValueError('Batch sizes, workers and memory budget must be positive')
    os.umask(0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.launcher.lock').open('a') as lock:
        if args.mode != 'status':
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidates = discover(args.train_run, args.rounds)
        if args.mode == 'status':
            print(json.dumps(summarize(args, candidates), ensure_ascii=False, indent=2)); return
        freeze(args, candidates)
        if args.mode == 'check':
            report = caption.cpu_check(candidates)
            api.load_config()  # Credentials only; no request.
            tts_env = dsd.environment()
            tts_env['PATH'] = str(dsd.TTS_PY.parent) + os.pathsep + tts_env.get('PATH', '')
            subprocess.run([str(dsd.TTS_PY), '-c', 'import torch, qwen_tts, peft, soundfile'],
                           env=tts_env, check=True, timeout=120)
            report.update(rounds=args.rounds, dsd_audios_per_round=2000, dsd_total=2000*len(candidates),
                          paid_api_requests=0, gpus=gpus, gpu_memory_gib=args.gpu_memory_gib,
                          caption_batch=args.caption_batch, tts_batch=args.tts_batch, judge_model=api.MODEL)
            caption.write_json(args.output_root / 'cpu_preflight.json', report)
            print(json.dumps(report, indent=2)); return
        runner = CaptionRunner(args.output_root / 'captioner', args.caption_batch, args.gpu_memory_gib)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            if args.mode in ('run', 'captioner', 'smoke'):
                dsd.check_gpu_memory(gpus, args.gpu_memory_gib)
                runner.stage(candidates, 'smoke' if args.mode == 'smoke' else 'full', gpus)
            if args.mode in ('run', 'tts', 'tts-generate', 'tts-judge', 'smoke'):
                run_tts(args, args.rounds, smoke=args.mode == 'smoke')
        finally:
            runner.cancel()
            summarize(args, candidates)
        print(f'[REPORT] {args.output_root / "summary.md"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped; rerun the same command to resume completed work.', flush=True)
        raise SystemExit(130)
