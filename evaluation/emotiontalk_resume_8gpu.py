#!/usr/bin/env python3
"""Resume an existing V5 EmotionTalk evaluation on eight independent GPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile

import v5_caption_round_eval as caption
from model_adapter_utils import describe_adapter, sha256_json

legacy = caption.legacy
ROOT = caption.ROOT
DEFAULT = ROOT / 'midasheng_v5_round001_caption_3gpu_run01'
TASKS = ['speaker', 'style', 'emotion', 'overall']


def key(row):
    return row['id'], row['task']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        tmp = Path(handle.name)
    try:
        tmp.chmod(0o666)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def encode_rows(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows).encode()


def parse_events(raw):
    """Recover only an interrupted final JSON line; never ignore corruption in the middle."""
    rows, dropped = [], b''
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines)-1 and not line.endswith(b'\n'):
                dropped = line
                break
            raise ValueError('Invalid complete event line; refusing to discard it')
        if not isinstance(row, dict):
            raise ValueError('Event must be a JSON object')
        rows.append(row)
    return rows, dropped


def successful(rows, run_id, allowed):
    result = {}
    for row in rows:
        if row.get('run_id') != run_id or key(row) not in allowed:
            raise ValueError('Event identity or sample key mismatch')
        if row.get('status') == 'ok':
            if not isinstance(row.get('prediction'), str) or not row['prediction'].strip():
                raise ValueError('Empty successful prediction')
            k = key(row)
            if k in result and result[k]['prediction'] != row['prediction']:
                raise ValueError('Conflicting duplicate successful predictions')
            result[k] = row
    return result


def identity(candidate, manifest, rows):
    return {'backend': 'midasheng', 'model_path': str(Path(candidate.model).resolve()),
            'max_new_tokens': 128, 'manifest_sha256': legacy.existing.sha256_file(manifest),
            'selected_prompt_hash': sha256_json([(r['task'], r['prompt']) for r in rows]),
            'tasks': TASKS, 'max_samples': None,
            'adapter': describe_adapter(candidate.adapter, candidate.model), 'attn_backend': 'sdpa'}


def partition(rows, completed, shards=8):
    buckets = [[] for _ in range(shards)]
    # Keep an audio's four task prompts together; distribute by pending request count.
    by_audio = {}
    for row in rows:
        if key(row) not in completed:
            by_audio.setdefault(row['id'], []).append(row)
    for group in by_audio.values():
        min(buckets, key=len).extend(group)
    return buckets


def validate_source(output, round_index):
    record, candidates = caption.inventory(output, round_index, 1)
    c = candidates[0]
    out = legacy.existing.output_dir(output, 'emotiontalk', 'full', c.candidate_id)
    manifest = legacy.ET / 'data/test_inference.jsonl'
    rows = legacy.rows(manifest)
    if len(rows) != 7716 or len({key(r) for r in rows}) != 7716:
        raise ValueError('Expected 7716 unique EmotionTalk requests')
    meta = legacy.existing.read_json(out / 'run_metadata.json')
    expected = identity(c, manifest, rows)
    if meta['identity'] != expected or meta['run_id'] != sha256_json(expected):
        raise ValueError('Original evaluation must use the selected checkpoint, full data, sdpa and batch=1')
    return candidates, out, rows, meta, sha256_json(record)


def prepare(work, source, rows, meta, inventory_hash):
    path = work / 'plan.json'
    code_paths = [Path(__file__), ROOT / 'run_emotiontalk_resume_8gpu.sh']
    code_hashes = {str(p): legacy.existing.sha256_file(p) for p in code_paths}
    if path.exists():
        plan = legacy.existing.read_json(path)
        if (plan['parent_run_id'] != meta['run_id'] or plan['inventory_sha256'] != inventory_hash
                or plan['code_hashes'] != code_hashes):
            raise ValueError('Resume checkpoint/code/inventory changed')
        raw = (work / 'original_events.bin').read_bytes()
        if digest(raw) != plan['source_events_sha256']:
            raise ValueError('Original event snapshot changed')
    else:
        raw = (source / 'events.jsonl').read_bytes()
        events, dropped = parse_events(raw)
        completed = successful(events, meta['run_id'], {key(r) for r in rows})
        buckets = partition(rows, completed)
        work.mkdir(parents=True, exist_ok=True)
        atomic_bytes(work / 'original_events.bin', raw)
        specs = []
        for index, batch in enumerate(buckets):
            manifest = work / 'shards' / f'{index:02d}' / 'manifest.jsonl'
            atomic_bytes(manifest, encode_rows(batch))
            specs.append({'index': index, 'manifest': str(manifest), 'count': len(batch),
                          'sha256': legacy.existing.sha256_file(manifest)})
        plan = {'version': 'emotiontalk-resume-eight-gpu-v1', 'parent_run_id': meta['run_id'],
                'inventory_sha256': inventory_hash, 'code_hashes': code_hashes,
                'source_events_sha256': digest(raw), 'preserved_successes': len(completed),
                'ignored_partial_tail_bytes': len(dropped), 'total': len(rows), 'shards': specs}
        legacy.write_json(path, plan)
    for spec in plan['shards']:
        if legacy.existing.sha256_file(Path(spec['manifest'])) != spec['sha256']:
            raise ValueError('Shard manifest changed')
    return plan


def worker_command(c, spec, gpu):
    out = Path(spec['manifest']).parent / 'output'
    return [c.python, str(legacy.ET / 'run_inference.py'), '--backend', 'midasheng', '--tasks', 'all',
            '--gpu', gpu, '--model-path', c.model, '--adapter-dir', c.adapter,
            '--manifest', spec['manifest'], '--attn-backend', 'sdpa', '--batch-size', '1',
            '--max-new-tokens', '128', '--output-dir', str(out), '--resume']


def run_workers(runner, c, plan, gpus, work):
    def worker(spec, gpu):
        if not spec['count']:
            return
        out = Path(spec['manifest']).parent / 'output'
        events = out / 'events.jsonl'
        if events.exists():
            raw = events.read_bytes()
            parsed, dropped = parse_events(raw)
            if dropped:
                atomic_bytes(out / ('interrupted_events_' + digest(raw)[:12] + '.bin'), raw)
                atomic_bytes(events, encode_rows(parsed))
        env = legacy.environment(gpu, work, f'shard{spec["index"]}')
        Path(env['AAC_METRICS_TMP_PATH']).mkdir(parents=True, exist_ok=True)
        logfile = work / 'logs' / f'shard_{spec["index"]:02d}_gpu{gpu}.log'
        logfile.parent.mkdir(exist_ok=True)
        print(f'[START] GPU={gpu} requests={spec["count"]} log={logfile}', flush=True)
        with logfile.open('a') as log:
            runner.run_command(worker_command(c, spec, gpu), env, log)
        print(f'[DONE] GPU={gpu} shard={spec["index"]}', flush=True)
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        pending = {pool.submit(worker, spec, gpu) for spec, gpu in zip(plan['shards'], gpus)}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            count = plan['preserved_successes']
            for spec in plan['shards']:
                events = Path(spec['manifest']).parent / 'output/events.jsonl'
                if events.exists():
                    parsed, _ = parse_events(events.read_bytes())
                    count += len({key(r) for r in parsed if r.get('status') == 'ok'})
            print(f'[PROGRESS] {count}/{plan["total"]}; active_shards={len(pending)}', flush=True)
    except BaseException:
        runner.cancel()
        raise
    finally:
        pool.shutdown(wait=True)


def merged_events(original, shard_events, rows, parent_run_id):
    allowed = {key(r) for r in rows}
    complete = successful(original, parent_run_id, allowed)
    for events, child_run_id, child_keys in shard_events:
        child = successful(events, child_run_id, child_keys)
        if set(child) != child_keys or set(child) & set(complete):
            raise ValueError('Missing or overlapping shard results')
        complete.update({k: {**event, 'run_id': parent_run_id} for k, event in child.items()})
    if set(complete) != allowed:
        raise ValueError('Merged results do not cover the full benchmark exactly')
    return [{k: complete[key(row)][k] for k in ('run_id', 'status', 'id', 'task', 'prediction')} for row in rows]


def merge(work, source, plan, rows, meta, c):
    raw = (work / 'original_events.bin').read_bytes()
    if digest(raw) != plan['source_events_sha256']:
        raise ValueError('Original event snapshot changed')
    original, _ = parse_events(raw)
    shards, provenance = [], []
    for spec in plan['shards']:
        if not spec['count']:
            continue
        manifest = Path(spec['manifest'])
        if legacy.existing.sha256_file(manifest) != spec['sha256']:
            raise ValueError('Shard manifest changed')
        selected = legacy.rows(manifest)
        expected = identity(c, manifest, selected)
        out = manifest.parent / 'output'
        child = legacy.existing.read_json(out / 'run_metadata.json')
        child_id = sha256_json(expected)
        if child['identity'] != expected or child['run_id'] != child_id:
            raise ValueError('Shard inference identity mismatch')
        events, dropped = parse_events((out / 'events.jsonl').read_bytes())
        if dropped:
            raise ValueError('Shard still has an incomplete event')
        shards.append((events, child_id, {key(r) for r in selected}))
        provenance.append({'shard': spec['index'], 'run_id': child_id,
                           'events_sha256': legacy.existing.sha256_file(out / 'events.jsonl')})
    merged = merged_events(original, shards, rows, meta['run_id'])
    merged_bytes = encode_rows(merged)
    current = (source / 'events.jsonl').read_bytes()
    # Also permit a crash between the two atomic replacements below.
    if digest(current) not in (plan['source_events_sha256'], digest(merged_bytes)):
        raise ValueError('Parent events changed after snapshot; original worker may still be running')
    atomic_bytes(source / 'events.jsonl', merged_bytes)
    predictions = [{k: r[k] for k in ('id', 'task', 'prediction')} for r in merged]
    atomic_bytes(source / 'predictions.jsonl', encode_rows(predictions))
    legacy.write_json(work / 'merge.json', {'complete': True, 'count': len(merged),
                      'preserved_successes': plan['preserved_successes'], 'shards': provenance,
                      'parent_events_sha256': digest(merged_bytes),
                      'note': 'Shard identities differ only in data selection; canonical parent run_id restored on merge.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'run'))
    parser.add_argument('--output-root', type=Path, default=DEFAULT)
    parser.add_argument('--round-index', type=int, default=1)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--gpu-memory-gib', type=float, default=26)
    args = parser.parse_args()
    gpus = caption.parse_gpus(args.gpus)
    if len(gpus) != 8 or args.gpu_memory_gib <= 0:
        raise ValueError('Select eight distinct physical GPUs and a positive memory budget')
    os.umask(0)
    output = args.output_root.resolve()
    candidates, source, rows, meta, inventory_hash = validate_source(output, args.round_index)
    if args.mode == 'check':
        report = legacy.cpu_check(candidates)
        events, dropped = parse_events((source / 'events.jsonl').read_bytes())
        done = successful(events, meta['run_id'], {key(r) for r in rows})
        print(json.dumps({**report, 'emotiontalk_preserved': len(done), 'pending': len(rows)-len(done),
                          'shard_counts_if_stopped_now': list(map(len, partition(rows, done))),
                          'incomplete_last_line_bytes': len(dropped), 'gpu_memory_gib': args.gpu_memory_gib}, indent=2))
        return 0
    with (output / '.launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Original Captioner launcher is still running. Ctrl+C there and wait for it to exit first.')
        work = source / 'resume_8gpu'
        if legacy.task_status(output, 'emotiontalk', 'full', candidates[0])[0]:
            print('[DONE] EmotionTalk is already fully evaluated; nothing to rerun.')
            legacy.summarize(output, candidates)
            return 0
        caption.check_gpu_memory(gpus, args.gpu_memory_gib)
        plan = prepare(work, source, rows, meta, inventory_hash)
        print(f'[RESUME] preserve={plan["preserved_successes"]} pending={len(rows)-plan["preserved_successes"]} batch=1', flush=True)
        runner = caption.BudgetRunner(output, 1, args.gpu_memory_gib)
        try:
            run_workers(runner, candidates[0], plan, gpus, work)
            merge(work, source, plan, rows, meta, candidates[0])
            env = legacy.environment(gpus[0], work, 'scoring')
            Path(env['AAC_METRICS_TMP_PATH']).mkdir(parents=True, exist_ok=True)
            (work / 'logs').mkdir(exist_ok=True)
            print('[SCORE] All 7716 predictions merged; computing standard_public metrics.', flush=True)
            with (work / 'logs/scoring.log').open('a') as log:
                runner.run_command(legacy.commands(candidates[0], 'emotiontalk', 'full', source)[1], env, log)
            ok, detail = legacy.task_status(output, 'emotiontalk', 'full', candidates[0])
            if not ok:
                raise RuntimeError('Metric validation failed: ' + detail)
        finally:
            runner.cancel()
            legacy.summarize(output, candidates)
        print(f'[REPORT] {output / "summary.md"}', flush=True)
    return 0


if __name__ == '__main__':
    def stopped(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stopped)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('[STOPPED] Original and shard results preserved; rerun the same command to resume.', flush=True)
        raise SystemExit(130)
