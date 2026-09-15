#!/usr/bin/env python3
"""Eight-H100 targeted RR vs prior-length placement check; no optimizer/critics."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time
import traceback
from probe_reference_reuse import ROOT, DEFAULT_RUN, verify_source, fresh_output, digest, save
from check_tts_rollout_reuse import production_rollout, load_module, compare


def inputs(source, count):
    history_path = source/'round_008/collections/round_008_caption_tts_rollout.output.jsonl'
    current_path = source/'round_009/collections/round_009_caption_tts_rollout.input.jsonl'
    # Fixed prefix of original request order, independent of r9 observed costs.
    with current_path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()][:count]
    if len(rows) != count:
        raise ValueError('Insufficient historical input groups')
    return rows, history_path, current_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--groups', type=int, default=16)
    p.add_argument('--reverse-order', action='store_true', help='Optional new-directory repeat with LPT first')
    args = p.parse_args()
    if args.groups < 16 or args.groups > 196:
        p.error('groups must be 16..196 (default16 for a targeted check)')
    source = args.source_run.resolve()
    rank = int(os.environ.get('RANK', '0'))
    identity, checkpoint, project = verify_source(source, metadata_only=rank != 0)
    rollout = production_rollout(project)
    balance = load_module('dual_isl_train.rollout_balance', ROOT/'dual_isl_train/rollout_balance.py')
    executor = load_module('dual_isl_train._balance_probe_distributed', ROOT/'dual_isl_train/distributed.py')
    config = copy.deepcopy(identity['config'])
    if int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('Launch with torchrun --standalone --nproc_per_node=8')
    import torch
    if torch.cuda.device_count() != 8:
        raise ValueError('Expose exactly eight allocated H100 GPUs')
    config['distributed'].update(enabled=True, world_size=8, backend='nccl')
    context = executor.DistributedContext.initialize(config)
    output = args.output_dir.resolve()
    results = {'status': 'running', 'checkpoint': checkpoint, 'groups': args.groups,
               'scope': 'two TTS rollout placements only; no optimizer or full-round speed claim', 'modes': {}}
    reserved = False
    try:
        if 'H100' not in torch.cuda.get_device_name(context.local_rank):
            raise ValueError('Expected H100 on every rank')
        os.umask(0)
        os.environ['DUALISL_SHARED_WRITABLE'] = '1'
        if context.is_main:
            fresh_output(output, source)
            reserved = True
            output.chmod(0o777)
        context.barrier()
        rows, history_path, current_path = inputs(source, args.groups)
        rows = balance.attach_previous_costs(rows, source, 9)
        plans = {mode: balance.assignment_plan(rows, 8, mode) for mode in ('round_robin', 'previous_round_lpt')}
        if plans['previous_round_lpt'][1]['algorithm'] != 'previous_round_lpt':
            raise ValueError('Selected prefix has no estimated LPT gain; increase --groups in a NEW directory')
        if context.is_main:
            results.update(input_sha256=digest(current_path), history_sha256=digest(history_path),
                code_sha256={str(f.relative_to(ROOT)): digest(f) for f in [Path(__file__),
                    ROOT/'scripts/check_tts_rollout_reuse.py', ROOT/'dual_isl_train/distributed.py',
                    ROOT/'dual_isl_train/rollout_balance.py', ROOT/'dual_isl_train/reference_replay.py',
                    ROOT/'dual_isl_train/workers/qwen_voice_design.py']})
            save(output/'results.json', results)
            save(output/'inputs.json', rows)
        config['tts']['generation']['reference_replay_reuse'] = True
        config['tts']['codec_cache_dir'] = str(output/'unused_codec_cache')
        from dual_isl_train.workers.qwen_voice_design import QwenVoiceDesignWorker
        worker = QwenVoiceDesignWorker(config, checkpoint['path'], checkpoint['path'], context)
        # One generated candidate per rank, same warmup group, excluded from timing.
        warm = {**rows[0], 'group_size': 1, 'id': f'warmup_rank{rank}'}
        rollout(worker, [warm], str(output/'warmup'/f'rank{rank}'/'output.jsonl'))
        torch.cuda.synchronize()
        context.barrier()
        outputs = {}
        modes = ['round_robin', 'previous_round_lpt']
        if args.reverse_order:
            modes.reverse()
        for mode in modes:
            owners, plan = plans[mode]
            context.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            target = output/mode/'rollout.output.jsonl'
            merged, metrics = executor.run_sharded_inference(rows, target, context,
                lambda local: rollout(worker, local, str(target)), owners=owners)
            elapsed = time.perf_counter()-started
            audits = context.gather_objects({'rank': rank, **worker.output_metrics['reference_replay']})
            if context.is_main:
                actual_frames = [0]*8
                for i, g in enumerate(merged):
                    actual_frames[owners[i]] += sum(len(c['codec_codes']) for c in g['candidates'])
                results['modes'][mode] = {'wall_seconds': elapsed, 'metrics': metrics, 'plan': plan,
                    'actual_frames_per_rank': actual_frames, 'reference_replay_per_rank': audits}
                outputs[mode] = merged
                save(output/'results.json', results)
                print(mode, 'wall_seconds=',elapsed, 'rank_seconds=',[m['inference_seconds'] for m in metrics['per_rank']],flush=True)
        if context.is_main:
            if [g['id'] for g in outputs['round_robin']] != [r['id'] for r in rows] or [g['id'] for g in outputs['previous_round_lpt']] != [r['id'] for r in rows]:
                raise ValueError('Merged order changed')
            for a,b in zip(outputs['round_robin'], outputs['previous_round_lpt']):
                compare([a], [b])
            for mode in modes:
                for audit in results['modes'][mode]['reference_replay_per_rank']:
                    if audit['candidates'] and (audit['reason'] != 'equal_adapters' or audit['validation_max_abs_error'] != 0):
                        raise ValueError('Reference reuse validation changed across ranks')
            results['comparison'] = {'exact': True, 'groups': len(rows), 'audio_bytes_equal': True}
            a,b = (results['modes'][m]['wall_seconds'] for m in ('round_robin','previous_round_lpt'))
            results.update(status='passed', observed_wall_reduction=1-b/a)
            save(output/'results.json', results)
            (output/'report.md').write_text(f'# TTS rollout placement check\n\nPASS: {len(rows)} groups match exactly across placements.\n\nRound-robin: {a:.3f}s; prior-length LPT: {b:.3f}s; observed reduction: {1-b/a:.1%}.\n\nOne measured pass per mode after warmup. This checks correctness and gives an initial timing observation, not a full-round benchmark. See results.json for every rank.\n')
            print('passed:',output/'results.json',flush=True)
        context.barrier()
    except Exception:
        if context.is_main and reserved:
            results.update(status='failed', error=traceback.format_exc())
            save(output/'results.json', results)
        raise
    finally:
        context.close()

if __name__ == '__main__':
    main()
