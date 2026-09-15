#!/usr/bin/env python3
"""Independent, read-only V3 TTS replay equivalence/timing probe; never trains."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True  # Importing verified V3 modules must not write into that source tree.

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT.parent / 'DualISL_Train_RewardV3/runs/midasheng_7b_reward_v3_10rounds_20260905_run01'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fresh_output(output, run):
    output, run = output.resolve(), run.resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError('Output must be separate from the historical run')
    output.mkdir(parents=True, exist_ok=False)
    return output


def verify_source(run, *, metadata_only=False):
    project = run.parent.parent
    identity_path = run / 'reward_v3_run_identity.json'
    identity = json.loads(identity_path.read_text())
    if identity['framework_version'] != 'DualISL-Train-RewardV3-0.3':
        raise ValueError('This historical probe requires an original V3 source run')
    for rel, expected in identity['code_sha256'].items():
        if digest(project / rel) != expected:
            raise ValueError(f'Historical source code changed: {rel}')
    # Import only the verified V3 implementation: never relabel V3 checkpoints as V4.
    if any(k == 'dual_isl_train' or k.startswith('dual_isl_train.') for k in sys.modules):
        raise RuntimeError('Run this probe as a fresh standalone process')
    sys.path.insert(0, str(project))
    from dual_isl_train.checkpoints import checkpoint_record
    previous = json.loads((run / 'round_008/commit.json').read_text())
    current = json.loads((run / 'round_009/commit.json').read_text())
    expected_path = (run / 'round_008/checkpoints/tts_final').resolve()
    record = previous['tts']
    if Path(record['path']).resolve() != expected_path:
        raise ValueError('Unexpected committed checkpoint path')
    if not metadata_only:
        record = checkpoint_record(expected_path)
    if record != previous['tts'] or record != current['input_tts']:
        raise ValueError('r8 committed TTS checkpoint does not match r9 input')
    return identity, record, project


def select_groups(path, longest=False):
    with path.open() as stream:
        groups = [json.loads(line) for line in stream if line.strip()]
    groups.sort(key=lambda g: (sum(len(c['codec_codes']) for c in g['candidates']), g['id']))
    selected = []
    for label, fraction in [('short', .1), ('medium', .5), ('long', .9)] + ([('longest', 1.)] if longest else []):
        group = groups[round((len(groups)-1) * fraction)]
        if len(group['candidates']) != 4 or not all(c.get('trajectory_valid') for c in group['candidates']):
            raise ValueError(f'Invalid four-candidate group: {group["id"]}')
        selected.append({'label': label, 'group': group,
                         'frames': sum(len(c['codec_codes']) for c in group['candidates'])})
    return selected


def replay(worker, candidate, reuse, *, identity_exact):
    if reuse and (not identity_exact or worker.policy.training or not worker.has_reference_adapter):
        raise ValueError('Reuse requires an exactly equal adapter pair in eval mode')
    policy = worker.incremental_trajectory_logprobs(candidate['request'], candidate['codec_codes'], reference=False, grad=False)
    reference = policy if reuse else worker.incremental_trajectory_logprobs(
        candidate['request'], candidate['codec_codes'], reference=True, grad=False)
    return policy, reference


def max_error(torch, left, right):
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError('Nonfinite or mismatched replay tensors')
    return (left.float() - right.float()).abs().max().item()


def verify_values(torch, worker, candidate, baseline, optimized, atol):
    errors = {}
    for kind, before, after in zip(('main', 'sub'), baseline[0], optimized[0]):
        errors[f'{kind}_policy_repeat'] = max_error(torch, before, after)
    for index, kind in enumerate(('main', 'sub')):
        before, after = baseline[1][index], optimized[1][index]
        errors[f'{kind}_independent_reference_vs_reuse'] = max_error(torch, before, after)
        old = torch.tensor(candidate[f'old_{kind}_logprobs'], device=before.device, dtype=before.dtype)
        errors[f'{kind}_historical_behavior_vs_policy'] = max_error(torch, old, baseline[0][index])
        recorded_ref = torch.tensor(candidate[f'ref_{kind}_logprobs'], device=before.device, dtype=before.dtype)
        errors[f'{kind}_historical_reference_vs_reference'] = max_error(torch, recorded_ref, before)
        # Check actual clipped-loss/KL formula and dL/d(current logprobs), including
        # nonzero KL / clipped ratios. This is NOT a model backward/optimizer test.
        for advantage in (-1., 1.):
            results = []
            for ref in (before, after):
                delta = torch.linspace(-.3, .3, old.numel(), device=old.device).reshape(old.shape)
                current = (baseline[0][index].detach().float() + delta).requires_grad_()
                policy_loss, kl = worker._clipped_loss(torch, current, old, ref, advantage,
                                                       worker.cfg['training']['clip_range'])
                loss = policy_loss + worker.cfg['training']['kl_beta'] * kl
                gradient, = torch.autograd.grad(loss, current)
                results.append((loss.detach(), gradient))
            for field, a, b in zip(('loss', 'logprob_gradient'), results[0], results[1]):
                errors[f'{kind}_adv{advantage}_{field}'] = max_error(torch, a, b)
    if max(errors.values()) > atol:
        raise ValueError(f'Equivalence/behavior replay tolerance {atol} exceeded: {errors}')
    return errors


def gpu_run(args, identity, checkpoint, selections, output, results):
    import torch
    from dual_isl_train.adapters import audit_adapter_pair
    from dual_isl_train.workers.qwen_voice_design import QwenVoiceDesignWorker
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly ONE allocated GPU with CUDA_VISIBLE_DEVICES; do not use torchrun')
    name = torch.cuda.get_device_name(0)
    if 'H100' not in name:
        raise ValueError(f'This planned H100 probe requires H100; found {name}')
    config = json.loads(json.dumps(identity['config']))  # Exact recorded settings, no env overrides.
    config['distributed'].update(enabled=False, world_size=1)
    config['tts']['device'] = 'cuda:0'
    config['tts']['codec_cache_dir'] = str(output / 'unused_codec_cache')
    results['probe_config'] = config
    results['environment'] = {'gpu': name, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'python': sys.version, 'executable': sys.executable}
    torch.manual_seed(config['run']['seed'])
    started = time.perf_counter()
    worker = QwenVoiceDesignWorker(config, checkpoint=checkpoint['path'], target_checkpoint=checkpoint['path'])
    worker.policy.eval()
    torch.cuda.synchronize()
    results['model_load_seconds'] = time.perf_counter() - started
    audit = audit_adapter_pair(worker.policy)
    results['adapter_pair'] = audit
    if not audit['exact'] or not audit.get('checked_tensors', 0):
        raise ValueError(f'Adapter pair is not exactly equal: {audit}')
    save(output / 'results.json', results)
    for selection in selections:
        candidates = selection['group']['candidates']
        record = {'label': selection['label'], 'group_id': selection['group']['id'],
                  'frames': selection['frames'], 'correctness': [], 'timings': []}
        results['groups'].append(record)
        print(f'Checking {record["label"]}: {record["group_id"]}, {record["frames"]} frames', flush=True)
        # These independent replays also warm up both adapter contexts; excluded from timing.
        for candidate in candidates:
            baseline = replay(worker, candidate, False, identity_exact=True)
            optimized = replay(worker, candidate, True, identity_exact=True)
            errors = verify_values(torch, worker, candidate, baseline, optimized, args.atol)
            record['correctness'].append({'candidate_id': candidate['candidate_id'], 'errors': errors})
            del baseline, optimized
        for repeat in range(args.repeats):
            for reuse in ([False, True] if repeat % 2 == 0 else [True, False]):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                start_event.record()
                for candidate in candidates:
                    value = replay(worker, candidate, reuse, identity_exact=True)
                    del value
                end_event.record()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                record['timings'].append({'repeat': repeat, 'mode': 'reuse' if reuse else 'baseline',
                    'replay_calls': len(candidates) * (1 if reuse else 2), 'wall_seconds': elapsed,
                    'cuda_elapsed_ms': start_event.elapsed_time(end_event),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
                print(f'  repeat={repeat} reuse={reuse} wall={elapsed:.3f}s', flush=True)
                save(output / 'results.json', results)
        medians = {mode: statistics.median(v['wall_seconds'] for v in record['timings'] if v['mode'] == mode)
                   for mode in ('baseline', 'reuse')}
        record['median_wall_seconds'] = medians
        record['replay_wall_reduction_fraction'] = 1 - medians['reuse'] / medians['baseline']
        save(output / 'results.json', results)
    results['adapter_pair_after'] = audit_adapter_pair(worker.policy)
    if not results['adapter_pair_after']['exact']:
        raise ValueError('Adapter identity changed during probe')
    results['status'] = 'passed'
    lines = ['# TTS reference replay probe', '',
             'PASS: fixed-trajectory numerical checks passed. No optimizer update or full-round timing was performed.', '',
             '| Group | Frames | Baseline median (s) | Reuse median (s) | Replay time reduction |',
             '|---|---:|---:|---:|---:|']
    for group in results['groups']:
        med = group['median_wall_seconds']
        lines.append(f'| {group["label"]} | {group["frames"]} | {med["baseline"]:.3f} | {med["reuse"]:.3f} | {group["replay_wall_reduction_fraction"]:.1%} |')
    lines += ['', 'This measures the two replay calls only; it excludes generation, reward scoring, training and multi-GPU scheduling.',
              'A passed probe does not enable reuse in V4. Production integration and smoke validation remain separate.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--include-longest', action='store_true')
    parser.add_argument('--atol', type=float, default=5e-4)
    args = parser.parse_args()
    if args.repeats < 2 or not 0 <= args.atol <= 5e-4:
        parser.error('repeats >=2 and 0 <= atol <=5e-4 required')
    run = args.source_run.resolve()
    identity, checkpoint, project = verify_source(run, metadata_only=args.action == 'prepare')
    source = run / 'round_009/collections/round_009_caption_tts_rollout.output.jsonl'
    selections = select_groups(source, args.include_longest)
    output = fresh_output(args.output_dir, run)
    results = {'status': 'prepared', 'source_run': str(run), 'source_project': str(project),
               'checkpoint': checkpoint, 'rollout_sha256': digest(source), 'groups': [],
               'probe_sha256': digest(Path(__file__)), 'atol': args.atol,
               'checkpoint_bytes_verified': args.action == 'run',
               'scope': 'TTS fixed-trajectory rollout replay only; no generation, optimizer update, DDP, or full-round speed claim'}
    save(output / 'inputs.json', selections)
    save(output / 'results.json', results)
    print(json.dumps({'output': str(output), 'groups': [{k: s[k] for k in ('label', 'frames')} | {'id': s['group']['id']} for s in selections]}, indent=2), flush=True)
    if args.action == 'run':
        results['status'] = 'running'
        save(output / 'results.json', results)
        try:
            gpu_run(args, identity, checkpoint, selections, output, results)
        except Exception:
            results.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            save(output / 'results.json', results)
    print(f'{results["status"]}: {output / "results.json"}', flush=True)

if __name__ == '__main__':
    main()
