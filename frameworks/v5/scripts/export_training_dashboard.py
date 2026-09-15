#!/usr/bin/env python3
"""Read-only JSONL -> TensorBoard bridge. No imports from training environments."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

PHASES = {'caption_after_grpo': 'Captioner_GRPO', 'caption_final': 'Captioner_SFT',
          'tts_after_grpo': 'TTS_GRPO', 'tts_final': 'TTS_SFT'}

# Short, numbered aliases: TensorBoard groups by the first slash and sorts tags.
# Keep original tags and values for existing filters/events and detailed analysis.
OVERVIEW_TAGS = {}
for _model, _group in [('Captioner', '01_Captioner'), ('TTS', '02_TTS')]:
    for _source, _title in [
        ('reward/semantic_reward_mean', '01_语义奖励'),
        ('reward/semantic_valid_fraction', '02_有效候选比例'),
        ('SFT/mean_loss', '03_监督损失'),
        ('cycle_SFT/sft_selected', '04_循环配对数'),
        ('GRPO/mean_loss', '05_强化学习损失'),
        ('GRPO/epoch_0_mean_kl', '06_更新偏离度KL'),
    ]:
        OVERVIEW_TAGS[f'rounds/{_model}_{_source}'] = f'{_group}/{_title}'

OVERVIEW_TAGS.update({
    'rounds/Captioner_reward/format_reward_mean': '01_Captioner/02_平均格式奖励',
    'rounds/Captioner_reward/combined_reward_mean': '01_Captioner/09_加权总奖励',
    'rounds/Captioner_reward/partial_input_valid_mean': '01_Captioner/10_可用于合成的候选比例',
    'rounds/Captioner_reward/trajectory_valid_mean': '01_Captioner/11_轨迹校验通过比例',
    'rounds/Captioner_reward/raw_schema_valid_mean': '01_Captioner/12_原始格式完整比例',
    'rounds/Captioner_reward/semantic_reward_mean': '01_Captioner/01_平均语义奖励',
    'rounds/Captioner_reward/schema_active_reward_mean': '01_Captioner/02_平均格式奖励',
    'rounds/Captioner_reward/semantic_valid_fraction': '01_Captioner/03_有效候选比例',
    'rounds/Captioner_reward/grpo_semantic_fraction': '01_Captioner/04_训练组成/语义组占比',
    'rounds/Captioner_reward/grpo_schema_fraction': '01_Captioner/04_训练组成/格式组占比',
    'rounds/Captioner_reward/grpo_skipped_fraction': '01_Captioner/04_训练组成/跳过组占比',
    'rounds/Captioner_SFT/mean_loss': '01_Captioner/05_监督损失',
    'rounds/Captioner_cycle_SFT/sft_selected': '01_Captioner/06_循环配对数',
    'rounds/Captioner_GRPO/mean_loss': '01_Captioner/07_强化学习损失',
    'rounds/Captioner_GRPO/epoch_0_mean_kl': '01_Captioner/08_更新偏离度KL',
})


def overview_layout():
    """Native Custom Scalars supports multiple tags on one chart; Time Series doesn't."""
    import re
    from tensorboard.plugins.custom_scalar import layout_pb2
    layout = layout_pb2.Layout()
    for group in ['01_Captioner', '02_TTS']:
        category = layout.category.add(title=group)
        charts = {}
        for tag in sorted(t for t in OVERVIEW_TAGS.values() if t.startswith(group + '/')):
            title = tag.split('/')[1]
            if title not in charts:
                charts[title] = category.chart.add(title=title)
            charts[title].multiline.tag.append('^' + re.escape(tag) + '$')
    return layout


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None  # A running writer may not have completed this artifact yet.


def read_rows(path):
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # Never consume a partly appended final record.
            rows.append(json.loads(line))  # Malformed complete lines are errors.
    return rows


def reward_scalars(folder, cache=None):
    """Candidate-weighted rollout rewards; never mix semantic and schema scores.

    Consume only completed reward stages, independently of round completion.
    Cache small aggregates, not trajectories, between live exporter polls.
    """
    cache = {} if cache is None else cache
    n = int(folder.name.split('_')[1])
    for pool, model in [('audio', 'Captioner'), ('caption', 'TTS')]:
        base = folder / 'rewards' / f'{folder.name}_{pool}_rewards'
        stage_path = Path(str(base) + '.stage.json')
        path = Path(str(base) + '.output.jsonl')
        stage = read_json(stage_path)
        if not stage or stage.get('status') not in ('complete', 'reused') or not path.is_file():
            continue
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stage_path.stat().st_mtime_ns)
        if path not in cache or cache[path][0] != signature:
            values, ids = {}, set()
            total = valid = groups = 0
            modes = {'semantic': 0, 'schema': 0, 'skipped': 0}

            def add(key, value):
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    values.setdefault(key, []).append(float(value))

            with path.open() as stream:
                for line in stream:
                    if not line.endswith('\n'):
                        raise ValueError(f'Incomplete record in completed reward stage: {path}')
                    row = json.loads(line)
                    groups += 1
                    active = [c for c in row['candidates'] if not c.get('skip_update', True)]
                    active_modes = {c.get('grpo_mode') for c in active}
                    if len(active) < 2:
                        modes['skipped'] += 1
                    elif active_modes == {'attribute_and_format'}:
                        modes['semantic' if any(c.get('semantic_valid') for c in active) else 'schema'] += 1
                    elif active_modes == {'dual_semantic'}:
                        modes['semantic'] += 1
                    elif active_modes == {'schema_curriculum'}:
                        modes['schema'] += 1
                    else:
                        raise ValueError(f'Unexpected mixed/unknown GRPO modes in {path}: {active_modes}')
                    for candidate in row['candidates']:
                        cid = candidate['candidate_id']
                        if cid in ids:
                            raise ValueError(f'Duplicate reward candidate {cid} in {path}')
                        ids.add(cid)
                        total += 1
                        if candidate.get('grpo_mode') == 'attribute_and_format':
                            add('format_reward', candidate.get('format_score'))
                            add('combined_reward', candidate.get('reward'))
                            add('partial_input_valid', float(bool(candidate.get('semantic_input_valid'))))
                            add('trajectory_valid', float(bool(candidate.get('trajectory_valid'))))
                            add('raw_schema_valid', float(bool(candidate.get('raw_schema_valid'))))
                            add('valid_field_count', len(candidate.get('valid_fields', [])))
                            for field, value in candidate.get('attribute_reconstruction', {}).get('fields', {}).items():
                                add('attribute_match/' + field, value)
                        if candidate.get('semantic_valid') is True:
                            valid += 1
                            add('semantic_reward', candidate.get('semantic_reward'))
                            for source, prefix in [('reward_components_raw', 'raw'),
                                                   ('reward_components_normalized', 'normalized')]:
                                for key, value in candidate.get(source, {}).items():
                                    add(f'{prefix}/{key}', value)
                        if candidate.get('grpo_mode') == 'schema_curriculum' and not candidate.get('skip_update', True):
                            add('schema_active_reward', candidate.get('reward'))
            after = path.stat()
            if signature != (after.st_size, after.st_mtime_ns, stage_path.stat().st_mtime_ns):
                raise ValueError(f'Completed reward artifact changed during read: {path}')
            stats = {'groups': groups, 'candidates': total, 'semantic_valid_candidates': valid,
                     'semantic_reward_missing_or_nonfinite': valid - len(values.get('semantic_reward', []))}
            if total:
                stats['semantic_valid_fraction'] = valid / total
            for mode, count in modes.items():
                stats[f'grpo_{mode}_groups'] = count
                if groups:
                    stats[f'grpo_{mode}_fraction'] = count / groups
            for key, samples in values.items():
                stats[f'{key}_mean'] = math.fsum(samples) / len(samples)
                stats[f'{key}_count'] = len(samples)
            cache[path] = (signature, stats)
        for key, value in cache[path][1].items():
            yield f'rounds/{model}_reward/{key}', n, value


def scalars(run, reward_cache=None):
    """Export original metrics plus numbered, readable overview aliases."""
    for tag, step, value in raw_scalars(run, reward_cache):
        yield tag, step, value
        if tag in OVERVIEW_TAGS:
            yield OVERVIEW_TAGS[tag], step, value


def raw_scalars(run, reward_cache=None):
    """Yield (tag, step, value); round and optimizer-step axes remain separate."""
    def numeric(prefix, step, obj, keys=None):
        for key in keys or obj:
            value = obj.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                yield f'{prefix}/{key}', step, float(value)
    for folder in sorted(run.glob('round_[0-9][0-9][0-9]')):
        n = int(folder.name.split('_')[1])
        yield from reward_scalars(folder, reward_cache)
        for name, seconds in (read_json(folder / 'collections/audio_labeling_metrics.json') or {}).items():
            yield f'rounds/labeling_minutes/{name.removesuffix("_seconds")}', n, seconds / 60
        summary = read_json(folder / 'summary.json')
        if summary:
            for pool, grpo, sft in [('audio_only', 'Captioner', 'TTS'), ('caption_only', 'TTS', 'Captioner')]:
                data = summary.get(pool, {})
                yield from numeric(f'rounds/{grpo}_GRPO', n, data, [
                    'groups', 'semantic_grpo_groups', 'schema_curriculum_groups',
                    'semantic_nonzero_advantage_groups', 'structurally_valid_candidates'])
                yield from numeric(f'rounds/{sft}_cycle_SFT', n, data, [
                    'sft_selected', 'sft_eligible_groups', 'sft_selected_despite_gate', 'sft_rejected_groups'])
                gate = summary.get('sft_thresholds', {}).get('loops', {}).get(pool, {})
                yield from numeric(f'rounds/{sft}_gate_diagnostic', n, gate)
                if 'threshold' in gate:
                    yield f'rounds/{sft}_gate_diagnostic/available', n, float(gate['threshold'] is not None)
            yield from numeric('rounds/anchors', n, summary, ['caption_anchor_rows', 'tts_anchor_rows'])
            audio = summary.get('audio_only', {})
            yield from numeric('rounds/Captioner_generated_unknown_count', n, audio.get('generated_unknown_counts', {}))
        for checkpoint, phase in PHASES.items():
            root = folder / 'checkpoints' / checkpoint
            # Captioner already reduces across ranks. TTS writes reduced loss at root.
            metrics = root / 'training_metrics' / 'rank_000' / 'metrics.jsonl'
            if not metrics.exists():
                metrics = root / 'training_metrics' / 'metrics.jsonl'
            if metrics.exists():
                seen = {}
                for row in read_rows(metrics):
                    key = (row.get('epoch', 0), row['step'])
                    if key in seen:
                        if seen[key] != row:
                            raise ValueError(f'Conflicting repeated step in {metrics}: {key}')
                        continue
                    seen[key] = row
                    epoch = row.get('epoch', 0)
                    yield from numeric(f'steps/r{n:03d}/{phase}/epoch_{epoch}', int(row['step']), row,
                                       ['loss', 'grad_norm'])
                    yield from numeric(f'steps/r{n:03d}/{phase}/epoch_{epoch}/rank0_local', int(row['step']), row,
                                       ['main_nll', 'sub_nll'])
            meta = read_json(root / 'dual_isl_train_training.json')
            if meta:
                yield from numeric(f'rounds/{phase}', n, meta, ['mean_loss', 'max_grad_norm', 'steps', 'optimizer_steps'])
                for i, kl in enumerate(meta.get('epoch_mean_kl', [])):
                    if isinstance(kl, (float, int)) and math.isfinite(kl):
                        yield f'rounds/{phase}/epoch_{i}_mean_kl', n, float(kl)
        for path in sorted(folder.rglob('*.stage.json')):
            stage = read_json(path)
            if stage and stage.get('status') in ('complete', 'reused') and 'elapsed_seconds' in stage:
                name = path.name.removeprefix(folder.name + '_').removesuffix('.stage.json')
                yield f'rounds/stage_minutes/{name}', n, stage['elapsed_seconds'] / 60
                for rank in stage.get('worker_metrics', {}).get('per_rank', []):
                    if 'gpu_peak_memory_bytes' in rank:
                        yield f'rounds/stage_peak_GiB/{name}/rank_{rank["rank"]}', n, rank['gpu_peak_memory_bytes'] / 2**30


def status(run):
    rounds = sorted(run.glob('round_[0-9][0-9][0-9]'))
    result = {'source': str(run), 'checked_at': time.time(), 'exists': run.exists(), 'latest_round': None}
    if rounds:
        latest = rounds[-1]
        result['latest_round'] = latest.name
        result['committed'] = (latest / 'commit.json').is_file()
        result['tts_rank_last_event'] = {}
        for p in sorted(latest.glob('checkpoints/*/training_progress/rank_*.jsonl')):
            # Read only the last 64 KiB; discard the first potentially truncated line.
            with p.open('rb') as stream:
                stream.seek(0, 2)
                start = max(0, stream.tell() - 65536)
                stream.seek(start)
                lines = stream.read().splitlines(keepends=True)
            for line in reversed(lines[1:] if start else lines):
                if line.endswith(b'\n'):
                    result['tts_rank_last_event'][str(p.relative_to(latest))] = json.loads(line)
                    break
    return result


def export(run, output, watch=0):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    import struct
    def f32(value):
        return struct.unpack('f', struct.pack('f', value))[0]
    run, output = run.resolve(), output.resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError('Dashboard output must be separate from the source run')
    output.mkdir(parents=True, exist_ok=True)
    # One exporter per output. Lock released automatically on exit/crash.
    import fcntl
    lock = (output / '.export.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = output / 'source.json'
    expected = {'source': str(run), 'schema': 1}
    if identity.exists() and read_json(identity) != expected:
        raise ValueError('Output belongs to a different source/schema; use a new directory')
    identity.write_text(json.dumps(expected, indent=2) + '\n')
    prior = EventAccumulator(str(output), size_guidance={'scalars': 0}).Reload()
    seen = {(tag, item.step): item.value for tag in prior.Tags()['scalars'] for item in prior.Scalars(tag)}
    writer = EventFileWriter(str(output))
    signature = None
    reward_cache = {}
    try:
        from tensorboard.plugins.custom_scalar import metadata as layout_metadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.types_pb2 import DT_STRING
        config_tag = layout_metadata.CONFIG_SUMMARY_TAG
        layout_bytes = overview_layout().SerializeToString()
        previous = prior.Tensors(config_tag) if config_tag in prior.Tags()['tensors'] else []
        if not previous or previous[-1].tensor_proto.string_val[0] != layout_bytes:
            writer.add_event(Event(wall_time=time.time(), step=0, summary=Summary(value=[
                Summary.Value(tag=config_tag, metadata=layout_metadata.create_summary_metadata(),
                              tensor=TensorProto(dtype=DT_STRING, string_val=[layout_bytes]))])))
            writer.flush()
        while True:
            # Do not parse unchanged historical JSON files on every poll.
            current = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                       for p in run.glob('round_*/*/*') if p.name.endswith('.stage.json')]
            for pattern in ['round_*/summary.json', 'round_*/checkpoints/*/dual_isl_train_training.json',
                            'round_*/checkpoints/*/training_metrics/**/metrics.jsonl',
                            'round_*/rewards/*_rewards.output.jsonl']:
                current += [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in run.glob(pattern)]
            added = 0
            if current != signature:
                for tag, step, value in scalars(run, reward_cache):
                    key, value = (tag, step), f32(value)
                    if key in seen:
                        if seen[key] != value:
                            raise ValueError(f'Source rewrote {key}; export to a NEW output directory')
                        continue
                    writer.add_event(Event(wall_time=time.time(), step=step,
                                           summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)])))
                    seen[key] = value
                    added += 1
                writer.flush()
                signature = current
                print(f'{run.name}: added={added}, total={len(seen)}, output={output}', flush=True)
            snapshot = status(run)
            temp = output / 'status.json.tmp'
            temp.write_text(json.dumps(snapshot, indent=2) + '\n')
            temp.replace(output / 'status.json')
            if not watch:
                break
            time.sleep(watch)
    finally:
        writer.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--watch', type=float, default=0, metavar='SECONDS', help='0 exports once; >=2 follows a future/live run')
    args = parser.parse_args()
    if args.watch < 0 or 0 < args.watch < 2:
        parser.error('--watch must be 0 or >=2')
    if not args.run_dir.is_dir() and not args.watch:
        parser.error('Source does not exist (use --watch to wait for a future run)')
    try:
        export(args.run_dir, args.output_dir, args.watch)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
