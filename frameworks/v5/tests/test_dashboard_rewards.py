"""CPU-only reward statistics: populations, stage completeness and restart safety."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('dashboard', Path(__file__).parents[1] / 'scripts/export_training_dashboard.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def fixture(run, pool='audio', complete=True):
    root = run / 'round_000/rewards'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'round_000_{pool}_rewards.output.jsonl'
    candidates = [
        {'candidate_id': 'a', 'semantic_valid': True, 'semantic_reward': 2,
         'reward': 100, 'grpo_mode': 'schema_curriculum', 'skip_update': False,
         'reward_components_raw': {'reconstruction': -4},
         'reward_components_normalized': {'anchor_penalty': .5}},
        {'candidate_id': 'b', 'semantic_valid': True, 'semantic_reward': 4,
         'reward': 4, 'skip_update': True, 'reward_components_raw': {'reconstruction': -2}},
        {'candidate_id': 'c', 'semantic_valid': False, 'semantic_reward': None,
         'reward': 0, 'grpo_mode': 'schema_curriculum', 'skip_update': False,
         'reward_components_raw': {'reconstruction': float('nan')}},
    ]
    path.write_text(json.dumps({'candidates': candidates}) + '\n')
    stage = root / f'round_000_{pool}_rewards.stage.json'
    stage.write_text(json.dumps({'status': 'complete' if complete else 'running'}))
    return path, stage


def test_population_direction_and_cache(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path)
    fixture(tmp_path, 'caption')
    cache = {}
    points = {(t, s): v for t, s, v in dashboard.scalars(tmp_path, cache)}
    for model in ['Captioner', 'TTS']:
        prefix = f'rounds/{model}_reward/'
        assert points[prefix + 'semantic_reward_mean', 0] == 3
        assert points[prefix + 'semantic_reward_count', 0] == 2
        assert points[prefix + 'schema_active_reward_mean', 0] == 50
        assert points[prefix + 'semantic_valid_fraction', 0] == pytest.approx(2/3)
        assert points[prefix + 'raw/reconstruction_mean', 0] == -3
        assert points[prefix + 'normalized/anchor_penalty_count', 0] == 1
    original = Path.open
    def guarded(self, *args, **kwargs):
        if self.name.endswith('_rewards.output.jsonl'):
            raise AssertionError('Unchanged trajectory file reread')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    assert {(t, s): v for t, s, v in dashboard.scalars(tmp_path, cache)} == points


def test_wait_for_complete_and_reject_truncation(tmp_path):
    path, stage = fixture(tmp_path, complete=False)
    path.write_text('{"candidates":')
    assert list(dashboard.scalars(tmp_path)) == []
    stage.write_text('{"status":"complete"}')
    with pytest.raises(ValueError, match='Incomplete record'):
        list(dashboard.scalars(tmp_path))


def test_missing_semantic_reward_and_duplicates(tmp_path):
    path, _ = fixture(tmp_path)
    row = json.loads(path.read_text())
    for c in row['candidates']:
        c['semantic_reward'] = float('nan')
    path.write_text(json.dumps(row) + '\n')
    points = {t: v for t, _, v in dashboard.scalars(tmp_path)}
    assert 'rounds/Captioner_reward/semantic_reward_mean' not in points
    assert points['rounds/Captioner_reward/semantic_reward_missing_or_nonfinite'] == 2
    row['candidates'].append(row['candidates'][0])
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='Duplicate reward candidate'):
        list(dashboard.scalars(tmp_path))


def test_backfill_and_restart(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run, output = tmp_path / 'run', tmp_path / 'events'
    root = run / 'round_000'
    root.mkdir(parents=True)
    (root / 'summary.json').write_text('{"caption_only":{"sft_selected":196}}')
    dashboard.export(run, output)
    fixture(run)
    dashboard.export(run, output)
    dashboard.export(run, output)
    events = EventAccumulator(str(output), size_guidance={'scalars': 0}).Reload()
    assert len(events.Scalars('rounds/Captioner_cycle_SFT/sft_selected')) == 1
    values = events.Scalars('rounds/Captioner_reward/semantic_reward_mean')
    assert len(values) == 1 and values[0].value == 3


def test_group_proportions_include_skipped_and_zero_advantage(tmp_path):
    path, _ = fixture(tmp_path)
    rows = []
    for index, mode in enumerate(['dual_semantic', 'schema_curriculum', None]):
        rows.append({'candidates': [dict(candidate_id=f'{index}_{j}', grpo_mode=mode,
                    skip_update=mode is None, advantage=0, semantic_valid=False)
                    for j in range(4)]})
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    points = {t: v for t, _, v in dashboard.scalars(tmp_path)}
    prefix = 'rounds/Captioner_reward/'
    assert sum(points[prefix + f'grpo_{m}_fraction'] for m in ['semantic', 'schema', 'skipped']) == 1
    for mode in ['semantic', 'schema', 'skipped']:
        assert points[prefix + f'grpo_{mode}_groups'] == 1
    # Zero advantage doesn't turn a group assigned to semantic training into a skip.


def test_custom_layout_and_restart(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.plugins.custom_scalar import metadata, layout_pb2
    run, output = tmp_path / 'run', tmp_path / 'events'
    fixture(run)
    dashboard.export(run, output)
    dashboard.export(run, output)
    events = EventAccumulator(str(output), size_guidance={'tensors': 0}).Reload()
    configs = events.Tensors(metadata.CONFIG_SUMMARY_TAG)
    assert len(configs) == 1
    layout = layout_pb2.Layout.FromString(configs[0].tensor_proto.string_val[0])
    assert [c.title for c in layout.category] == ['01_Captioner', '02_TTS']
    assert [len(c.chart) for c in layout.category] == [8, 6]
    composition = layout.category[0].chart[3]
    assert composition.title == '04_训练组成' and len(composition.multiline.tag) == 3
    assert len(events.Scalars('01_Captioner/02_平均格式奖励')) == 1
