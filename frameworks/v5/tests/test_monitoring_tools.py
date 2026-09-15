"""CPU checks for independent monitoring / replay tools."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def module(name):
    path = Path(__file__).parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

monitor = module('export_training_dashboard')
probe = module('probe_reference_reuse')


def write(path, obj, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(json.dumps(r) for r in obj) if lines else json.dumps(obj)) + '\n')


def test_direction_and_missing_values(tmp_path):
    write(tmp_path/'round_009/summary.json', {'audio_only': {'sft_selected': 0, 'semantic_grpo_groups': 55},
        'caption_only': {'sft_selected': 109}, 'caption_anchor_rows': 78})
    points = {(tag, step): value for tag, step, value in monitor.scalars(tmp_path)}
    assert points['rounds/Captioner_cycle_SFT/sft_selected', 9] == 109
    assert points['rounds/TTS_cycle_SFT/sft_selected', 9] == 0
    assert points['rounds/Captioner_GRPO/semantic_grpo_groups', 9] == 55
    assert not any('mean_loss' in tag for tag, step in points)


def test_rank_dedup_and_partial_append(tmp_path):
    root = tmp_path/'round_000/checkpoints/caption_after_grpo/training_metrics'
    row = {'step': 1, 'epoch': 0, 'loss': .3, 'padded': True}
    write(root/'rank_000/metrics.jsonl', [row, row], True)
    write(root/'rank_001/metrics.jsonl', [{'step': 1, 'loss': 99}], True)
    with (root/'rank_000/metrics.jsonl').open('a') as stream:
        stream.write('{"step": 2')
    assert list(monitor.scalars(tmp_path)) == [('steps/r000/Captioner_GRPO/epoch_0/loss', 1, .3)]
    # Padded local rows still contain a global mean: do not discard their global loss.


def test_rewritten_step_is_not_silently_averaged(tmp_path):
    write(tmp_path/'round_000/checkpoints/tts_final/training_metrics/metrics.jsonl',
          [{'step': 1, 'loss': 1}, {'step': 1, 'loss': 2}], True)
    with pytest.raises(ValueError, match='Conflicting'):
        list(monitor.scalars(tmp_path))


def test_replay_calls_and_guard():
    calls = []
    def compute(*args, **kwargs):
        calls.append(kwargs)
        return object(), object()
    worker = SimpleNamespace(policy=SimpleNamespace(training=False), has_reference_adapter=True,
                             incremental_trajectory_logprobs=compute)
    candidate = {'request': {}, 'codec_codes': [[0]*16]}
    before = probe.replay(worker, candidate, False, identity_exact=True)
    assert [c['reference'] for c in calls] == [False, True]
    assert before[0] is not before[1]
    calls.clear()
    after = probe.replay(worker, candidate, True, identity_exact=True)
    assert len(calls) == 1 and after[0] is after[1]
    with pytest.raises(ValueError):
        probe.replay(worker, candidate, True, identity_exact=False)
    worker.policy.training = True
    with pytest.raises(ValueError):
        probe.replay(worker, candidate, True, identity_exact=True)


def test_output_protects_history(tmp_path):
    run = tmp_path/'run'
    run.mkdir()
    with pytest.raises(ValueError):
        probe.fresh_output(run/'probe', run)
    output = probe.fresh_output(tmp_path/'probe', run)
    with pytest.raises(FileExistsError):
        probe.fresh_output(output, run)


def test_export_restart_idempotent(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run, output = tmp_path/'run', tmp_path/'events'
    write(run/'round_000/summary.json', {'caption_only': {'sft_selected': 196}})
    monitor.export(run, output)
    monitor.export(run, output)
    events = EventAccumulator(str(output), size_guidance={'scalars': 0}).Reload()
    assert len(events.Scalars('rounds/Captioner_cycle_SFT/sft_selected')) == 1
    write(run/'round_001/summary.json', {'caption_only': {'sft_selected': 195}})
    monitor.export(run, output)
    events.Reload()
    assert len(events.Scalars('rounds/Captioner_cycle_SFT/sft_selected')) == 2


def test_tensor_equivalence_and_wrong_reference_rejected():
    torch = pytest.importorskip('torch')
    from dual_isl_train.workers.qwen_voice_design import QwenVoiceDesignWorker
    worker = SimpleNamespace(cfg={'training': {'clip_range': .2, 'kl_beta': .02}},
                             _clipped_loss=QwenVoiceDesignWorker._clipped_loss)
    main, sub = torch.full((2,), -2.), torch.full((2, 15), -3.)
    candidate = {'old_main_logprobs': main.tolist(), 'old_sub_logprobs': sub.tolist(),
                 'ref_main_logprobs': main.tolist(), 'ref_sub_logprobs': sub.tolist()}
    baseline = ((main, sub), (main.clone(), sub.clone()))
    optimized = ((main.clone(), sub.clone()), (main.clone(), sub.clone()))
    errors = probe.verify_values(torch, worker, candidate, baseline, optimized, 5e-4)
    assert max(errors.values()) == 0
    wrong = (optimized[0], (main + .1, sub))
    with pytest.raises(ValueError, match='tolerance'):
        probe.verify_values(torch, worker, candidate, baseline, wrong, 5e-4)
