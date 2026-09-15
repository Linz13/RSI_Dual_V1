from types import SimpleNamespace
import pytest
import torch
from dual_isl_train.reference_replay import RolloutReferenceReplay
from dual_isl_train.workers.qwen_voice_design import QwenVoiceDesignWorker
from dual_isl_train.config import load_config
from scripts.reward_v4_run_guard import run_contract

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.peft_config = {key: SimpleNamespace(to_dict=lambda: {'r': 16}) for key in ('default', 'reference')}
        self.eval()
    def set_adapter(self, name):
        pass
    def get_base_model(self):
        return SimpleNamespace(speech_tokenizer=SimpleNamespace(decode=lambda codes: ([torch.zeros(240).numpy()], 24000)))

def worker(monkeypatch, enabled=True, reference=True, exact=True):
    monkeypatch.setattr('dual_isl_train.reference_replay.audit_adapter_pair',
                        lambda model: {'exact': exact, 'checked_tensors': 2, 'max_abs_diff': 0 if exact else 1})
    obj = QwenVoiceDesignWorker.__new__(QwenVoiceDesignWorker)
    obj.policy = Policy()
    obj.cfg = {'generation': {'reference_replay_reuse': enabled}}
    obj.has_reference_adapter = reference
    obj.torch, obj.device, obj.seed = torch, torch.device('cpu'), 1
    obj.output_metrics, obj.calls = {}, []
    def compute(*args, **kwargs):
        obj.calls.append(kwargs.get('reference', False))
        return values()
    obj.incremental_trajectory_logprobs = compute
    return obj

def values():
    return torch.full((2,), -1.), torch.full((2, 15), -1.)

def test_exact_pair_validates_once_then_reuses(monkeypatch):
    obj = worker(monkeypatch)
    session = RolloutReferenceReplay(obj)
    assert session.reference({}, [], values())[1] == 'independent_validation'
    reference, mode = session.reference({}, [], values())
    assert mode == 'reused' and not reference[0].requires_grad
    assert obj.calls == [True]
    assert session.metrics['validation_max_abs_error'] == 0

@pytest.mark.parametrize('enabled,reference,exact,reason', [(False, True, True, 'disabled'), (True, False, True, 'base_reference'), (True, True, False, 'adapter_mismatch')])
def test_fallback(monkeypatch, enabled, reference, exact, reason):
    obj = worker(monkeypatch, enabled, reference, exact)
    session = RolloutReferenceReplay(obj)
    for _ in range(2):
        assert session.reference({}, [], values())[1] == 'independent_' + reason
    assert obj.calls == [True, True]

def test_config_and_eval_conditions(monkeypatch):
    obj = worker(monkeypatch)
    obj.policy.peft_config['reference'] = SimpleNamespace(to_dict=lambda: {'r': 8})
    assert RolloutReferenceReplay(obj).metrics['reason'] == 'adapter_config_mismatch'
    obj.policy.train()
    assert RolloutReferenceReplay(obj).metrics['reason'] == 'not_eval'

@pytest.mark.parametrize('change', ['weights', 'train_mode'])
def test_state_change_invalidates_reuse(monkeypatch, change):
    obj = worker(monkeypatch)
    session = RolloutReferenceReplay(obj)
    session.reference({}, [], values())
    if change == 'weights':
        with torch.no_grad(): obj.policy.weight.add_(1)
    else:
        obj.policy.train()
    assert session.reference({}, [], values())[1] == 'independent_model_state_changed'
    assert obj.calls == [True, True]

@pytest.mark.parametrize('bad', [1e-5, float('nan')])
def test_numerical_difference_falls_back(monkeypatch, bad):
    obj = worker(monkeypatch)
    session = RolloutReferenceReplay(obj)
    changed = (torch.full((2,), -1.+bad), values()[1])
    assert session.reference({}, [], changed)[1] == 'independent_validation_failed'
    assert session.reference({}, [], values())[1].startswith('independent_')
    assert session.metrics['reused'] == 0

def test_real_rollout_method_preserves_training_inputs(monkeypatch, tmp_path):
    import dual_isl_train.workers.qwen_voice_design as module
    monkeypatch.setattr(module, 'main_generation_logprobs', lambda *args: values()[0])
    monkeypatch.setattr(module, 'sub_generation_logprobs', lambda *args: values()[1])
    row = {'id': 'sample', 'request': {'language': 'English'}, 'group_size': 4, 'candidate_seeds': [1, 2, 3, 4]}
    outputs = []
    for enabled in (False, True):
        obj = worker(monkeypatch, enabled)
        obj._tokenize_request = lambda request: (torch.ones((1, 2), dtype=torch.long), None)
        obj._generate_with_main_scores = lambda **kwargs: (([torch.zeros((2, 16), dtype=torch.long)], []), None, None)
        outputs.append(obj.rollout([row], str(tmp_path/str(enabled)/'rollout.jsonl')))
        assert obj.calls.count(False) == 4
        assert obj.calls.count(True) == (1 if enabled else 4)
    for a,b in zip(outputs[0][0]['candidates'], outputs[1][0]['candidates']):
        for key in ('old_main_logprobs', 'old_sub_logprobs', 'ref_main_logprobs', 'ref_sub_logprobs', 'codec_codes', 'trajectory_valid'):
            assert a[key] == b[key]
        assert b['trajectory_valid']
    assert [c['reference_replay_mode'] for c in outputs[1][0]['candidates']] == ['independent_validation'] + ['reused']*3

def test_optional_smoke_contract(monkeypatch):
    from dual_isl_train.config import ENV_OVERRIDES, INTEGER_ENV_OVERRIDES
    for key in (*ENV_OVERRIDES, *INTEGER_ENV_OVERRIDES): monkeypatch.delenv(key, raising=False)
    cfg = load_config('configs/replay_smoke_8gpu_h100_midasheng_reward_v4.yaml')
    assert cfg['training']['rounds'] == 2 and cfg['data']['max_records_per_role'] == 8
    assert run_contract(cfg, 'replay-smoke', 2)['mode'] == 'replay-smoke'
    cfg['tts']['generation']['reference_replay_reuse'] = False
    with pytest.raises(ValueError, match='reference_replay_reuse'):
        run_contract(cfg, 'replay-smoke', 2)


def test_optional_verifier_requires_actual_reuse(monkeypatch, tmp_path):
    import json
    from scripts import verify_tts_reference_reuse as verifier
    monkeypatch.setattr(verifier, 'checkpoint_record', lambda p: {'path': p, 'sha256': 'test'})
    for n in (0, 1):
        folder = tmp_path/f'round_{n:03d}'
        (folder/'collections').mkdir(parents=True)
        (folder/'commit.json').write_text(json.dumps({'round': n, 'same_round_start': True,
            'captioner': {'path': 'c', 'sha256': 'test'}, 'tts': {'path': 't', 'sha256': 'test'}}))
        groups = []
        for rank in range(8):
            modes = ['independent_base_reference']*4 if n == 0 else ['independent_validation']+['reused']*3
            groups.append({'candidates': [{'trajectory_valid': True, 'reference_replay_mode': mode} for mode in modes]})
        p = folder/'collections'/f'round_{n:03d}_caption_tts_rollout.output.jsonl'
        p.write_text('\n'.join(json.dumps(g) for g in groups))
        ranks = [{'rank': rank, 'requested': True, 'candidates': 4, 'reason': 'base_reference' if n==0 else 'equal_adapters',
                  'reused': 0 if n==0 else 3, 'independent': 4 if n==0 else 1,
                  'validation_candidates': 0 if n==0 else 1, 'validation_max_abs_error': 0,
                  'adapter_audit': {'exact': True, 'checked_tensors': 462}} for rank in range(8)]
        Path = type(tmp_path)
        Path(str(p)+'.metrics.json').write_text(json.dumps({'reference_replay_per_rank': ranks}))
    assert verifier.verify(tmp_path)['ok']
    ranks[0]['reused'] = 0
    Path(str(p)+'.metrics.json').write_text(json.dumps({'reference_replay_per_rank': ranks}))
    with pytest.raises(ValueError, match='exercised'):
        verifier.verify(tmp_path)
