"""CPU-only: reject malformed output and preserve audited resume state."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery', ROOT / 'scripts/resume_midasheng_parserfix.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_bad_wrapper_inputs_no_crash():
    code = '''
import json
from dual_isl_train.dual_space import parse_synth_caption_text_detailed, synth_validation_errors
from dual_isl_train.constants import WRAPPER_KEY as K
for malformed in ['bad', None, 3, [], True, {'nested': 'bad'}]:
    value = {K: {K: malformed}}
    assert synth_validation_errors(value)
    caption, errors, metadata = parse_synth_caption_text_detailed(json.dumps(value))
    assert caption is None and errors and not metadata['normalized_schema_valid']
for value in ['bad', None, 3, [], True]:
    caption, errors, _ = parse_synth_caption_text_detailed(json.dumps(value))
    assert caption is None and errors
'''
    subprocess.run([sys.executable, '-c', code], cwd=recovery.SNAPSHOT,
                   env=dict(os.environ, PYTHONPATH=str(recovery.SNAPSHOT), PYTHONDONTWRITEBYTECODE='1'), check=True)


def state_fixture(run):
    run.mkdir()
    config = {'run': {'output_dir': str(run)}, 'training': {'rounds': 10}, 'distributed': {'world_size': 8}}
    manifest = json.loads((recovery.SNAPSHOT / 'manifest.json').read_text())
    old = recovery.implementation_config_hash(config, manifest['original_code_sha256'])
    new = recovery.implementation_config_hash(config, manifest['patched_code_sha256'])
    current = {'round': 3, 'caption_checkpoint': 'caption', 'caption_checkpoint_sha256': 'c',
               'tts_checkpoint': 'tts', 'tts_checkpoint_sha256': 't'}
    latest = {'round': 3, 'captioner': {'path': 'caption', 'sha256': 'c'}, 'tts': {'path': 'tts', 'sha256': 't'}}
    state = {'config_hash': old, 'current': current, 'stages': {
        'round_003_tts_sft': {'status': 'complete', 'preserve': 'all metadata'},
        'round_004_audio_caption_rollout': {'status': 'failed'}}}
    (run / 'run_state.json').write_text(json.dumps(state))
    (run / 'reward_v4_run_identity.json').write_text(json.dumps({'config': config}))
    (run / 'latest.json').write_text(json.dumps(latest))
    (run / 'round_003').mkdir()
    (run / 'round_003/commit.json').write_text(json.dumps(latest))
    return config, manifest, state, old, new


def test_transition_retains_history_and_is_idempotent(tmp_path):
    run = tmp_path / 'run'
    config, manifest, state, old, new = state_fixture(run)
    identity = (run / 'reward_v4_run_identity.json').read_bytes()
    before = (run / 'run_state.json').read_bytes()
    recovery.inspect_recovery(config, manifest, run)
    recovery.transition(run, state, old, new)
    updated = json.loads((run / 'run_state.json').read_text())
    assert updated == dict(state, config_hash=new)
    assert (run / 'recovery/midasheng_parserfix_v1/original_run_state.json').read_bytes() == before
    assert (run / 'reward_v4_run_identity.json').read_bytes() == identity
    recovery.inspect_recovery(config, manifest, run)
    recovery.transition(run, updated, old, new)
    assert json.loads((run / 'run_state.json').read_text()) == updated


@pytest.mark.parametrize('change', ['hash', 'extra_stage', 'config', 'missing_audit'])
def test_refuse_unexpected_state(tmp_path, change):
    run = tmp_path / 'run'
    config, manifest, state, old, new = state_fixture(run)
    if change == 'hash': state['config_hash'] = 'unknown'
    if change == 'extra_stage': state['stages']['round_004_caption_grpo'] = {'status': 'complete'}
    if change == 'config': config['training']['rounds'] = 12
    if change == 'missing_audit': state['config_hash'] = new
    (run / 'run_state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError): recovery.inspect_recovery(config, manifest, run)


def test_partial_audit_can_resume_transition(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    config, manifest, state, old, new = state_fixture(run)
    import dual_isl_train.io as io
    original = io.atomic_json
    def interrupted(*args, **kwargs): raise RuntimeError('simulated interruption before state replacement')
    monkeypatch.setattr(io, 'atomic_json', interrupted)
    with pytest.raises(RuntimeError): recovery.transition(run, state, old, new)
    recovery.inspect_recovery(config, manifest, run)
    monkeypatch.setattr(io, 'atomic_json', original)
    recovery.transition(run, state, old, new)
    recovery.inspect_recovery(config, manifest, run)
