#!/usr/bin/env python3
"""Audited, isolated parser fix for the interrupted MiDasheng V4 ten-round run."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT / '.recovery/midasheng_parserfix_v1'
RUN = PROJECT / 'runs/midasheng_7b_reward_v4_10rounds_optimized_run01'
PATCH = 'midasheng_parserfix_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_snapshot(project=PROJECT, snapshot=SNAPSHOT, run=RUN):
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    require(manifest['patch_id'] == PATCH and manifest['source_run'] == str(run), 'Wrong recovery snapshot/run')
    require(digest(run / 'reward_v4_run_identity.json') == manifest['original_identity_sha256'],
            'Original run identity changed; recovery refused')
    for field, root in [('original_code_sha256', project), ('patched_code_sha256', snapshot)]:
        for name, expected in manifest[field].items():
            require(digest(root / name) == expected, f'Code changed: {root / name}')
    actual_package = {str(p.relative_to(snapshot)) for extension in ['*.py', '*.json']
                      for p in (snapshot / 'dual_isl_train').rglob(extension)}
    expected_package = {p for p in manifest['patched_code_sha256'] if p.startswith('dual_isl_train/')}
    require(actual_package == expected_package, 'Unexpected recovery package files')
    require(digest(project / 'scripts/resume_midasheng_parserfix.py') == manifest['entry_sha256'],
            'Recovery entry changed')
    require(digest(project / 'scripts/verify_tts_ddp_sync.py') == manifest['verifier_sha256'],
            'Final verification script changed')
    return manifest


def implementation_config_hash(config, hashes):
    from dual_isl_train.io import stable_hash
    files = []
    for extension in ['.py', '.json']:
        files.extend(sorted(p for p in hashes if p.startswith('dual_isl_train/') and p.endswith(extension)))
    implementation = stable_hash([{'path': p.removeprefix('dual_isl_train/'), 'sha256': hashes[p]} for p in files])
    return stable_hash({'config': {k: v for k, v in config.items() if not k.startswith('_')},
                        'implementation_hash': implementation})


def inspect_recovery(config, manifest, run=RUN):
    from dual_isl_train.config import public_config
    identity = json.loads((run / 'reward_v4_run_identity.json').read_text())
    require(public_config(config) == identity['config'], 'Saved config differs from original run contract')
    require(Path(config['run']['output_dir']).resolve() == run, 'Wrong output directory')
    require(config['training']['rounds'] == 10 and config['training'].get('round_offset', 0) == 0,
            'Expected original ten-round schedule')
    require(config['distributed']['world_size'] == 8, 'Expected eight ranks')
    state = json.loads((run / 'run_state.json').read_text())
    old_hash = implementation_config_hash(config, manifest['original_code_sha256'])
    new_hash = implementation_config_hash(config, manifest['patched_code_sha256'])
    require(state['config_hash'] in {old_hash, new_hash}, 'Unrecognized stage implementation/config hash')
    current = state['current']
    require(3 <= current['round'] <= 9, 'Expected committed round 3 or later')
    latest = json.loads((run / 'latest.json').read_text())
    commit = json.loads((run / f'round_{current["round"]:03d}' / 'commit.json').read_text())
    require(latest == commit and latest['round'] == current['round'], 'Latest/current/commit disagree')
    for role, key in [('captioner', 'caption'), ('tts', 'tts')]:
        require(latest[role] == {'path': current[f'{key}_checkpoint'],
                                'sha256': current[f'{key}_checkpoint_sha256']}, 'Checkpoint state disagrees')
    record_path = run / 'recovery' / PATCH / 'record.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    if record:
        require(record['old_config_hash'] == old_hash and record['new_config_hash'] == new_hash,
                'Recovery record hash mismatch')
        require(record['snapshot_manifest_sha256'] == digest(SNAPSHOT / 'manifest.json'),
                'Recovery manifest changed since transition')
        require(record['backup_state_sha256'] == digest(record_path.parent / 'original_run_state.json'),
                'Recovery backup changed')
    if state['config_hash'] == old_hash:
        require(current['round'] == 3, 'Initial parser-fix transition requires round 3 boundary')
        require(state['stages'].get('round_004_audio_caption_rollout', {}).get('status') == 'failed',
                'Expected the recorded fifth-round rollout failure')
        for name, stage in state['stages'].items():
            if name.startswith('round_') and int(name.split('_')[1]) >= 4:
                require(name == 'round_004_audio_caption_rollout' and stage['status'] == 'failed',
                        'Unexpected work beyond interrupted rollout; refusing transition')
    else:
        require(record is not None, 'Patched state has no recovery audit record')
    return state, latest, old_hash, new_hash


def write_once(path, data):
    if path.exists():
        require(path.read_bytes() == data, f'Existing audit file differs: {path}')
    else:
        # Atomic file replacement prevents a crash leaving a half-written audit.
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)


def transition(run, state, old_hash, new_hash, snapshot=SNAPSHOT):
    """Under the run lock; retain original identity and exact pre-recovery state."""
    from dual_isl_train.io import atomic_json
    if state['config_hash'] == new_hash:
        return
    audit = run / 'recovery' / PATCH
    audit.mkdir(parents=True, exist_ok=True)
    backup = (run / 'run_state.json').read_bytes()
    require(json.loads(backup) == state, 'Run state changed during preflight')
    write_once(audit / 'original_run_state.json', backup)
    write_once(audit / 'original_run_identity.json', (run / 'reward_v4_run_identity.json').read_bytes())
    write_once(audit / 'snapshot_manifest.json', (snapshot / 'manifest.json').read_bytes())
    record = {'patch_id': PATCH, 'resume_from_round': 4, 'snapshot': str(snapshot),
              'old_config_hash': old_hash, 'new_config_hash': new_hash,
              'snapshot_manifest_sha256': digest(snapshot / 'manifest.json'),
              'backup_state_sha256': hashlib.sha256(backup).hexdigest(),
              'change': 'Reject malformed nested wrapper; guard validation object type; no training config change'}
    write_once(audit / 'record.json', (json.dumps(record, indent=2) + '\n').encode())
    state = dict(state, config_hash=new_hash)
    atomic_json(run / 'run_state.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'resume'])
    args = parser.parse_args()
    os.umask(0)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    manifest = verify_snapshot()
    sys.path.insert(0, str(SNAPSHOT))
    from dual_isl_train.config import ENV_OVERRIDES, INTEGER_ENV_OVERRIDES, load_config
    # Saved absolute config is authoritative for driver AND subsequent workers.
    for key in {*ENV_OVERRIDES, *INTEGER_ENV_OVERRIDES}:
        os.environ.pop(key, None)
    config = load_config(RUN / 'resolved_config.yaml')
    os.environ['PYTHONPATH'] = str(SNAPSHOT)
    os.environ['DUALISL_SHARED_WRITABLE'] = '1'
    for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'TOKENIZERS_PARALLELISM': 'false',
                       'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}.items():
        os.environ.setdefault(key, value)
    # The driver holds the same lock as the original launcher throughout training.
    with (RUN / '.launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, latest, old_hash, new_hash = inspect_recovery(config, manifest)
        from dual_isl_train.checkpoints import checkpoint_record
        for role in ['captioner', 'tts']:
            require(checkpoint_record(latest[role]['path']) == latest[role], f'{role} checkpoint hash mismatch')
        print(json.dumps({'mode': args.mode, 'run': str(RUN), 'snapshot': str(SNAPSHOT),
                          'completed_rounds': latest['round'] + 1, 'total_rounds': 10,
                          'next_round_index': latest['round'] + 1,
                          'checkpoint_hashes': 'both verified', 'config_changed': False}, indent=2), flush=True)
        if args.mode == 'check':
            return
        if latest['round'] >= 9:
            verify_completed()
            return
        import torch
        require(torch.cuda.device_count() == 8, 'Exactly eight visible GPUs required')
        require(all('H100' in torch.cuda.get_device_name(i) for i in range(8)), 'Expected eight H100 GPUs')
        transition(RUN, state, old_hash, new_hash)
        os.chdir(SNAPSHOT)
        from dual_isl_train.orchestrator import DualISLOrchestrator
        result = DualISLOrchestrator(config).train(resume_only=True)
        verify_completed()
        print(json.dumps(result, indent=2), flush=True)


def verify_completed():
    subprocess.run([sys.executable, str(PROJECT / 'scripts/verify_tts_ddp_sync.py'),
                    '--run-dir', str(RUN), '--expected-world-size', '8',
                    '--require-memory-bounded-grpo'], cwd=SNAPSHOT, check=True)


if __name__ == '__main__':
    main()
