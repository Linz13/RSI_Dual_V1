"""Optional two-round 8-GPU smoke acceptance; read-only, no model loading."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from dual_isl_train.checkpoints import checkpoint_record


def verify(run):
    reports = []
    for n in (0, 1):
        folder = run / f'round_{n:03d}'
        commit = json.loads((folder/'commit.json').read_text())
        if commit['round'] != n or not commit['same_round_start']:
            raise ValueError('Invalid round commit')
        for role in ('captioner', 'tts'):
            if checkpoint_record(commit[role]['path']) != commit[role]:
                raise ValueError(f'Checkpoint hash mismatch: r{n} {role}')
        path = folder/'collections'/f'round_{n:03d}_caption_tts_rollout.output.jsonl'
        groups = [json.loads(line) for line in path.read_text().splitlines()]
        if len(groups) != 8 or any(len(g['candidates']) != 4 for g in groups):
            raise ValueError('Expected eight groups of four TTS candidates')
        candidates = [c for g in groups for c in g['candidates']]
        if not all(c['trajectory_valid'] for c in candidates):
            raise ValueError('Invalid generated trajectory')
        meta = json.loads(Path(str(path)+'.metrics.json').read_text())
        ranks = meta['reference_replay_per_rank']
        if sorted(r['rank'] for r in ranks) != list(range(8)):
            raise ValueError('Missing or duplicated rank audit')
        for rank in ranks:
            if not rank['requested'] or rank['candidates'] != 4:
                raise ValueError('Reuse flag/candidate count mismatch')
            if n == 0:
                if rank['reason'] != 'base_reference' or rank['reused'] != 0 or rank['independent'] != 4:
                    raise ValueError('Base fallback was not covered')
            elif (rank['reused'] != 3 or rank['independent'] != 1
                  or rank['validation_candidates'] != 1 or rank['validation_max_abs_error'] != 0
                  or not rank['adapter_audit']['exact'] or rank['adapter_audit']['checked_tensors'] <= 0):
                raise ValueError('Adapter reuse was not validated and exercised on every rank')
        expected = {'independent_base_reference': 32} if n == 0 else {'independent_validation': 8, 'reused': 24}
        observed = {mode: sum(c['reference_replay_mode'] == mode for c in candidates) for mode in expected}
        if observed != expected:
            raise ValueError('Candidate reuse modes do not match rank counters')
        reports.append({'round': n, 'modes': observed, 'ranks': ranks})
    return {'ok': True, 'rounds': reports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.run_dir.resolve()), indent=2))

if __name__ == '__main__':
    main()
