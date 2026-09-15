"""Previous-round-only whole-group TTS inference placement. No training scheduling."""
from __future__ import annotations
import json
import math
from pathlib import Path
from .io import read_jsonl, sha256_file


def attach_previous_costs(rows, run_dir, round_index):
    """Freeze provenance and estimates into the hashed stage input.

    Missing prior history or new IDs falls back for the entire stage. Corrupt
    complete history raises rather than silently selecting a different schedule.
    """
    if round_index <= 0:
        return [{**r, 'rollout_cost_status': 'first_round'} for r in rows]
    prior = Path(run_dir) / f'round_{round_index-1:03d}'
    path = prior/'collections'/f'round_{round_index-1:03d}_caption_tts_rollout.output.jsonl'
    if not (prior/'commit.json').is_file() or not path.is_file():
        return [{**r, 'rollout_cost_status': 'missing_previous_history'} for r in rows]
    commit = json.loads((prior/'commit.json').read_text())
    if commit.get('round') != round_index-1 or not commit.get('same_round_start'):
        raise ValueError('Invalid previous-round commit for rollout length estimates')
    return attach_costs(rows, list(read_jsonl(path)), {'round': round_index-1,
        'path': str(path.resolve()), 'sha256': sha256_file(path),
        'commit_sha256': sha256_file(prior/'commit.json')})


def attach_costs(rows, history, provenance):
    costs = {}
    for group in history:
        key = group['id']
        if key in costs:
            raise ValueError(f'Duplicate historical group: {key}')
        candidates = group.get('candidates', [])
        frames = [len(c['codec_codes']) for c in candidates]
        if not frames or min(frames) <= 0:
            raise ValueError(f'Empty historical trajectory: {key}')
        costs[key] = (sum(frames), len(frames))
    if any(r['id'] not in costs or costs[r['id']][1] != r['group_size'] for r in rows):
        return [{**r, 'rollout_cost_status': 'incomplete_previous_history'} for r in rows]
    return [{**r, 'rollout_cost_status': 'previous_round', 'rollout_estimated_frames': costs[r['id']][0],
             'rollout_cost_source': provenance} for r in rows]


def assignment_plan(rows, world_size, mode):
    if world_size < 1 or mode not in ('round_robin', 'previous_round_lpt'):
        raise ValueError('Invalid rollout placement configuration')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('TTS rollout requires one input row per unique group')
    ready = bool(rows) and all(r.get('rollout_cost_status') == 'previous_round' for r in rows)
    costs = [r.get('rollout_estimated_frames') for r in rows]
    if ready and any(not isinstance(c, (int, float)) or isinstance(c, bool) or not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError('Invalid previous-round frame estimate')
    owners = [i % world_size for i in range(len(rows))]
    algorithm = 'round_robin'
    rr_loads = [sum(costs[i] for i, owner in enumerate(owners) if owner == rank)
                for rank in range(world_size)] if ready else None
    if mode == 'previous_round_lpt' and ready:
        loads, counts = [0]*world_size, [0]*world_size
        candidate = [0]*len(rows)
        for i in sorted(range(len(rows)), key=lambda i: (-costs[i], ids[i])):
            rank = min(range(world_size), key=lambda rank: (loads[rank], counts[rank], rank))
            candidate[i] = rank
            loads[rank] += costs[i]
            counts[rank] += 1
        # A heuristic can regress on some inputs. Preserve RR if its estimated
        # bottleneck is already as good or better.
        if max(loads) < max(rr_loads):
            owners, algorithm = candidate, 'previous_round_lpt'
        else:
            algorithm = 'round_robin_no_estimated_gain'
    loads = [sum(costs[i] for i, owner in enumerate(owners) if owner == rank)
             for rank in range(world_size)] if ready else None
    return owners, {'requested': mode, 'algorithm': algorithm, 'groups': len(rows),
        'estimated_frames_per_rank': loads, 'round_robin_estimated_frames_per_rank': rr_loads,
        'fallback_reasons': sorted(set(r.get('rollout_cost_status', 'no_estimate') for r in rows)) if not ready else [],
        'owners': owners, 'group_ids': ids}
