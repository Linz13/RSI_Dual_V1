"""Offline analysis of existing logs; no imports from torch or training packages."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
CAPTION = OUT.parents[2]
RUN = CAPTION / 'DualISL_Train_RewardV3/runs/midasheng_7b_reward_v3_10rounds_20260905_run01'
V4 = CAPTION / 'DualISL_Train_RewardV4'
SOURCES = {}


def rows(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for line in f:
            digest.update(line)
            if line.strip():
                yield json.loads(line)
    SOURCES[str(path)] = digest.hexdigest()


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def quantiles(values):
    ordered = sorted(values)
    return {k: ordered[round((len(ordered) - 1) * q)] for k, q in [('min', 0), ('p50', .5), ('p90', .9), ('max', 1)]}


def main():
    stage_records, rollout_records, rank_records, group_records = [], [], [], []
    progress_records, candidate_events = [], []
    for r in range(10):
        rd = RUN / f'round_{r:03d}'
        for path in sorted(rd.rglob('*.stage.json')):
            value = read(path)
            if 'elapsed_seconds' in value:
                assert value['status'] == 'complete'
                stage_records.append({'round': r, 'stage': path.name.replace(f'round_{r:03d}_', '').replace('.stage.json', ''),
                                      'seconds': value['elapsed_seconds']})
        for name, role, expected in [('audio_caption_rollout', 'Captioner', 184), ('caption_tts_rollout', 'TTS', 196)]:
            path = rd / f'collections/round_{r:03d}_{name}.output.jsonl'
            metadata = read(Path(str(path) + '.metrics.json'))
            assert metadata['world_size'] == 8
            owner = {}
            peaks = {}
            for rank in metadata['per_rank']:
                peaks[rank['rank']] = rank['gpu_peak_memory_bytes']
                for sample in rank['sample_ids']:
                    assert sample not in owner
                    owner[sample] = rank['rank']
            costs = [0] * 8
            counts = [0] * 8
            candidates = valid = 0
            errors, nonzero = defaultdict(float), Counter()
            local_groups = []
            for g in rows(path):
                assert len(g['candidates']) == 4
                rank = owner[g['id']]
                cost = 0
                for c in g['candidates']:
                    candidates += 1
                    valid += bool(c['trajectory_valid'])
                    if role == 'TTS':
                        frames = len(c['codec_codes'])
                        assert frames == c['codec_frames']
                        assert len(c['old_main_logprobs']) == len(c['ref_main_logprobs']) == frames
                        assert len(c['old_sub_logprobs']) == len(c['ref_sub_logprobs']) == frames
                        assert all(len(a) == len(b) == 15 for a, b in zip(c['old_sub_logprobs'], c['ref_sub_logprobs']))
                        keys = ['main_behavior_replay_max_abs_error', 'subtalker_behavior_replay_max_abs_error',
                                'main_policy_reference_max_abs_error', 'subtalker_policy_reference_max_abs_error']
                        stored_diff = max([abs(a-b) for a,b in zip(c['old_main_logprobs'],c['ref_main_logprobs'])] +
                                          [abs(a-b) for left,right in zip(c['old_sub_logprobs'],c['ref_sub_logprobs']) for a,b in zip(left,right)])
                        cost += frames
                    else:
                        tokens = len(c['sampled_token_ids'])
                        assert tokens == c['generated_token_count']
                        assert len(c['old_token_logprobs']) == len(c['ref_token_logprobs']) == tokens
                        keys = ['behavior_replay_max_abs_error', 'policy_reference_max_abs_error']
                        stored_diff = max(abs(a-b) for a,b in zip(c['old_token_logprobs'],c['ref_token_logprobs']))
                        cost += tokens
                    for key in keys:
                        value = float(c[key]);assert math.isfinite(value)
                        errors[key] = max(errors[key], value)
                        nonzero[key] += value != 0
                    errors['stored_old_reference_max_abs_diff'] = max(errors['stored_old_reference_max_abs_diff'], stored_diff)
                    nonzero['stored_old_reference_max_abs_diff'] += stored_diff != 0
                costs[rank] += cost;counts[rank] += 1
                item = {'round':r, 'role':role, 'group_id':g['id'], 'rank':rank, 'cost':cost,
                        'unit':'codec_frames' if role=='TTS' else 'sampled_tokens',
                        'source_jsonl':str(path)}
                local_groups.append(item);group_records.append(item)
            assert len(local_groups)==expected and candidates==expected*4
            # Post-hoc oracle proxy: generated lengths would not be known before generation.
            lpt = [0]*8
            for g in sorted(local_groups, key=lambda g:(-g['cost'],g['group_id'])):
                i=min(range(8),key=lambda i:(lpt[i],i));lpt[i]+=g['cost']
            pair = metadata.get('adapter_audits',{}).get('policy_reference')
            if r>0:
                assert pair and pair['exact'] and pair['max_abs_diff']==0
            record={'round':r, 'role':role,'groups':expected,'candidates':candidates,'trajectory_valid':valid,
                    'adapter_pair_audit':pair,'costs_by_rank':costs,'total_cost':sum(costs),
                    'max_over_mean_cost':max(costs)/(sum(costs)/8),
                    'oracle_lpt_costs':lpt,'oracle_proxy_reduction':1-max(lpt)/max(costs),
                    'error_max':dict(errors),'error_nonzero_candidates':dict(nonzero)}
            rollout_records.append(record)
            for rank in range(8):
                rank_records.append({'round':r,'role':role,'rank':rank,'groups':counts[rank],
                                     'cost':costs[rank],'unit':local_groups[0]['unit'],'peak_allocated_gib':peaks[rank]/1024**3})
            print(f'r{r} {role}: candidates={candidates}, costs={costs}, max/mean={record["max_over_mean_cost"]:.3f}, errors={dict(errors)}',flush=True)
        for rank in range(8):
            path=rd / f'checkpoints/tts_after_grpo/training_progress/rank_{rank:03d}.jsonl'
            starts={};ready={};step_starts={};step_frames={};steps=[]
            forward=backward=full=0.;peak=0;num_events=0
            for event in rows(path):
                num_events+=1;peak=max(peak,event.get('peak_allocated_bytes',0))
                kind=event['event'];key=(event['step'],event.get('candidate_index'))
                if kind=='step_start':
                    step_starts[event['step']]=event['timestamp']
                    step_frames[event['step']]=event['codec_frames']
                elif kind=='candidate_start':
                    starts[key]=event
                elif kind=='candidate_loss_ready':ready[key]=event
                elif kind=='candidate_complete':
                    a=starts[key];b=ready[key]
                    f=b['timestamp']-a['timestamp'];bw=event['timestamp']-b['timestamp']
                    assert f>=0 and bw>=0
                    forward+=f;backward+=bw;full+=event['elapsed_seconds']
                    candidate_events.append({'round':r,'rank':rank,'step':event['step'],
                        'candidate_index':event['candidate_index'],'candidate_id':event['candidate_id'],
                        'frames':event['codec_frames'],'padded':event['padded'],
                        'forward_loss_host_seconds':f,'backward_sync_host_seconds':bw,
                        'candidate_elapsed_seconds':event['elapsed_seconds']})
                elif kind=='step_complete':
                    steps.append({'step':event['step'],'seconds':event['elapsed_seconds'],
                                  'frames':step_frames[event['step']],'padded':event['padded']})
            assert len(starts)==len(ready)==100 and len(steps)==25
            assert all(x['seconds']>0 for x in steps)
            progress_records.append({'round':r,'rank':rank,'events':num_events,'steps':len(steps),
                'step_total_seconds':sum(s['seconds'] for s in steps),
                'candidate_total_seconds':full,'forward_loss_host_seconds':forward,
                'backward_sync_host_seconds':backward,'peak_allocated_gib':peak/1024**3,
                'steps_detail':steps})
    # Current V4 worker paths must match the V3 source for applicable performance conclusions.
    code_paths=['dual_isl_train/workers/qwen_voice_design.py','dual_isl_train/workers/qwen3_captioner.py',
                'scripts/midasheng_captioner_candidate.py','dual_isl_train/distributed.py','dual_isl_train/telemetry.py',
                'dual_isl_train/adapters.py']
    parity=[]
    for rel in code_paths:
        old=CAPTION/'DualISL_Train_RewardV3'/rel;new=V4/rel
        a=old.read_bytes();b=new.read_bytes();assert a==b,rel
        SOURCES[str(old)]=hashlib.sha256(a).hexdigest();SOURCES[str(new)]=hashlib.sha256(b).hexdigest();parity.append(rel)
    total_forward=sum(v['forward_loss_host_seconds'] for v in progress_records)
    total_backward=sum(v['backward_sync_host_seconds'] for v in progress_records)
    representatives={}
    for role in ('TTS','Captioner'):
        gs=sorted([g for g in group_records if g['role']==role and g['round']==9],key=lambda g:(g['cost'],g['group_id']))
        representatives[role]={name:gs[round((len(gs)-1)*q)] for name,q in [('short',.1),('median',.5),('long',.9),('longest',1.)]}
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),'source_run':str(RUN),
            'method':'Read existing stages/rollout scores/rank metadata/TTS GRPO heartbeat; no model imports or inference.',
            'worker_source_parity':parity,'rollouts':rollout_records,'tts_grpo_progress':progress_records,
            'representative_groups':representatives,
            'aggregate':{'stage_seconds_per_round_mean':sum(v['seconds'] for v in stage_records)/10,
                         'tts_grpo_forward_loss_host_share':total_forward/(total_forward+total_backward),
                         'tts_grpo_backward_sync_host_share':total_backward/(total_forward+total_backward),
                         'tts_grpo_step_seconds_per_round_rank_mean':statistics.mean(v['step_total_seconds'] for v in progress_records),
                         'candidate_forward_seconds_quantiles':quantiles([v['forward_loss_host_seconds'] for v in candidate_events]),
                         'candidate_backward_seconds_quantiles':quantiles([v['backward_sync_host_seconds'] for v in candidate_events])},
            'limitations':['No measured rollout substage timing, CPU/CUDA trace or continuous GPU utilization.',
                          'Post-hoc token/frame balance is a workload proxy, not measured speedup or deployable future-length knowledge.',
                          'Heartbeat host intervals include Python/logging and backward synchronization/wait; not pure CUDA kernel timing.',
                          'Historical equality is evidence, not a guarantee after batching, mode/dtype changes or new checkpoints.'],
            'sources_sha256':SOURCES}
    for name,values in [('stage_times',stage_records),('rollout_rank_loads',rank_records),('rollout_group_costs',group_records),('tts_grpo_candidate_intervals',candidate_events)]:
        with (OUT/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (OUT/'evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('AGGREGATE',json.dumps(report['aggregate']),flush=True)


if __name__=='__main__':main()
