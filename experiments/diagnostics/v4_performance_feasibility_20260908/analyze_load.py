"""Analyze already-extracted costs; scheduling simulations never launch workers."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    source=OUT/'rollout_group_costs.csv'
    with source.open() as f:
        groups=list(csv.DictReader(f))
    estimates={};simulations=[]
    for r in range(10):
        for role in ('TTS','Captioner'):
            current=[g for g in groups if int(g['round'])==r and g['role']==role]
            actual={g['group_id']:int(g['cost']) for g in current}
            if r:
                prior=estimates[role]
                assert set(prior)==set(actual)
                loads=[0]*8;measured=[0]*8;counts=[0]*8
                for gid in sorted(prior,key=lambda k:(-prior[k],k)):
                    rank=min(range(8),key=lambda i:(loads[i],i))
                    loads[rank]+=prior[gid];measured[rank]+=actual[gid];counts[rank]+=1
                baseline=[0]*8
                for g in current:baseline[int(g['rank'])]+=int(g['cost'])
                simulations.append({'round':r,'role':role,'method':'LPT whole-group placement estimated only from preceding round lengths',
                    'baseline_costs':baseline,'simulated_costs':measured,'simulated_groups_per_rank':counts,
                    'max_cost_reduction_proxy':1-max(measured)/max(baseline),
                    'max_over_mean_simulated':max(measured)/statistics.mean(measured)})
            estimates[role]=actual
    evidence=json.loads((OUT/'evidence.json').read_text())
    grpo=[]
    for r in range(10):
        ranks=[v for v in evidence['tts_grpo_progress'] if v['round']==r]
        means=[];maxima=[]
        for step in range(25):
            costs=[next(s['frames'] for s in v['steps_detail'] if s['step']==step) for v in ranks]
            means.append(statistics.mean(costs));maxima.append(max(costs))
        grpo.append({'round':r,'sum_step_max_frames':sum(maxima),'sum_step_mean_frames':sum(means),
                     'step_cost_max_mean_proxy':sum(maxima)/sum(means)})
    result={'sources_sha256':{source.name:hashlib.sha256(source.read_bytes()).hexdigest(),
                             'evidence.json':hashlib.sha256((OUT/'evidence.json').read_bytes()).hexdigest()},
            'simulations':simulations,'tts_grpo_existing_bucket_balance':grpo,
            'limitation':'Fixed historical lengths only; excludes prefix cost, hardware/I/O variation and changed outputs after scheduling. Not a timing measurement or speedup claim.'}
    (OUT/'load_simulations.json').write_text(json.dumps(result,indent=2)+'\n')
    for role in ('TTS','Captioner'):
        values=[v['max_cost_reduction_proxy'] for v in simulations if v['role']==role]
        print(role,'previous-round LPT reduction proxy min/mean/max',min(values),statistics.mean(values),max(values))
    print('r9 baseline/new',simulations[-2:])
    print('GRPO existing cost max/mean avg',statistics.mean(v['step_cost_max_mean_proxy'] for v in grpo))
    total_frames=sum(v['total_cost'] for v in evidence['rollouts'] if v['role']=='TTS')
    print('TTS frames',total_frames,'predictor calls two replays',total_frames*30)
    print('heartbeat records',sum(v['events'] for v in evidence['tts_grpo_progress']))


if __name__=='__main__':main()
