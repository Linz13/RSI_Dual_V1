import copy
import json
import random
import pytest
from dual_isl_train.rollout_balance import attach_costs, attach_previous_costs, assignment_plan

def rows(costs):
    return [{'id': str(i), 'group_size': 4, 'candidate_seeds': [i*4+j for j in range(4)],
             'rollout_cost_status': 'previous_round', 'rollout_estimated_frames': c} for i,c in enumerate(costs)]

def test_deterministic_whole_groups_no_mutation():
    rng = random.Random(42)
    for n in range(80):
        source = rows([rng.randint(1,1000) for _ in range(n)])
        before = copy.deepcopy(source)
        owners, meta = assignment_plan(source, 8, 'previous_round_lpt')
        assert (owners,meta) == assignment_plan(source,8,'previous_round_lpt')
        assert source == before and len(owners)==n and all(0<=r<8 for r in owners)
        if n: assert max(meta['estimated_frames_per_rank']) <= max(meta['round_robin_estimated_frames_per_rank'])
        shards = [[(i,source[i]) for i in range(n) if owners[i]==rank] for rank in range(8)]
        assert [row for i,row in sorted(sum(shards, []))] == source

def test_no_gain_and_missing_fallback():
    assert assignment_plan(rows([10]*8),8,'previous_round_lpt')[1]['algorithm']=='round_robin_no_estimated_gain'
    source=rows([100,1,100,1]);del source[0]['rollout_cost_status']
    assert assignment_plan(source,2,'previous_round_lpt')[0]==[0,1,0,1]

def test_bad_estimates_and_duplicates_rejected():
    for c in (0,-1,float('nan'),True):
        with pytest.raises(ValueError): assignment_plan(rows([c]),8,'previous_round_lpt')
    source=rows([1,2]);source[1]['id']=source[0]['id']
    with pytest.raises(ValueError): assignment_plan(source,8,'previous_round_lpt')

def test_only_immediate_committed_history(tmp_path):
    source=[{'id':'a','group_size':4}]
    assert attach_previous_costs(source,tmp_path,0)[0]['rollout_cost_status']=='first_round'
    assert attach_previous_costs(source,tmp_path,2)[0]['rollout_cost_status']=='missing_previous_history'
    prior=tmp_path/'round_000';(prior/'collections').mkdir(parents=True)
    (prior/'commit.json').write_text(json.dumps({'round':0,'same_round_start':True}))
    path=prior/'collections/round_000_caption_tts_rollout.output.jsonl'
    path.write_text(json.dumps({'id':'a','candidates':[{'codec_codes':[[1]*16]*length} for length in (1,2,3,4)]})+'\n')
    out=attach_previous_costs(source,tmp_path,1)
    assert out[0]['rollout_estimated_frames']==10
    assert out[0]['rollout_cost_source']['round']==0
    assert 'sha256' in out[0]['rollout_cost_source']
    assert 'rollout_estimated_frames' not in source[0]
    assert attach_previous_costs(source,tmp_path,2)[0]['rollout_cost_status']=='missing_previous_history'
    assert attach_costs([{'id':'new','group_size':4}],[],{})[0]['rollout_cost_status']=='incomplete_previous_history'

def test_actual_executor_reassembles_and_measures(monkeypatch,tmp_path):
    import torch
    from dual_isl_train.distributed import DistributedContext,run_sharded_inference
    for name, fn in {'synchronize':lambda:None,'current_device':lambda:0,'get_device_name':lambda _: 'fake', 'max_memory_allocated':lambda:0}.items():
        monkeypatch.setattr(torch.cuda,name,fn)
    monkeypatch.setattr(DistributedContext,'barrier',lambda self:None)
    source=rows([100,1,100,1,100,1,100,1,100])
    owners,_=assignment_plan(source,8,'previous_round_lpt')
    for rank in list(range(1,8))+[0]:
        output,metrics=run_sharded_inference(source,tmp_path/'out.jsonl',DistributedContext(True,rank,rank,8),
                                            lambda local:[{**r,'done':True} for r in local],owners=owners)
    assert [r['id'] for r in output]==[r['id'] for r in source]
    assert sum(m['processed_rows'] for m in metrics['per_rank'])==len(source)
    assert all(m['inference_seconds']>=0 for m in metrics['per_rank'])
    with pytest.raises(ValueError,match='ownership'):
        run_sharded_inference(source,tmp_path/'bad',DistributedContext(),lambda r:r,owners=[8]*len(source))
