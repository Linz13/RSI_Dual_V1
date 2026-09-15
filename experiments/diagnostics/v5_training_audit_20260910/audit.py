"""Read-only training audit. No models or APIs are invoked; reports live here."""
import collections
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('/data/L202500147/Caption/DualISL_Train_RewardV5_FastResume')
RUN = ROOT.parent / 'DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01'
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from dual_isl_train.attribute_reward import reconstruction, reference_mask, score_audio_groups, field_known, SCORE_FIELDS
from dual_isl_train.partial_caption import admit
from dual_isl_train.io import load_yaml, sha256_file, stable_hash, hash_path
from dual_isl_train.render import render_qwen_request
from dual_isl_train.schema import get_path
from dual_isl_train.labeling import evaluator_resources


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def distribution(values):
    v = sorted(values)
    if not v:
        return {'n': 0}
    return {'n': len(v), 'mean': statistics.mean(v), 'min': min(v), 'max': max(v),
            'p25': v[int((len(v)-1)*.25)], 'median': statistics.median(v), 'p75': v[int((len(v)-1)*.75)]}


def main():
    cfg, state = read(RUN/'run_state.json'), None
    state, cfg = cfg, load_yaml(RUN/'resolved_config.yaml')
    files = (sorted((ROOT/'dual_isl_train').rglob('*.py')) + sorted((ROOT/'dual_isl_train').rglob('*.json'))
             + sorted((ROOT/'scripts').glob('*captioner_candidate.py')))
    implementation = stable_hash([{'path':str(p.relative_to(ROOT)), 'sha256':sha256_file(p)} for p in files])
    report = {'captured_unix':time.time(), 'run':str(RUN), 'source':str(ROOT), 'current':state['current'],
              'implementation_hash':implementation,
              'config_source_lock_matches': state['config_hash'] == stable_hash({'config':cfg, 'implementation_hash':implementation}),
              'evaluator_resources_match':cfg['labeling']['frozen_resources'] == evaluator_resources(cfg['labeling'])}
    errors = []
    def require(ok, message):
        if not ok: errors.append(message)
    require(report['config_source_lock_matches'], 'config/source lock mismatch')
    require(report['evaluator_resources_match'], 'external evaluator changed')
    rd = RUN/'round_000'
    commit = read(rd/'commit.json')
    for role in ('captioner','tts'):
        require(hash_path(commit[role]['path']) == commit[role]['sha256'], 'checkpoint hash '+role)
    report['stage_minutes'] = {p.stem: read(p).get('elapsed_seconds',0)/60 for p in rd.rglob('*.stage.json')}
    report['labeling_times'] = read(rd/'collections/audio_labeling_metrics.json')
    report['rollouts'] = {}
    for ri in (0,1):
        path=RUN/f'round_{ri:03d}/collections/round_{ri:03d}_audio_caption_rollout.output.jsonl'
        if not path.exists():continue
        groups=rows(path); cs=[c for g in groups for c in g['candidates']]
        parsed=[admit(c['raw_text']) for c in cs]
        invalid=[(c,a) for c,a in zip(cs,parsed) if not a['semantic_input_valid']]
        eligible=[(c,a) for c,a in zip(cs,parsed) if a['semantic_input_valid'] and c['trajectory_valid']]
        report['rollouts'][str(ri)]={
            'candidates':len(cs), 'raw_schema_valid':sum(a.get('raw_schema_valid',False) for a in parsed),
            'json_parseable':sum(a['json_parseable'] for a in parsed),
            'semantic_input_valid':sum(a['semantic_input_valid'] for a in parsed),
            'trajectory_valid':sum(c['trajectory_valid'] for c in cs), 'synthesis_eligible':len(eligible),
            'format':distribution([a['format_score'] for a in parsed]),
            'finish_reasons':dict(collections.Counter(c['finish_reason'] for c in cs)),
            'invalid_finish_reasons':dict(collections.Counter(c['finish_reason'] for c,a in invalid)),
            'parse_errors':dict(collections.Counter(e.split(':')[0] for c,a in invalid for e in a.get('parse_errors',[]))),
            'batch_sizes':dict(collections.Counter(c.get('generation_batch_size',1) for c in cs)),
            'fallback_reasons':dict(collections.Counter(c.get('generation_batch_fallback','none') for c in cs)),
            'eligible_known_condition_fields':distribution([len(a['condition_fields']) for c,a in eligible]),
            'eligible_no_voice_instruction':sum(render_qwen_request(a['caption'])['instruct']=='Speak naturally in a neutral voice.' for c,a in eligible),
        }
    audio=rows(rd/'rewards/round_000_audio_rewards.output.jsonl')
    caption=rows(rd/'rewards/round_000_caption_rewards.output.jsonl')
    judges={}
    for p in (RUN/cfg['labeling']['cache_dir']).glob('judge/*.json'):
        d=read(p)
        judges[stable_hash(d['pairs'])]=d['scores']
    # Reconstruct judge cache keys from stored pairs without calling the service.
    from dual_isl_train.attribute_reward import JUDGE_FIELDS
    denoms=[]; numeric=[]; best=[]; masks={}; complete=[]
    inversions=pair_count=zero_rec_groups=only_format_groups=0
    recomputed=score_audio_groups(audio,.9,.1)
    for g,e in zip(audio,recomputed):
        ref=g['reference_attributes']; masks[g['id']]=reference_mask(ref)
        denoms.append(len(masks[g['id']]))
        valid=[]
        for c,expected in zip(g['candidates'],e['candidates']):
            for key in ('reward','advantage','skip_update','sft_selected'):
                require(c[key]==expected[key], c['candidate_id']+': mismatch '+key)
            if c.get('generated_attributes') is not None:
                pred=c['generated_attributes']
                pairs=[{'id':f,'field':f,'reference':get_path(ref,f),'candidate':get_path(pred,f)} for f in JUDGE_FIELDS if field_known(ref,f) and field_known(pred,f)]
                scores=judges.get(stable_hash(pairs),{}) if pairs else {}
                value=reconstruction(ref,pred,scores)
                require(value==c['attribute_reconstruction'], c['candidate_id']+': attribute score mismatch')
                numeric.append(c['reconstruction_score']); complete.append(c)
            if c['semantic_valid']:valid.append(c)
        if valid:
            best.append(max(c['reconstruction_score'] for c in valid))
        else:
            zero_rec_groups+=1
            only_format_groups+=any(not c['skip_update'] for c in g['candidates'])
        for a,b in itertools.combinations(valid,2):
            if a['reconstruction_score']!=b['reconstruction_score']:
                pair_count+=1
                inversions+=(a['reward']-b['reward'])*(a['reconstruction_score']-b['reconstruction_score']) < 0
    report['audio_reward']={'reconstruction':distribution(numeric), 'selected':distribution(best),
        'reference_denominator':distribution(denoms),'valid_pair_count':pair_count,
        'format_reverses_reconstruction_order_pairs':inversions,'no_reconstruction_groups':zero_rec_groups,
        'format_only_update_groups':only_format_groups,
        'reference_unknown':{f:sum(f not in m for m in masks.values()) for f in SCORE_FIELDS},
        'generated_unknown':{f:sum(not field_known(c['generated_attributes'],f) for c in complete) for f in SCORE_FIELDS},
        'per_field_mean':{f:statistics.mean(v) for f in SCORE_FIELDS if (v:=[c['attribute_reconstruction']['fields'][f] for c in complete if f in c['attribute_reconstruction']['fields']])}}
    audio_index={c['candidate_id']:(g,c) for g in audio for c in g['candidates']}
    caption_index={c['candidate_id']:(g,c) for g in caption for c in g['candidates']}
    report['sft']={}
    for role in ('caption','tts'):
        data=rows(rd/f'training/round_000_{role}_sft.input.jsonl')
        report['sft'][role]={'total':len(data),'anchors':sum(r['is_anchor'] for r in data)}
        for row in data:
            require(row['target_origin']=='source_domain', 'wrong SFT origin '+row['id'])
            if role=='caption': require(row['target_schema']=='synth_v1','source format anchor retained')
            if row['is_anchor']:continue
            g,c=(audio_index if role=='tts' else caption_index)[row['id']]
            require(c['sft_selected'], 'unselected SFT record '+row['id'])
            if role=='tts':
                require(row['audio_path']==row['source_audio_path']==g['audio_path'],'wrong TTS target')
                require(row['request']==render_qwen_request(c['caption']),'TTS conditioning mismatch')
            else:
                require(row['caption']==g['source_caption'] and row['audio_path']==c['audio_path'],'wrong Captioner SFT target')
    report['updates']={}
    for phase in ('caption_grpo','caption_sft','tts_grpo','tts_sft'):
        d=read(rd/f'training/round_000_{phase}.output.jsonl.metrics.json')
        ranks=d.get('per_rank',d.get('distributed',{}).get('per_rank',[]))
        ids=[s for rank in ranks for s in rank['sample_ids']]
        require(len(ids)==len(set(ids)),phase+': duplicate real samples across ranks')
        if phase.endswith('sft'):
            expected={r['id'] for r in rows(rd/f'training/round_000_{phase}.input.jsonl')}
        else:
            groups=audio if phase.startswith('caption') else caption
            expected={g['id'] for g in groups if any(not c['skip_update'] for c in g['candidates'])}
        require(set(ids)==expected,phase+': missing or extra updated samples')
        require(all(r['parameter_changed'] and math.isfinite(r['mean_loss']) for r in ranks),phase+': no change or nonfinite')
        report['updates'][phase]={'steps':sorted({r['optimizer_steps'] for r in ranks}),
           'unique_samples':len(set(ids)), 'loss':ranks[0]['mean_loss'],
           'peak_gib':max(r['gpu_peak_memory_bytes'] for r in ranks)/2**30,
           'parameter_sync':d.get('parameter_sync_after',{}).get('ok'),
           'loaded_adapters_exact':{k:v.get('exact') for k,v in d.get('adapter_audits',{}).items()}}
    old=ROOT.parent/'DualISL_Train_RewardV5_LabelRobust'
    report['changed_core_files']=[str(p.relative_to(ROOT)) for p in files if (old/p.relative_to(ROOT)).exists() and sha256_file(p)!=sha256_file(old/p.relative_to(ROOT))]
    report['errors']=errors
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
