import json,sys,math
from pathlib import Path
from collections import Counter,defaultdict
from statistics import fmean,pstdev
sys.path.insert(0,'/data/L202500147/Caption/DualRSI_Train_V6_AudioOnly4Attr')
from dual_isl_train.schema_v6 import ATTRS,known
from dual_isl_train.content_v6 import content_check
root=Path('/data/L202500147/Caption/DualRSI_Train_V6_AudioOnly4Attr/runs/midasheng_v6_8gpu_run01')
refs=json.loads((root/'prepared/references.json').read_text())['records']
def mean(v):return fmean(v) if v else None
def corr(pairs):
 if len(pairs)<2:return None
 x,y=zip(*pairs);mx,my=mean(x),mean(y);xx=sum((t-mx)**2 for t in x);yy=sum((t-my)**2 for t in y)
 return sum((a-mx)*(b-my) for a,b in pairs)/(xx*yy)**.5 if xx*yy else None
report={'note':'Offline diagnostic only; intermediate caption agreement uses frozen judge labels, not independent human GT. Rollout of round i uses checkpoint i-1. No training or API calls.', 'rounds':[]}
examples=[]
for i in range(3):
 groups=[json.loads(l) for l in (root/f'round_{i:03}/collection/round_{i:03}_audio_rewards.output.jsonl').open()]
 cs=[c for g in groups for c in g['candidates']]
 statuses=Counter(c['evaluation_status'] for c in cs)
 stats=json.loads((root/f'round_{i:03}/summary.json').read_text());metrics=json.loads(next((root/f'round_{i:03}/training/round_{i:03}_caption_grpo.output.jsonl').open()))
 data={'round':i,'statuses':dict(statuses),'format_mean':stats['format_mean'],'reconstruction_mean':stats['reconstruction_mean'], 'format_only_groups':stats['format_only_groups'],'usable_groups':stats['grpo_usable_groups'], 'optimizer_steps':metrics['steps'],'kl':metrics['epoch_mean_kl'], 'param_sync_ok':metrics['parameter_sync']['ok']}
 attr={f:defaultdict(list) for f in ATTRS};groupcounts=Counter();content=Counter();all_caption_scores=[];top_caption_scores=[];centered_pairs=[];all_pair_counts=Counter();selected_scores=[];selected_caps=[];same_condition=[]
 for g in groups:
  ref=refs[g['id']];refattr=ref['attributes'];active=[c for c in g['candidates'] if c['trajectory_valid']];passed=[c for c in active if c['evaluation_status']=='complete'];selected=[c for c in active if c['sft_selected']]
  groupcounts['passed_'+str(min(2,len(passed)))]+=1
  groupcounts['passed_distinct_rec']+=int(len({c['reconstruction_score'] for c in passed})>=2)
  mask=[f for f in ATTRS if known(refattr[f])]
  capscores={c['candidate_id']:mean([c['caption'][f]==refattr[f] for f in mask]) for c in active}
  valid=[c for c in active if c['semantic_input_valid']]
  all_caption_scores += [capscores[c['candidate_id']] for c in valid]
  for c in selected:
   selected_scores.append(c['reconstruction_score']);selected_caps.append(capscores[c['candidate_id']])
  if valid:
   top=max(valid,key=lambda c:c['reward']);top_caption_scores.append(capscores[top['candidate_id']])
   capmean=mean([capscores[c['candidate_id']] for c in valid]);rewmean=mean([c['reward'] for c in valid])
   centered_pairs += [(capscores[c['candidate_id']]-capmean,c['reward']-rewmean) for c in valid]
  for a_idx,a in enumerate(valid):
   for b in valid[a_idx+1:]:
    dc=capscores[a['candidate_id']]-capscores[b['candidate_id']];dr=a['reward']-b['reward']
    if dc and dr:all_pair_counts['concordant' if dc*dr>0 else 'discordant']+=1
  bycondition=defaultdict(list)
  for c in valid:
   bycondition[json.dumps(c['request'],sort_keys=True)].append(c)
   inter=content_check(ref['transcript'],c['caption']['transcript']);final=c.get('content_check')
   if final:
    content['inter_pass_final_pass' if inter['passed'] and final['passed'] else 'inter_pass_final_fail' if inter['passed'] else 'inter_fail_final_pass' if final['passed'] else 'inter_fail_final_fail']+=1
  for cond,v in bycondition.items():
   if len(v)>=2:
    same_condition.append({'n':len(v),'span':max(c['reconstruction_score'] for c in v)-min(c['reconstruction_score'] for c in v),'group':g['id']})
  for f in mask:
   vals=[c['caption'][f] for c in valid]
   attr[f]['groups_any_correct'].append(refattr[f] in vals)
   attr[f]['groups_all_correct'].append(bool(vals) and all(v==refattr[f] for v in vals))
   attr[f]['groups_all_same'].append(len(set(vals))<=1)
   attr[f]['groups_can_contrast_correct_wrong'].append(refattr[f] in vals and any(v!=refattr[f] for v in vals))
   for c in valid:
    correct=c['caption'][f]==refattr[f];a=attr[f];a['caption_correct'].append(correct);a['caption_unknown'].append(not known(c['caption'][f]));a['adv_correct' if correct else 'adv_incorrect'].append(c['advantage'])
    if c['advantage']>0:a['positive_adv_caption_correct'].append(correct)
    if c['sft_selected']:a['sft_caption_correct'].append(correct)
    if c['evaluation_status']=='complete':
     generated=c['generated_attributes'][f];s=c['attribute_reconstruction']['fields'][f];a['final_audio_correct'].append(s);a['judge_unknown'].append(not known(generated));a['tts_follows_known_instruction'].extend([generated==c['caption'][f]] if known(c['caption'][f]) else [])
     if s==1:
      a['audio_correct_caption_correct'].append(correct)
      if not correct and c['advantage']>0 and len(examples)<30:examples.append({'round':i,'id':c['candidate_id'],'field':f,'reference':refattr[f],'caption':c['caption'][f],'final_audio_label':generated,'reward':c['reward'],'advantage':c['advantage'],'other_caption':c['caption'],'reference_attributes':refattr})
     if correct:a['caption_correct_but_final_wrong'].append(s==0)
 data['attrs']={f:{k:{'n':len(v),'mean':mean(v)} for k,v in a.items()} for f,a in attr.items()}
 data.update(groups=dict(groupcounts),content=dict(content),caption_agreement_mean=mean(all_caption_scores),top_reward_caption_agreement=mean(top_caption_scores),within_group_correlation=corr(centered_pairs),pairwise_order=dict(all_pair_counts),sft_rec_distribution=dict(Counter(selected_scores)),sft_caption_agreement_mean=mean(selected_caps),same_condition_sets=len(same_condition),same_condition_nonzero_span=sum(x['span']>0 for x in same_condition),same_condition_max_span=max([x['span'] for x in same_condition],default=0))
 report['rounds'].append(data)
report['examples']=examples
remote=Counter();unknownraw=Counter()
for p in (root/'label_cache/remote').glob('*.gemini.json'):
 d=json.loads(p.read_text());raw=d.get('raw_text','');attrs=d['attributes'];remote['files']+=1
 if attrs.get('pitch_level')=='unknown':
  remote['unknown']+=1;remote['unknown_with_field_error']+=int(bool(d.get('field_errors')));unknownraw[raw]+=1
report['gemini_cache_all_scopes']={'stats':dict(remote),'common_unknown_raw':unknownraw.most_common(5)}
Path('/tmp/v6_caption_reward_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for d in report['rounds']:
 print(json.dumps({k:v for k,v in d.items() if k!='attrs'},ensure_ascii=False))
 print('ATTRS',json.dumps({f:{k:v for k,v in a.items() if k in ['caption_correct','groups_can_contrast_correct_wrong','groups_any_correct','positive_adv_caption_correct','sft_caption_correct','final_audio_correct','audio_correct_caption_correct','caption_correct_but_final_wrong','tts_follows_known_instruction']} for f,a in d['attrs'].items()}))
print('GEMINI_CACHE',report['gemini_cache_all_scopes']);print('EXAMPLES',json.dumps(examples[:5],ensure_ascii=False))
