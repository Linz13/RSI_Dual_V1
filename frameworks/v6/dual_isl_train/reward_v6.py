"""Only final audio content/attributes plus intermediate format; no GT-caption reward."""
from copy import deepcopy
from statistics import fmean, pstdev
from .schema_v6 import ATTRS, normalize, known


class EvaluationPending(RuntimeError):
    pass


def reconstruction(reference, predicted):
    mask = [k for k in ATTRS if known(normalize(k, reference.get(k)))]
    if not mask: raise ValueError('No known reference attributes')
    scores = {k: float(normalize(k, predicted.get(k)) == normalize(k, reference[k])) for k in mask}
    return {'score': fmean(scores.values()), 'fields': scores, 'reference_fields': mask, 'denominator': len(mask)}


def score_groups(groups):
    groups = deepcopy(groups)
    for g in groups:
        for c in g['candidates']:
            status = c.get('evaluation_status')
            if status not in ('complete', 'content_failed', 'unrenderable', 'invalid_trajectory'):
                raise EvaluationPending('Incomplete candidate: ' + c['candidate_id'])
            rec = c['attribute_reconstruction']['score'] if status == 'complete' else 0.0
            if not 0 <= rec <= 1 or not 0 <= c['format_score'] <= 1:
                raise ValueError('Invalid reward range')
            c.update(reconstruction_score=rec, reward=0.9*rec+0.1*c['format_score'],
                     semantic_reward=rec, semantic_valid=status == 'complete',
                     sft_selected=False, skip_update=True, advantage=0.0)
        active = [c for c in g['candidates'] if c.get('trajectory_valid')]
        std = pstdev(c['reward'] for c in active) if active else 0
        if len(active) >= 2 and std > 1e-12:
            mean = fmean(c['reward'] for c in active)
            for c in active:
                c.update(advantage=(c['reward']-mean)/std, skip_update=False)
        # No attribute threshold. SFT selection is independent of GRPO flat-group skipping.
        eligible = [c for c in active if c['evaluation_status'] == 'complete' and c.get('semantic_input_valid')]
        if eligible:
            best = min(eligible, key=lambda c: (-c['attribute_reconstruction']['score'],
                       c['content_check']['error_rate'], int(c['candidate_id'].rsplit('::',1)[1])))
            best['sft_selected'] = True
        g['cycle_sft_selection'] = 'content_pass_attribute_top1_no_threshold'
    return groups


def summarize(groups):
    cs = [c for g in groups for c in g['candidates']]
    def avg(key):
        vals = [c[key] for c in cs if isinstance(c.get(key),(int,float))]
        return fmean(vals) if vals else None
    spans=[]; format_only=0
    for g in groups:
        active=[c for c in g['candidates'] if c.get('trajectory_valid')]
        if len(active)>=2:
            spans.append(max(c['reconstruction_score'] for c in active)-min(c['reconstruction_score'] for c in active))
            format_only+=int(spans[-1]==0 and any(not c['skip_update'] for c in active))
    return {'groups':len(groups),'candidates':len(cs),'grpo_usable_groups':sum(any(not c['skip_update'] for c in g['candidates']) for g in groups),
            'sft_selected':sum(c['sft_selected'] for c in cs),'format_only_groups':format_only,
            'content_passed':sum(c['evaluation_status']=='complete' for c in cs),
            'content_failed':sum(c['evaluation_status']=='content_failed' for c in cs),
            'content_checks':{'count':len(errs),'mean_error_rate':fmean(errs) if errs else None,
                              'max_error_rate':max(errs) if errs else None}
                if (errs:=[c['content_check']['error_rate'] for c in cs if c.get('content_check')]) else {'count':0},
            'json_parseable':sum(c['json_parseable'] for c in cs),
            'raw_schema_valid':sum(c.get('raw_schema_valid',False) for c in cs),
            'empty_acoustic_conditions':sum(not any(k in c.get('condition_fields',[]) for k in ATTRS) for c in cs),
            'caption_batch_fallback_count':sum(bool(c.get('generation_batch_fallback')) for c in cs),
            'caption_actual_batch_counts':{str(n):sum(c.get('generation_batch_size',1)==n for c in cs)
                                          for n in sorted({c.get('generation_batch_size',1) for c in cs})},
            'tts_actual_batch_counts':{str(n):sum(c.get('synthesis_batch_size')==n for c in cs)
                                      for n in sorted({c['synthesis_batch_size'] for c in cs if 'synthesis_batch_size' in c})},
            'attribute_unknown_counts':{k:sum(c['generated_attributes'].get(k)=='unknown' for c in cs
                                            if c.get('generated_attributes')) for k in ATTRS},
            'reward_mean':avg('reward'),'reconstruction_mean':avg('reconstruction_score'),'format_mean':avg('format_score'),
            'mean_group_reconstruction_span':fmean(spans) if spans else None,
            'per_attribute':{k:{'count':len(v),'mean':fmean(v) if v else None} for k in ATTRS
                              for v in [[c['attribute_reconstruction']['fields'][k] for c in cs
                                  if k in (c.get('attribute_reconstruction') or {}).get('fields',{})]]}}
