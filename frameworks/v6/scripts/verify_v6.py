"""Read-only verification: committed ancestry, actual updates, and V6-only stage contracts."""
import json
from pathlib import Path
from dual_isl_train.checkpoints import checkpoint_record
from dual_isl_train.constants import FRAMEWORK_VERSION, CHECKPOINT_METADATA
from dual_isl_train.config import validate_config
from dual_isl_train.io import load_yaml, read_jsonl


def verify(root):
    root=Path(root);cfg=load_yaml(root/'resolved_config.yaml');validate_config(cfg)
    errors=[];rounds=[];previous_c=previous_t=None
    for i,p in enumerate(sorted(root.glob('round_*/commit.json'))):
        c=json.loads(p.read_text());summary=json.loads((p.parent/'summary.json').read_text())
        if c['round']!=i:errors.append('Non-contiguous committed rounds')
        if c['input_captioner']!=checkpoint_record(previous_c) or c['input_tts']!=checkpoint_record(previous_t):errors.append('Broken ancestry')
        updates={}
        for role,action in (('captioner','grpo'),('tts','sft')):
            checkpoint=c[role]['path'];metadata=Path(checkpoint)/CHECKPOINT_METADATA
            if checkpoint_record(checkpoint)!=c[role]:errors.append(f'{p.parent.name}/{role}: checkpoint modified')
            if not metadata.is_file():errors.append(f'{role}: missing training metadata');continue
            info=json.loads(metadata.read_text());steps=info.get('steps',0)
            if info.get('framework_version')!=FRAMEWORK_VERSION:errors.append(f'{role}: wrong framework')
            if info.get('update')!=action:errors.append(f'{role}: wrong update kind')
            changed=info.get('parameter_before')!=info.get('parameter_after')
            if steps>0 and not changed:errors.append(f'{role}: optimizer steps without parameter change')
            if role=='tts' and steps>0 and not info.get('parameter_delta',{}).get('ok'):errors.append('TTS parameter delta failed')
            updates[role]={'steps':steps,'parameter_changed':changed}
        previous_c=c['captioner']['path'];previous_t=c['tts']['path']
        groups=list(read_jsonl(p.parent/'collection'/f'round_{i:03d}_audio_rewards.output.jsonl'))
        sft=list(read_jsonl(p.parent/'training'/f'round_{i:03d}_tts_sft.input.jsonl'))
        originals={g['id']:g['audio_path'] for g in groups}
        for g in groups:
            if len(g['candidates'])!=cfg['training']['group_size']:errors.append('Wrong group size')
            if sum(c['sft_selected'] for c in g['candidates'])>1:errors.append('Multiple SFT candidates')
        for row in sft:
            if row['audio_path']!=originals[row['id']] or row['source_audio_path']!=row['audio_path']:errors.append('SFT target is not original')
            if row['content_error_rate']>0.1 or row['request']['language']!='Auto':errors.append('SFT content/language contract violated')
        rounds.append({'round':i,'summary':summary,'updates':updates})
    if not rounds:errors.append('No completed round')
    state=json.loads((root/'run_state.json').read_text())
    for name in state['stages']:
        if any(x in name for x in ('caption_sft','tts_grpo','anchor','calibration','caption_only')):errors.append('Unexpected legacy stage '+name)
    if cfg['run'].get('smoke'):
        for role in ('captioner','tts'):
            if not any(r['updates'].get(role,{}).get('steps',0)>0 for r in rounds):errors.append(f'Smoke did not exercise {role} optimizer; increase smoke sample coverage')
            if state['stages'].get('smoke_reload_'+role,{}).get('status')!='complete':errors.append(f'Smoke has not reloaded {role} adapter')
    return {'ok':not errors,'errors':errors,'rounds':rounds,'gpu_checks_performed':not cfg['run'].get('inline_mock',False)}
