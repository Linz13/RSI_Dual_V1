"""Restore all three source pools as real audio, keeping references out of rollout jobs."""
from pathlib import Path
from .io import read_jsonl, sha256_file, stable_hash
from .schema import get_path
from .schema_v6 import known


def build_records(cfg):
    import soundfile as sf
    source=Path(cfg['source_dir'])
    original={str(r['sample_id']):r for r in read_jsonl(cfg['original_labels_path'])}
    rows=[]; excluded=[]; seen_ids=set(); seen_audio={}; counts={}; conflicts=[]
    forbidden=set()
    for manifest in cfg.get('exclude_audio_manifests',[]):
        for r in read_jsonl(manifest):
            p=Path(r['audio_path']);p=p if p.is_absolute() else Path(manifest).parent/p
            forbidden.add(sha256_file(p))
    for role in ('paired','audio_only','caption_only'):
        records=list(read_jsonl(source/(role+'.jsonl'))); counts[role]=len(records)
        for r in records:
            sid=str(r['id'])
            if sid in seen_ids: raise ValueError('Duplicate sample ID '+sid)
            seen_ids.add(sid)
            if r.get('split')!='train': raise ValueError('Non-training source '+sid)
            old=original[sid]
            if r.get('audio_path'):
                p=Path(r['audio_path']); p=p if p.is_absolute() else source/p
            else: p=Path(cfg['original_audio_dir'])/Path(old['audio_path']).name
            p=p.resolve()
            info=sf.info(str(p));digest=sha256_file(p)
            if digest in forbidden: raise ValueError('Evaluation audio overlap '+sid)
            reason=None
            if digest in seen_audio: reason='duplicate_audio:'+seen_audio[digest]
            elif not cfg['duration']['min']<=info.duration<=cfg['duration']['max']: reason='duration'
            if reason:
                excluded.append({'id':sid,'reason':reason,'duration':info.duration});continue
            seen_audio[digest]=sid
            # Legacy corrected labels call this field transcription, not transcript.
            canonical=get_path(r.get('caption',{}),'semantic_content.transcript','')
            corrected=old.get('Target_JSON_Schema',{})
            transcript=get_path(corrected,'semantic_content.transcript') or get_path(corrected,'semantic_content.transcription','')
            if known(canonical) and known(transcript) and canonical!=transcript:conflicts.append(sid)
            rows.append({'id':sid,'audio_path':str(p),'audio_sha256':digest,'duration':info.duration,
                         'original_pool':role,'reference_transcript':transcript if known(transcript) else '',
                         'transcript_origin':'existing_reference:'+str(cfg['original_labels_path']),
                         'split':'train'})
    # Smoke covers long inputs and a range of durations, not just first IDs.
    maximum=cfg.get('max_records',0)
    if maximum and len(rows)>maximum:
        ordered=sorted(rows,key=lambda r:(-r['duration'],r['id']))
        selected=[ordered[round(i*(len(ordered)-1)/max(1,maximum-1))] for i in range(maximum)]
        rows=sorted(selected,key=lambda r:r['id'])
    report={'source_records':counts,'excluded':excluded,'audio_records':len(rows),
            'duration_seconds':sum(r['duration'] for r in rows),
            'transcript_conflicts_with_old_manifest':conflicts,
            'transcript_policy':'prefer_qwen_asr_completed_original_labels; missing references use ASR',
            'records_sha256':stable_hash(rows),'external_overlap_manifests':cfg.get('exclude_audio_manifests',[])}
    return rows,report
