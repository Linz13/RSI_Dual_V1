"""One loop only: sample C_r -> synthesize T_r -> final-audio reward -> C GRPO -> T SFT."""
import json
import time
from copy import deepcopy
from pathlib import Path
from .stage_runner import StageRunner
from .checkpoints import checkpoint_record, commit_round
from .constants import SOURCE_DOMAIN_TARGET
from .content_v6 import content_check
from .data_v6 import build_records
from .io import atomic_json, read_jsonl, write_jsonl, sha256_file, stable_hash
from .schema_v6 import VERSION, ATTRS, admit, prompt, render, known
from .reward_v6 import reconstruction, score_groups, summarize
from .labeling_v6 import LabelService, resources


class DualRSIOrchestrator(StageRunner):
    def __init__(self, config):
        config=deepcopy(config)
        identity=resources(config['labeling'])
        previous=config['labeling'].get('frozen_resources')
        if previous is not None and previous!=identity:raise ValueError('Evaluator resources changed; start a new run')
        config['labeling']['frozen_resources']=identity
        super().__init__(config)

    def status(self,phase,**values):
        atomic_json(self.run_dir/'status.json',{'phase':phase,'timestamp':time.time(),**values})
        print(json.dumps({'phase':phase,**values},ensure_ascii=False),flush=True)

    def prepare_data(self):
        rows,report=build_records(self.config['data']);p=self.run_dir/'prepared/audio_manifest.jsonl'
        if p.exists() and list(read_jsonl(p))!=rows:raise ValueError('Source audio/data changed since run preparation')
        write_jsonl(p,rows);atomic_json(self.run_dir/'data_contract.json',report)
        return rows

    def references(self,rows,service):
        path=self.run_dir/'prepared/references.json'
        if path.exists():
            data=json.loads(path.read_text())
            if data['identity']!=service.identity or data['source_hash']!=stable_hash(rows):
                raise ValueError('Frozen reference identity mismatch')
            if data.get('records_hash')!=stable_hash(data['records']):
                raise ValueError('Frozen reference contents were modified')
            return data['records']
        self.status('reference_labels',audio_records=len(rows));service.scope='reference'
        missing=[r for r in rows if not known(r['reference_transcript'])]
        transcripts=service.asr(missing)
        labels=service.attributes(rows)
        records={};excluded=[]
        for r in rows:
            transcript=r['reference_transcript'] if known(r['reference_transcript']) else transcripts[r['id']]['transcript']
            attrs=labels[r['id']]
            if not known(transcript) or not any(known(attrs.get(k)) for k in ATTRS):
                excluded.append({'id':r['id'],'reason':'no_reference_content_or_attributes'});continue
            # Validate normalization before spending time on rollout.
            try:content_check(transcript,transcript)
            except ValueError:
                excluded.append({'id':r['id'],'reason':'empty_normalized_reference'});continue
            records[r['id']]={'transcript':transcript,'attributes':attrs,'audio_sha256':r['audio_sha256'],
                             'transcript_origin':r['transcript_origin'] if known(r['reference_transcript']) else 'Qwen3-ASR-1.7B'}
        if not records:raise ValueError('No usable audio references')
        atomic_json(path,{'identity':service.identity,'source_hash':stable_hash(rows),'records':records,
                         'records_hash':stable_hash(records),'excluded':excluded})
        return records

    def collect(self,index,rows,refs,c_checkpoint,t_checkpoint,service):
        directory=self.run_dir/f'round_{index:03d}'/'collection';k=self.config['training']['group_size']
        jobs=[{'id':r['id'],'audio_path':r['audio_path'],'prompt':prompt(),'caption_schema':VERSION,'group_size':k,
               'candidate_seeds':[self._seed(index,'audio_only',r['id'],i) for i in range(k)]} for r in rows]
        self.status('caption_rollout',round=index,groups=len(jobs),group_size=k)
        rollout=self._stage(name=f'round_{index:03d}_caption_rollout',role='captioner',action='rollout',rows=jobs,
                            directory=directory,checkpoint_in=c_checkpoint,target_checkpoint=c_checkpoint)
        groups=list(read_jsonl(rollout.output_path));requests=[]
        if {g['id'] for g in groups}!={r['id'] for r in rows}:raise ValueError('Caption rollout lost source IDs')
        for g in groups:
            if len(g['candidates'])!=k:raise ValueError('Caption rollout group size mismatch')
            for c in g['candidates']:
                parsed=admit(c['raw_text']);c.update(parsed)
                if not c.get('trajectory_valid'):
                    c['evaluation_status']='invalid_trajectory'
                elif not c['semantic_input_valid']:
                    c['evaluation_status']='unrenderable'
                else:
                    c['request']=render(c['caption']);c['evaluation_status']='evaluation_pending'
                    requests.append({'id':c['candidate_id'],'candidate_id':c['candidate_id'],'request':c['request'],
                                     'generation_seed':self._seed(index,'tts_reconstruction',g['id'],int(c['candidate_id'].rsplit('::',1)[1]))})
        generated=[]
        if requests:
            self.status('tts_synthesis',round=index,candidates=len(requests))
            out=self._stage(name=f'round_{index:03d}_tts_synthesis',role='tts',action='generate-audio',rows=requests,
                            directory=directory/'synthesis',checkpoint_in=t_checkpoint)
            generated=list(read_jsonl(out.output_path))
            if len(generated)!=len(requests) or {r['id'] for r in generated}!={r['id'] for r in requests}:
                raise ValueError('TTS synthesis lost candidate IDs')
            import soundfile as sf
            for r in generated:
                r['audio_sha256']=sha256_file(r['audio_path']);r['duration']=sf.info(r['audio_path']).duration
        service.scope=f'round_{index:03d}'
        self.status('asr_content_check',round=index,audios=len(generated))
        asr=service.asr(generated);by_id={r['id']:r for r in generated};passed=[]
        for g in groups:
            for c in g['candidates']:
                cid=c['candidate_id']
                if cid not in by_id:continue
                c['reconstructed_audio_path']=by_id[cid]['audio_path']
                c['synthesis_batch_size']=by_id[cid].get('synthesis_batch_size',1)
                c['asr_transcript']=asr[cid]['transcript']
                check=content_check(refs[g['id']]['transcript'],asr[cid]['transcript'])
                c['content_check']=check
                if check['passed']:passed.append(by_id[cid])
                else:
                    c['evaluation_status']='content_failed';c['attribute_reconstruction']=None
                    c['attribute_status']='attribute_not_requested_content_failed'
        self.status('attribute_labels',round=index,content_passed=len(passed),content_failed=len(generated)-len(passed))
        labels=service.attributes(passed)
        for g in groups:
            g['reference_attributes']=refs[g['id']]['attributes']
            for c in g['candidates']:
                cid=c['candidate_id']
                if cid in labels:
                    c['generated_attributes']=labels[cid]
                    c['attribute_reconstruction']=reconstruction(refs[g['id']]['attributes'],labels[cid])
                    c['evaluation_status']='complete';c['attribute_status']='complete'
        scored=score_groups(groups)
        self._local_stage(name=f'round_{index:03d}_audio_rewards',inputs=groups,outputs=scored,
                          directory=directory,action='v6_final_audio_reward',invocation_material=self.config['reward'])
        return scored

    @staticmethod
    def sft_rows(groups,codecs):
        return [{'id':g['id'],'caption':c['caption'],'request':c['request'],
                 'audio_path':g['audio_path'],'source_audio_path':g['audio_path'],
                 'codec_path':codecs[g['id']]['codec_path'],'target_origin':SOURCE_DOMAIN_TARGET,
                 'selected_candidate_id':c['candidate_id'],'selection_score':c['attribute_reconstruction']['score'],
                 'content_error_rate':c['content_check']['error_rate']}
                for g in groups for c in g['candidates'] if c['sft_selected']]

    def train(self,resume_only=False,prepare_only=False):
        service=LabelService(self.config)
        try:
            rows=self.prepare_data();refs=self.references(rows,service);rows=[r for r in rows if r['id'] in refs]
            if prepare_only:
                self.status('references_ready',training_audio_records=len(rows));return {'prepared':len(rows)}
            c_checkpoint=t_checkpoint=None
            for index in range(self.config['training']['rounds']):
                directory=self.run_dir/f'round_{index:03d}';commitpath=directory/'commit.json'
                if commitpath.exists():
                    record=json.loads(commitpath.read_text())
                    if record['input_captioner']!=checkpoint_record(c_checkpoint) or record['input_tts']!=checkpoint_record(t_checkpoint):
                        raise ValueError('Broken round checkpoint ancestry')
                    for role in ('captioner','tts'):
                        if checkpoint_record(record[role]['path'])!=record[role]:raise ValueError('Committed checkpoint hash mismatch')
                    c_checkpoint=record['captioner']['path'];t_checkpoint=record['tts']['path']
                    self.manager.set_current(c_checkpoint,t_checkpoint,index)
                    atomic_json(self.run_dir/'latest.json',record)
                    continue
                started=time.perf_counter()
                scored=self.collect(index,rows,refs,c_checkpoint,t_checkpoint,service)
                stats=summarize(scored);atomic_json(directory/'summary.json',stats)
                # Codec targets are always originals; never encode reconstruction as an SFT target.
                self.status('prepare_source_codecs',round=index)
                codecs_stage=self._stage(name='prepare_source_codecs',role='tts',action='prepare-codecs',
                    rows=[{'id':r['id'],'audio_path':r['audio_path']} for r in rows],directory=self.run_dir/'prepared')
                codecs={r['id']:r for r in read_jsonl(codecs_stage.output_path)}
                sft=self.sft_rows(scored,codecs)
                self.status('caption_grpo',round=index,usable_groups=stats['grpo_usable_groups'])
                final_c=directory/'checkpoints/caption_final'
                self._stage(name=f'round_{index:03d}_caption_grpo',role='captioner',action='grpo-update',rows=scored,
                    directory=directory/'training',checkpoint_in=c_checkpoint,target_checkpoint=c_checkpoint,checkpoint_out=final_c,phase='grpo')
                self.status('tts_sft',round=index,records=len(sft))
                final_t=directory/'checkpoints/tts_final'
                self._stage(name=f'round_{index:03d}_tts_sft',role='tts',action='sft-update',rows=sft,
                    directory=directory/'training',checkpoint_in=t_checkpoint,checkpoint_out=final_t,phase='cycle_sft')
                stats['elapsed_seconds_this_invocation']=time.perf_counter()-started
                stats['api_usage']=self.api_usage(f'round_{index:03d}')
                atomic_json(directory/'summary.json',stats)
                commit_round(self.run_dir,index,caption_checkpoint=str(final_c),tts_checkpoint=str(final_t),
                             input_caption_checkpoint=c_checkpoint,input_tts_checkpoint=t_checkpoint)
                c_checkpoint=str(final_c);t_checkpoint=str(final_t)
                self.manager.set_current(c_checkpoint,t_checkpoint,index)
                self.status('round_complete',round=index,**stats)
            self.status('training_complete',completed_rounds=self.config['training']['rounds'])
            return {'completed_rounds':self.config['training']['rounds'],'captioner':c_checkpoint,'tts':t_checkpoint}
        except BaseException as exc:
            self.status('interrupted_or_failed',error_type=type(exc).__name__,detail=str(exc))
            raise
        finally:service.close()

    def api_usage(self,scope):
        from collections import Counter
        events=Path(self.config['labeling']['cache_dir'])/'events.jsonl'
        counts=Counter();tokens=Counter()
        if events.exists():
            for row in read_jsonl(events):
                if row.get('scope')!=scope:continue
                model=row.get('model','local');kind=row.get('kind','')
                counts[f'{model}.{kind}']+=1
                usage=row.get('usage') or {}
                for k in ('prompt_tokens','completion_tokens','promptTokenCount','candidatesTokenCount','thoughtsTokenCount'):
                    if type(usage.get(k)) in (int,float):tokens[f'{model}.{k}']+=usage[k]
        return {'event_counts':dict(counts),'reported_tokens':dict(tokens),'currency_cost':None,
                'note':'Token usage from returned responses; transport failures may lack usage. Actual billing depends on provider.'}
