import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from dual_isl_train.config import load_config,public_config,validate_config
from dual_isl_train.schema_v6 import admit,render,ATTRS
from dual_isl_train.content_v6 import content_check,normalize_text
from dual_isl_train.reward_v6 import reconstruction,score_groups,EvaluationPending
from dual_isl_train.io import atomic_json,write_jsonl,read_jsonl,sha256_file
from dual_isl_train.constants import CHECKPOINT_METADATA,FRAMEWORK_VERSION,TRAJECTORY_VERSION

ROOT=Path(__file__).resolve().parents[1]
CAP={'transcript':'hello world','gender':'female','pitch_level':'high','emotion':'happy','emotion_intensity':'high'}


def candidate(i,attrs=None,status='complete'):
    return {'candidate_id':f'a::{i}','raw_text':json.dumps(CAP),**admit(json.dumps(CAP)),
            'trajectory_valid':True,'evaluation_status':status,'content_check':{'error_rate':0.0},
            'attribute_reconstruction':reconstruction(CAP,attrs or CAP) if status=='complete' else None,
            'request':render(CAP),'sampled_token_ids':[10+i]}


class ProtocolTests(unittest.TestCase):
    def test_partial_format_and_no_gt_repair(self):
        raw=json.dumps({'transcript':'hello','gender':'Female','emotion':'Happy','pitch_level':'ultra','extra':'bad'})
        out=admit(raw)
        self.assertEqual(out['format_score'],.8)
        self.assertEqual(out['caption']['gender'],'female')
        self.assertEqual(out['caption']['pitch_level'],'unknown')
        self.assertNotIn('Pitch',render(out['caption'])['instruct'])
        self.assertEqual(render(out['caption'])['language'],'Auto')
        self.assertEqual(out['valid_fields'],['transcript','gender','emotion'])

    def test_duplicate_or_nested_json_not_admitted(self):
        for s in ('{"transcript":"one","transcript":"two"}','[]','{"x":NaN}',json.dumps({'Target_JSON_Schema':CAP})):
            self.assertFalse(admit(s)['json_parseable'])
        self.assertEqual(admit('{}')['format_score'],.5)
        self.assertEqual(admit(json.dumps({**CAP,'gender':'unknown'}))['format_score'],1)

    def test_neutral_intensity_drops_condition_only(self):
        a=admit(json.dumps({**CAP,'emotion':'neutral'}))
        self.assertEqual(a['caption']['emotion_intensity'],'unknown')
        self.assertEqual(a['format_score'],1)
        self.assertNotIn('Emotion intensity',render(a['caption'])['instruct'])

    def test_exact_ten_percent_boundary_and_empty(self):
        a='甲乙丙丁戊己庚辛壬癸'
        self.assertTrue(content_check(a,'甲乙丙丁戊己庚辛壬字')['passed'])
        self.assertFalse(content_check(a,'甲乙丙丁戊己庚辛字字')['passed'])
        self.assertFalse(content_check('hello','')['passed'])
        self.assertEqual(content_check('one','two three four')['error_rate'],3)
        with self.assertRaises(ValueError):content_check('','test')

    def test_content_normalization(self):
        self.assertEqual(content_check('Hello, WORLD!','hello world')['error_rate'],0)
        self.assertEqual(content_check('二十个','20个')['error_rate'],0)
        self.assertEqual(content_check('We need twenty five items.','we need 25 items')['error_rate'],0)
        self.assertEqual(content_check('2026年','二〇二六年')['error_rate'],0)
        self.assertNotEqual(normalize_text('not happy'),normalize_text('happy'))
        self.assertNotEqual(normalize_text('1.5'),normalize_text('15'))
        self.assertEqual(normalize_text('一会'), '一会')

    def test_fixed_reference_mask(self):
        out=reconstruction({**CAP,'gender':'unknown'}, {**CAP,'pitch_level':'unknown'})
        self.assertEqual(out['denominator'],3)
        self.assertAlmostEqual(out['score'],2/3)
        with self.assertRaises(ValueError):reconstruction({k:'unknown' for k in ATTRS},CAP)

    def test_content_failure_format_only_and_pending(self):
        bad=candidate(1,status='content_failed');good=candidate(0)
        gs=score_groups([{'id':'a','candidates':[good,bad]}])[0]['candidates']
        self.assertEqual(gs[1]['reward'],.1);self.assertFalse(gs[1]['sft_selected'])
        self.assertFalse(gs[1]['skip_update']);self.assertLess(gs[1]['advantage'],0)
        with self.assertRaises(EvaluationPending):score_groups([{'candidates':[candidate(0,status='evaluation_pending')]}])

    def test_zero_top1_sft_even_when_grpo_flat(self):
        wrong={k:'unknown' for k in ATTRS}
        a,b=candidate(0,wrong),candidate(1,wrong)
        a['content_check']['error_rate']=.1
        cs=score_groups([{'candidates':[a,b]}])[0]['candidates']
        self.assertTrue(all(c['skip_update'] for c in cs))
        self.assertTrue(cs[1]['sft_selected']);self.assertFalse(cs[0]['sft_selected'])
        self.assertEqual(cs[1]['attribute_reconstruction']['score'],0)

    def test_configuration_rejects_old_training(self):
        c=public_config(load_config(ROOT/'configs/v6.yaml'))
        for role,phase in [('tts','grpo'),('captioner','cycle_sft')]:
            wrong=deepcopy(c);wrong[role]['training']['phases'][phase]={'epochs':1}
            with self.assertRaises(ValueError):validate_config(wrong)
        wrong=deepcopy(c);wrong['reward']['sft_attribute_threshold']=.75
        with self.assertRaises(ValueError):validate_config(wrong)


class IntegrationTests(unittest.TestCase):
    def test_legacy_reference_transcription_is_restored(self):
        import numpy as np
        import soundfile as sf
        from dual_isl_train.data_v6 import build_records
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);labels=[]
            for i,role in enumerate(('paired','audio_only','caption_only')):
                path=root/f'{i}.wav';sf.write(path,np.ones(16000)*(i+.1)/10,16000)
                row={'id':str(i),'split':'train','audio_path':'' if role=='caption_only' else str(path),
                     'caption':{'semantic_content':{'transcript':'outdated'}}}
                write_jsonl(root/(role+'.jsonl'),[row])
                labels.append({'sample_id':str(i),'audio_path':f'/old/server/{i}.wav',
                               'Target_JSON_Schema':{'semantic_content':{'transcription':'corrected reference'}}})
            write_jsonl(root/'labels.jsonl',labels)
            rows,report=build_records({'source_dir':str(root),'original_audio_dir':str(root),
                        'original_labels_path':str(root/'labels.jsonl'),'duration':{'min':.5,'max':2},'max_records':0})
            self.assertEqual(len(rows),3)
            self.assertTrue(all(r['reference_transcript']=='corrected reference' for r in rows))
            self.assertEqual(len(report['transcript_conflicts_with_old_manifest']),3)

    def test_two_rounds_asr_shortcircuit_source_gt_and_resume(self):
        from dual_isl_train.orchestrator_v6 import DualRSIOrchestrator
        from scripts.verify_v6 import verify
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source.wav';sf.write(source,np.zeros(16000),16000)
            cfg=public_config(load_config(ROOT/'configs/v6.yaml'));cfg['run'].update(output_dir=str(root/'run'),inline_mock=True)
            cfg['training']['rounds']=2;cfg['distributed'].update(enabled=False,world_size=1)
            cfg['labeling']['cache_dir']=str(root/'run/labels');cfg['tts']['codec_cache_dir']=str(root/'run/codecs')
            rows=[{'id':'a','audio_path':str(source),'audio_sha256':sha256_file(source),'reference_transcript':'hello world','transcript_origin':'fixture'}]
            calls=[];attribute_inputs=[];fail={'once':True}
            class Service:
                identity='fixture'
                def __init__(self,c):pass
                def asr(self,items):return {r['id']:{'transcript':'wrong words' if r['id'].endswith('::1') else 'hello world'} for r in items}
                def attributes(self,items):
                    attribute_inputs.extend(r['id'] for r in items)
                    return {r['id']:{k:CAP[k] for k in ATTRS} for r in items}
                def close(self):pass
            def worker(**kw):
                action=kw['action'];role='captioner' if 'captioner' in kw['module'] else 'tts'
                items=list(read_jsonl(kw['input_path']));calls.append((role,action,str(kw['output_path'])))
                if action=='rollout':
                    result=[]
                    for r in items:
                        self.assertNotIn('reference_transcript',r);self.assertNotIn('reference_attributes',r)
                        cs=[]
                        for i in range(r['group_size']):
                            cap={**CAP,'transcript':'wrong words' if i==1 else 'hello world'}
                            cs.append({'candidate_id':f'a::{i}','raw_text':json.dumps(cap),'sampled_token_ids':[i+1],
                                       'old_token_logprobs':[-.2],'ref_token_logprobs':[-.2],'trajectory_valid':True})
                        result.append({**r,'candidates':cs})
                elif action=='generate-audio':
                    self.assertEqual(items[1]['request']['text'],'wrong words')
                    result=[]
                    for r in items:
                        p=root/(r['id'].replace(':','_')+'.wav');sf.write(p,np.ones(1600)*.01,16000)
                        result.append({**r,'audio_path':str(p),'synthesis_batch_size':8})
                elif action=='prepare-codecs':
                    p=root/'codes.json';atomic_json(p,{'codec_codes':[[0]*16]})
                    result=[{**r,'codec_path':str(p)} for r in items]
                else:
                    self.assertIn((role,action),(('captioner','grpo-update'),('tts','sft-update')))
                    if role=='tts':
                        self.assertEqual(len(items),1);self.assertEqual(items[0]['source_audio_path'],str(source))
                        self.assertEqual(items[0]['audio_path'],str(source))
                        if fail['once']:
                            fail['once']=False;raise RuntimeError('simulated interruption after Captioner update')
                    out=kw['checkpoint_out'];out.mkdir(parents=True,exist_ok=True)
                    atomic_json(out/CHECKPOINT_METADATA,{'framework_version':FRAMEWORK_VERSION,'trajectory_version':TRAJECTORY_VERSION,
                        'update':'grpo' if role=='captioner' else 'sft','steps':1,'parameter_before':0,'parameter_after':1,'parameter_delta':{'ok':True}})
                    result=[{'status':'updated','steps':1}]
                write_jsonl(kw['output_path'],result)
            with patch('dual_isl_train.orchestrator_v6.build_records',return_value=(rows,{'audio_records':1})), \
                 patch('dual_isl_train.orchestrator_v6.LabelService',Service), \
                 patch('dual_isl_train.stage_runner.run_worker',side_effect=worker):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):DualRSIOrchestrator(cfg).train()
                DualRSIOrchestrator(cfg).train(resume_only=True)
                before=len(calls);DualRSIOrchestrator(cfg).train(resume_only=True);self.assertEqual(len(calls),before)
            self.assertNotIn('a::1',attribute_inputs)
            self.assertEqual(sum(role=='captioner' and action=='grpo-update' for role,action,_ in calls),2)
            self.assertEqual(sum(action=='prepare-codecs' for _,action,_ in calls),1)
            report=verify(root/'run');self.assertTrue(report['ok'],report['errors'])
            self.assertEqual(len(report['rounds']),2)


if __name__=='__main__':unittest.main()
