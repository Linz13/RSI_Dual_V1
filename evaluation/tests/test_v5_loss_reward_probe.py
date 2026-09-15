from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v5_loss_reward_probe as probe


class ProbeTests(unittest.TestCase):
    def caption(self):
        return {'semantic_content':{'transcript':'correct text','language':'English'},
                'speaker_profile':{'gender':'male','age':'adult','accent':'US English','timbre':'neutral'},
                'paralinguistic':{'emotion':'angry','emotion_intensity':'high','speaking_rate':'moderate',
                    'pitch_level':'medium','volume_level':'medium','emphasis':{'level':'none','emphasized_text':[]},
                    'prosody':'expressive','pause':'none','nonverbal_vocalization':['none']}}

    def test_only_text_language_replaced_and_input_not_mutated(self):
        ref=self.caption();candidate=deepcopy(ref)
        candidate['semantic_content']={'transcript':'wrong sentence','language':'Chinese'}
        candidate['speaker_profile']['gender']='female'
        before=deepcopy(candidate)
        result=probe.fixed_request(candidate,ref)
        self.assertEqual(result['text'],'correct text')
        self.assertEqual(result['language'],'English')
        self.assertIn('gender: female',result['instruct'])
        self.assertEqual(candidate,before)

    def test_sigmoid_calibration_direction_and_extremes(self):
        m=3.5;s=1/math.log(9)
        self.assertAlmostEqual(probe.sigmoid_reward(2.5,m,s),.9)
        self.assertAlmostEqual(probe.sigmoid_reward(4.5,m,s),.1)
        self.assertEqual(probe.sigmoid_reward(m,m,s),.5)
        self.assertEqual(probe.sigmoid_reward(1e9,m,s),0)
        self.assertEqual(probe.sigmoid_reward(-1e9,m,s),1)

    def test_controls_skip_invalid_reference_without_altering_candidates(self):
        ref = self.caption(); before = deepcopy(ref)
        control = probe.reference_controls(ref)
        self.assertIn('emotion: sad', control['wrong_emotion']['instruct'])
        self.assertIn('emotion intensity: high', control['wrong_emotion']['instruct'])
        self.assertIn('gender: female', control['wrong_gender']['instruct'])
        self.assertEqual(ref, before)
        ref['paralinguistic']['emotion'] = 'neutral'
        self.assertIsNone(probe.reference_controls(ref))
        # Invalid reference attributes do not enter the actual candidate prompt.
        request = probe.fixed_request(before, ref)
        self.assertIn('emotion: angry', request['instruct'])
        self.assertEqual(request['text'], 'correct text')

    def test_result_cache_rejects_identity_nonfinite_and_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'scores.jsonl'
            row={'id':'a','plan_hash':'right','loss':3,'main_nll':2,'sub_nll':10/3}
            for rs in ([{**row,'plan_hash':'wrong'}],[{**row,'loss':float('nan')}],[row,row]):
                p.write_text(''.join(json.dumps(r)+'\n' for r in rs))
                with self.assertRaises(ValueError):probe.load_results(p,'right')

    def test_analysis_uses_disjoint_calibration_and_reports_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'scores').mkdir()
            jobs=[]
            for split in ('calibration','analysis'):
                for i in range(4):
                    jobs.append({'id':f'{split}_{i}','group_id':split,'split':split,'kind':'candidate',
                                 'reconstruction':.5+i*.01,'format':1,'transcript_exact_normalized':True})
                jobs.append({'id':split+'_text','group_id':split,'split':split,'kind':'text_only'})
            for kind in ('reference','wrong_gender','wrong_emotion'):
                jobs.append({'id':kind,'group_id':'analysis','split':'analysis','kind':kind})
            plan={'jobs':jobs,'shards':[[j['id'] for j in jobs]],'calibration_groups':['calibration'],
                  'checkpoint':'round-start','notes':[]}
            ph=probe.sha256_json(plan)
            scores=[]
            for j in jobs:
                loss=3.0
                if j['kind']=='candidate':
                    loss=2.5+int(j['id'][-1])*.5 if j['split']=='calibration' else 3-int(j['id'][-1])*.2
                if j['kind']=='wrong_gender':loss=3.2
                if j['kind']=='wrong_emotion':loss=2.8
                scores.append({'id':j['id'],'plan_hash':ph,'loss':loss,'main_nll':loss-1,'sub_nll':10/3})
            (root/'scores/00.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores))
            result=probe.analyze(root,plan)
            self.assertEqual(result['calibration']['candidate_losses']['count'],4)
            self.assertEqual(result['analysis_candidate_loss']['count'],4)
            self.assertEqual(result['weights']['0.2']['groups'],1)
            self.assertEqual(result['controls']['wrong_gender']['reference_lower_loss'],1)
            self.assertEqual(result['controls']['wrong_emotion']['wrong_lower_loss'],1)
            self.assertTrue((root/'report.md').is_file())


if __name__=='__main__':
    unittest.main()
