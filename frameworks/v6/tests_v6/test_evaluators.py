import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dual_isl_train.config import load_config
from dual_isl_train.labeling_v6 import LabelService
from dual_isl_train.reward_v6 import EvaluationPending
from dual_isl_train.local_v6 import emotion_prediction

ROOT=Path(__file__).resolve().parents[1]


class EvaluatorTests(unittest.TestCase):
    def test_remote_partial_unknown_cache_and_usage(self):
        cfg=load_config(ROOT/'configs/v6.yaml')
        with tempfile.TemporaryDirectory() as d:
            cfg['labeling']['cache_dir']=d;cfg['labeling']['backoff_seconds']=0
            s=LabelService(cfg);audio=Path(d)/'test.wav';audio.write_bytes(b'fixture')
            row={'id':'a','audio_path':str(audio)}
            with patch('dual_isl_train.api_transport.request_with_metadata',
                       return_value=('{"gender":"Female","emotion_intensity":"extreme"}',{'usage':{'prompt_tokens':10,'completion_tokens':7},'model':'qwen3.5-omni-plus'})) as request:
                out=s._remote(row,'qwen35')
                self.assertEqual(out,{'gender':'female','emotion_intensity':'unknown'})
                self.assertEqual(s._remote(row,'qwen35'),out);self.assertEqual(request.call_count,1)
            events=[json.loads(l) for l in (Path(d)/'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(v.get('usage',{}).get('completion_tokens')==7 for v in events))

    def test_transport_failure_never_becomes_unknown_or_zero(self):
        cfg=load_config(ROOT/'configs/v6.yaml')
        with tempfile.TemporaryDirectory() as d:
            cfg['labeling']['cache_dir']=d;cfg['labeling']['backoff_seconds']=0
            s=LabelService(cfg);audio=Path(d)/'test.wav';audio.write_bytes(b'fixture')
            with patch('dual_isl_train.api_transport.request_with_metadata',side_effect=TimeoutError) as request:
                with self.assertRaises(EvaluationPending):s._remote({'id':'a','audio_path':str(audio)},'gemini')
                self.assertEqual(request.call_count,3)
            self.assertFalse(list(Path(d).glob('remote/*.json')))

    def test_emotion_mapping_does_not_guess_missing_prediction(self):
        class Model:
            def __init__(self,value):self.value=value
            def generate(self,**kwargs):return [self.value]
        self.assertEqual(emotion_prediction(Model({'labels':['开心/happy','其他/other'],'scores':[.9,.1]}),'x')['emotion'],'happy')
        self.assertEqual(emotion_prediction(Model({'labels':['<unk>'],'scores':[1]}),'x')['emotion'],'unknown')
        with self.assertRaises(ValueError):emotion_prediction(Model({'labels':[],'scores':[]}),'x')


if __name__=='__main__':unittest.main()
