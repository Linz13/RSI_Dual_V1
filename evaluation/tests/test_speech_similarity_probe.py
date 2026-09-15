from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speech_similarity_probe as probe


class SimilarityProbeTests(unittest.TestCase):
    def test_chunked_scores_match_official_for_unequal_lengths(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(21)
        a, b = torch.randn(17, 9), torch.randn(31, 9)
        official = probe.load_official_speechbertscore_module().bert_score(a, b)
        for block in (1, 7, 128):
            actual = probe.similarity(a, b, block)
            self.assertAlmostEqual(actual['sbs'], float(official[0]), places=6)
            self.assertAlmostEqual(actual['recall'], float(official[1]), places=6)
            self.assertAlmostEqual(actual['cosine'], float(torch.nn.functional.cosine_similarity(a.mean(0), b.mean(0), dim=0)), places=6)
        reverse = probe.similarity(b, a)
        self.assertAlmostEqual(reverse['recall'], float(official[0]), places=6)

    def test_self_similarity_and_invalid_embeddings(self):
        import torch
        a = torch.eye(3)
        self.assertTrue(all(abs(v-1) < 1e-6 for v in probe.similarity(a, a).values()))
        for bad in (torch.empty(0, 3), torch.zeros(2, 3), torch.full((2, 3), float('nan'))):
            with self.assertRaises(ValueError):
                probe.similarity(bad, a)

    def test_transcript_filter_is_not_word_error_rate(self):
        self.assertEqual(probe.transcript_distance('Hello, WORLD!', 'hello world'), 0)
        self.assertEqual(probe.transcript_distance('你好，世界。', '你好世界'), 0)
        self.assertAlmostEqual(probe.transcript_distance('abcde', 'abXde'), .2)
        for value in ('', 'unknown', None):
            self.assertIsNone(probe.transcript_distance('hello', value))

    def test_cache_rejects_stale_missing_duplicate_and_nonfinite(self):
        g = {'id': 'g', 'candidates': [{'id': 'a'}, {'id': 'b'}]}
        value = {'plan_hash': 'hash', 'encoder': 'wavlm', 'group_id': 'g',
                 'candidates': [{'id': x, 'sbs': .7, 'recall': .8, 'cosine': .9} for x in ('a', 'b')]}
        probe.validate_result(value, 'hash', 'wavlm', g)
        variants = []
        bad = deepcopy(value); bad['plan_hash'] = 'old'; variants.append(bad)
        bad = deepcopy(value); bad['candidates'].pop(); variants.append(bad)
        bad = deepcopy(value); bad['candidates'].append(bad['candidates'][0]); variants.append(bad)
        bad = deepcopy(value); bad['candidates'][0]['sbs'] = float('nan'); variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):
                probe.validate_result(bad, 'hash', 'wavlm', g)

    def test_content_filter_changes_comparable_groups(self):
        cs = [{'id': x, 'transcript_matched': i == 0, 'reconstruction': .6,
               **{m: .5+i*.1 for m in probe.METRICS}} for i, x in enumerate(('a', 'b'))]
        all_report = probe.subset_report([{'candidates': cs}], False)
        filtered = probe.subset_report([{'candidates': cs}], True)
        self.assertEqual(all_report['comparable_groups'], 1)
        self.assertEqual(filtered['comparable_groups'], 0)
        self.assertEqual(filtered['candidates'], 1)
        self.assertEqual(all_report['metrics']['wavlm_sbs']['reference_reward_top_overlap_groups'], 1)

    def test_report_listening_and_vote_import_end_to_end(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            wav = out/'audio.wav'
            sf.write(wav, .1*np.sin(np.arange(3200)*.05), 16000)
            asset = {'path': str(wav), 'sha256': probe.sha256_file(wav)}
            cs = [{'id': x, 'audio': asset, 'transcript_matched': True, 'reconstruction': .5,
                   'generated_cached_transcript': 'hello'} for x in ('a', 'b')]
            group = {'id': 'g', 'language': 'English', 'reference_audio': asset,
                     'reference_transcript': 'hello', 'candidates': cs}
            plan = {'groups': [group]}; ph = probe.sha256_json(plan)
            probe.write(out/'plan.json', plan)
            for name in probe.MODELS:
                scored = [{'id': x, 'sbs': .8-i*.1, 'recall': .7, 'cosine': .6+i*.1} for i, x in enumerate(('a', 'b'))]
                probe.write(probe.result_path(out, name, 'g'), {'plan_hash': ph, 'encoder': name,
                            'group_id': 'g', 'candidates': scored})
            probe.write(out/'votes.json', {'plan_hash': ph, 'votes': {'g': 'a'}})
            probe.analyze(out, out/'votes.json')
            summary = probe.read(out/'summary.json')
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['all']['main_metrics_disagree_groups'], 1)
            self.assertEqual(summary['human_listening']['wavlm_sbs']['selected_in_metric_top'], 1)
            self.assertEqual(summary['human_listening']['xlsr_cosine']['selected_in_metric_top'], 0)
            self.assertIn('random', summary['human_listening_by_sampling'])
            page = (out/'listen.html').read_text()
            self.assertIn('data:audio/wav;base64,', page)
            self.assertNotIn('PLAN_HASH', page)
            self.assertTrue((out/'report.md').exists())


if __name__ == '__main__':
    unittest.main()
