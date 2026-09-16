#!/usr/bin/env python3
"""Recompute archived A/B metrics and token means using only Python's standard library."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/bidirectional_scoring'
sys.path.insert(0, str(EXPERIMENT / 'src'))
from common import digest, description, validate_annotations
from summarize import a_metrics, b_metrics, bootstrap_uplift


def load(path):
    return json.loads(path.read_text())


def main():
    provenance = load(ROOT / 'provenance/bidirectional_scoring.json')
    for entry in provenance['files']:
        assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['export_sha256'], entry['path']
    results = []
    for name, a_tag, b_tag, report in [('smoke01', 'v3_fp32', 'v2', 'v3_combined'),
                                      ('pilot30', 'v1', 'v1', 'v1')]:
        run = EXPERIMENT / 'results' / name
        manifest = load(run / 'manifest.json')
        ann = load(run / 'annotations.json')
        summary = load(run / 'reports' / report / 'summary.json')
        assert digest(manifest['identity']) == manifest['manifest_id'] == summary['manifest_id']
        validate_annotations(ann, manifest)
        metrics = {'a': [], 'b': []}
        count = 0
        max_error = 0.0
        score_sets = {}
        for kind, tag in [('a', a_tag), ('b', b_tag)]:
            folder = run / 'scores' / kind / tag
            assert load(folder / 'status.json')['status'] == 'complete'
            identity = load(folder / 'identity.json')
            assert digest(identity['identity']) == identity['identity_key']
            assert identity['identity']['manifest_id'] == manifest['manifest_id']
            assert load(folder / 'numeric_audit.json')['passed'] is True
            items = {p.stem: load(p) for p in sorted((folder / 'items').glob('*.json'))}
            score_sets[kind] = items
            reported = {r['id']: r for r in summary[kind]['rows']}
            for row in manifest['identity'][kind]:
                rid = row['id']
                label = ann[kind][rid]
                if kind == 'a' and label['status'] != 'verified':
                    assert rid not in items and reported[rid]['status'] == 'human_' + label['status']
                    continue
                item = items[rid]
                assert item['status'] == 'ok' and reported[rid]['status'] == 'scored'
                cs = item['conditions'] if kind == 'a' else item['candidates']
                assert len(cs) == 4
                for c in [*cs, *item.get('controls', {}).values()]:
                    tokens = c['token_logprobs']
                    if kind == 'a':
                        assert c['codebooks'] == 16 and all(len(v) == 16 for v in tokens)
                        assert len(tokens) == c['codec_frames'] and c['audio_eos_scored'] is False
                        for k, average in enumerate(c['per_codebook_mean']):
                            assert abs(math.fsum(frame[k] for frame in tokens)/len(tokens)-average) < 1e-10
                        tokens = [v for frame in tokens for v in frame]
                    else:
                        assert not c['target_in_user_prompt']
                        assert c['target_end']-c['prefix_tokens'] == c['target_tokens']
                        assert c['excluded_suffix_tokens'] == 2
                    assert len(tokens) == c['target_tokens'] and all(math.isfinite(v) for v in tokens)
                    error = abs(math.fsum(tokens)/len(tokens)-c['score'])
                    assert error < 1e-10
                    count += 1
                    max_error = max(max_error, error)
                if kind == 'a':
                    assert item['reference_emotion'] == label['emotion']
                    assert item['verified_transcript'] == label['transcript'].strip()
                    assert item['audio_sha256'] == row['audio_sha256']
                    for c in cs:
                        assert c['description'] == description(c['emotion'])
                        assert c['is_positive'] == (c['emotion'] == label['emotion'])
                    metric = a_metrics(item, summary['tie_epsilon'])
                else:
                    assert item['target'] == description(row['emotion'])
                    assert {c['blind_id']: c['audio_sha256'] for c in cs} == {c['blind_id']: c['audio_sha256'] for c in row['candidates']}
                    assert all(c['target'] == item['target'] for c in cs)
                    metric = b_metrics(item, label, summary['tie_epsilon'])
                for k, value in metric.items():
                    assert reported[rid][k] == value, (name, rid, k)
                metrics[kind].append(metric)
        a = metrics['a']
        b = [r for r in metrics['b'] if r['agreement'] is not None]
        strict = sum(r['strict_top1'] for r in a)
        pairs = {label: sum(m['outcome'] == label for r in a for m in r['margins']) for label in ['win', 'loss', 'tie']}
        agreement = sum(r['agreement'] for r in b)/len(b)
        random_baseline = sum(r['random_baseline'] for r in b)/len(b)
        assert strict == summary['a']['strict_top1_count'] and strict/len(a) == summary['a']['strict_top1_rate']
        assert pairs == summary['a']['pairwise']
        assert len(b) == summary['b']['comparable_groups'] and agreement == summary['b']['agreement']
        assert random_baseline == summary['b']['random_baseline']
        assert bootstrap_uplift(b) == summary['b']['descriptive_group_bootstrap_95']
        for sensitivity in summary['tie_sensitivity']:
            eps = sensitivity['epsilon']
            am = [a_metrics(item, eps) for item in score_sets['a'].values()]
            bm = [b_metrics(item, ann['b'][rid], eps) for rid, item in score_sets['b'].items()]
            comparable = [r for r in bm if r['agreement'] is not None]
            assert sum(r['strict_top1'] for r in am)/len(am) == sensitivity['a_strict_top1_rate']
            assert sum(r['tied_top1'] for r in am) == sensitivity['a_tied_top1_count']
            assert sum(r['agreement'] for r in comparable)/len(comparable) == sensitivity['b_agreement']
            assert sum(r['model_tie'] for r in bm) == sensitivity['b_model_tie_groups']
        results.append({'run': name, 'a_valid': len(a), 'a_strict_first': strict, 'a_pairwise': pairs,
                        'b_scored': len(metrics['b']), 'b_comparable': len(b), 'b_agreement': agreement,
                        'b_random_baseline': random_baseline, 'recomputed_scores_including_controls': count,
                        'max_mean_error': max_error})
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    print(json.dumps({'status': 'passed', 'provenance_files_verified': len(provenance['files']),
                      'results': results, 'scope': 'saved text artifacts only; no audio verification, model loading, GPU, training or API'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
