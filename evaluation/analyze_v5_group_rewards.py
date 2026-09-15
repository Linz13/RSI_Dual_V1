#!/usr/bin/env python3
"""Read committed rollout rewards and make an offline listening/report artifact."""
import base64
from collections import Counter
import csv
import hashlib
import html
import itertools
import json
from pathlib import Path
import statistics as st

import soundfile as sf

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01'
OUT = ROOT / 'reports/v5_round000_001_group_rewards'


def read_groups(round_index, branch):
    path = RUN / f'round_{round_index:03d}/rewards/round_{round_index:03d}_{branch}_rewards.output.jsonl'
    groups = []
    keep = {'candidate_id', 'caption', 'generated_attributes', 'reconstructed_audio_path', 'audio_path',
            'attribute_reconstruction', 'reconstruction_score', 'format_score', 'semantic_valid',
            'semantic_input_valid', 'trajectory_valid', 'reward', 'semantic_reward', 'sft_selected',
            'reward_components_raw', 'reward_components_normalized', 'audio_health_details'}
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for line in source:
            digest.update(line)
            if not line.strip():
                continue
            g = json.loads(line)
            groups.append({**{k: g[k] for k in ('id', 'audio_path', 'reference_attributes', 'request', 'source_caption') if k in g},
                           'candidates': [{k: v for k, v in c.items() if k in keep} for c in g['candidates']]})
    return groups, {'path': str(path), 'sha256': digest.hexdigest()}


def distribution(values):
    v = sorted(values)
    def quantile(p):
        i = (len(v) - 1) * p
        lo, hi = int(i), min(int(i) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)
    if not v:
        return {'count': 0}
    return {'count': len(v), 'mean': st.fmean(v), 'min': v[0], 'p10': quantile(.1),
            'p25': quantile(.25), 'median': st.median(v), 'p75': quantile(.75), 'p90': quantile(.9), 'max': v[-1]}


def score(c, branch):
    return c.get('reconstruction_score') if branch == 'audio' else c.get('reward_components_raw', {}).get('reconstruction')


def eligible(g, branch):
    return [c for c in g['candidates'] if c.get('semantic_valid') and type(score(c, branch)) in (int, float)]


def metrics(g, branch):
    cs = eligible(g, branch)
    v = sorted(score(c, branch) for c in cs)
    result = {'id': g['id'], 'valid_candidates': len(v), 'scores': [score(c, branch) for c in cs]}
    if len(v) >= 2:
        result.update(mean=st.fmean(v), span=v[-1]-v[0], std=st.pstdev(v), top2_gap=v[-1]-v[-2],
                      pairwise_gap=st.fmean(abs(a-b) for a, b in itertools.combinations(v, 2)))
    return result


def summarize(groups, branch):
    ms = [metrics(g, branch) for g in groups]
    comparable = [m for m in ms if m['valid_candidates'] >= 2]
    cs = [c for g in groups for c in eligible(g, branch)]
    summary = {'groups': len(groups), 'candidates': sum(len(g['candidates']) for g in groups),
               'valid_candidates': len(cs), 'valid_count_histogram': dict(Counter(m['valid_candidates'] for m in ms)),
               'comparable_groups': len(comparable), 'candidate_score': distribution([score(c, branch) for c in cs]),
               **{key: distribution([m[key] for m in comparable]) for key in ('span', 'std', 'top2_gap', 'pairwise_gap')},
               'span_thresholds': {'equal': sum(m['span'] < 1e-9 for m in comparable),
                                   'le_0.01': sum(m['span'] <= .01 + 1e-9 for m in comparable),
                                   'le_0.05': sum(m['span'] <= .05 + 1e-9 for m in comparable),
                                   'ge_0.10': sum(m['span'] >= .10 - 1e-9 for m in comparable)},
               'four_candidate_span': distribution([m['span'] for m in comparable if m['valid_candidates'] == 4])}
    if branch == 'audio':
        summary['evaluation_status'] = dict(Counter(c.get('attribute_reconstruction', {}).get('status')
                                                   for g in groups for c in g['candidates']))
        summary['reference_denominators'] = dict(Counter(c['attribute_reconstruction']['denominator'] for c in cs))
        summary['groups_with_inconsistent_reference_mask'] = sum(len({tuple(sorted(c['attribute_reconstruction']['fields']))
                                                                   for c in eligible(g, branch)}) > 1 for g in groups)
        fields = sorted({f for c in cs for f in c['attribute_reconstruction']['fields']})
        summary['field_discrimination'] = {}
        for field in fields:
            values = [[c['attribute_reconstruction']['fields'][field] for c in eligible(g, branch)
                       if field in c['attribute_reconstruction']['fields']] for g in groups]
            values = [v for v in values if len(v) >= 2]
            summary['field_discrimination'][field] = {
                'comparable_groups': len(values), 'differing_groups': sum(max(v)-min(v) > 1e-9 for v in values),
                'mean_span': st.fmean(max(v)-min(v) for v in values)}
        for c in cs:
            ev = c['attribute_reconstruction']
            assert abs(st.fmean(ev['fields'].values()) - score(c, branch)) < 1e-10
            assert abs(c['reward'] - (.9 * score(c, branch) + .1 * c['format_score'])) < 1e-10
        assert summary['groups_with_inconsistent_reference_mask'] == 0
    else:
        summary['semantic_reward'] = distribution([c['semantic_reward'] for c in cs])
        summary['semantic_reward_span'] = distribution([max(c['semantic_reward'] for c in eligible(g, branch)) -
                                                        min(c['semantic_reward'] for c in eligible(g, branch))
                                                        for g in groups if len(eligible(g, branch)) >= 2])
    return summary, ms


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    path.chmod(0o666)


def player(path):
    path = Path(path)
    info = sf.info(path)
    # Original WAV bytes: no resampling, normalization, trimming or re-synthesis.
    wav = path.read_bytes()
    assert info.frames > 0
    encoded = base64.b64encode(wav).decode('ascii')
    meta = {'source_path': str(path), 'sha256': hashlib.sha256(wav).hexdigest(),
            'duration_seconds': info.duration, 'sample_rate': info.samplerate, 'channels': info.channels}
    return f'<audio controls preload="none" src="data:audio/wav;base64,{encoded}"></audio><small>{info.duration:.2f} 秒</small>', meta


def get(obj, path):
    for key in path.split('.'):
        obj = obj.get(key, {}) if isinstance(obj, dict) else {}
    return obj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, report, csv_rows = {}, {'run': str(RUN), 'sources': [], 'rounds': {}}, []
    for r in (0, 1):
        assert json.loads((RUN / f'round_{r:03d}/commit.json').read_text())['round'] == r
        report['rounds'][str(r)] = {}
        for branch in ('audio', 'caption'):
            groups, source = read_groups(r, branch)
            data[(r, branch)] = groups
            report['sources'].append(source)
            summary, ms = summarize(groups, branch)
            report['rounds'][str(r)][branch] = summary
            csv_rows.extend({'round_index': r, 'branch': branch, **m} for m in ms)
    indexed = {r: {g['id']: g for g in data[(r, 'audio')]} for r in (0, 1)}
    common = sorted(set(indexed[0]) & set(indexed[1]))
    comparable = [gid for gid in common if all(len(eligible(indexed[r][gid], 'audio')) >= 2 for r in (0, 1))]
    report['matched_audio_groups'] = {'count': len(comparable), **{
        str(r): {'mean_group_score': st.fmean(st.fmean(score(c, 'audio') for c in eligible(indexed[r][gid], 'audio')) for gid in comparable),
                 'mean_span': st.fmean(metrics(indexed[r][gid], 'audio')['span'] for gid in comparable)} for r in (0, 1)}}
    dump(OUT / 'statistics.json', report)
    with (OUT / 'all_groups.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['round_index', 'branch', 'id', 'valid_candidates', 'scores', 'mean', 'span', 'std', 'top2_gap', 'pairwise_gap'])
        writer.writeheader()
        writer.writerows(csv_rows)

    picks = [(1, 'audio', 'random440_expressivespeech_266', '第二轮：差距大（中文）'),
             (1, 'audio', 'random440_expressivespeech_065', '第二轮：差距最小的完整四候选组'),
             (1, 'audio', 'random440_expressivespeech_321', '第二轮：差距接近总体中位数'),
             (0, 'audio', 'random440_expressivespeech_004', '第一轮：差距较小'),
             (0, 'audio', 'random440_expressivespeech_183', '第一轮：差距接近总体中位数'),
             (0, 'audio', 'random440_expressivespeech_313', '第一轮：差距最大')]
    # Same instruction across rounds and within each four-way group: isolate TTS sampling from caption variation.
    picks += [(r, 'caption', 'random440_expressivespeech_164', f'第{r+1}轮：同一 TTS 输入，四次采样') for r in (0, 1)]
    esc = lambda v: html.escape(str(v))
    sections, selected = [], []
    for r, branch, gid, label in picks:
        g = next(g for g in data[(r, branch)] if g['id'] == gid)
        assert len(eligible(g, branch)) == 4
        m = metrics(g, branch)
        blocks = [f'<section><h2>{esc(label)}</h2><p>{esc(gid)} · {branch}-only</p>']
        sample = {'round_index': r, 'branch': branch, 'id': gid, 'selection': label,
                  'span': m['span'], 'candidates': []}
        if branch == 'audio':
            audio, meta = player(g['audio_path'])
            sample['reference_audio'] = meta
            blocks.append('<div class="reference"><b>原始音频</b>' + audio + '</div>')
        else:
            blocks.append('<p>四条音频使用完全相同的 TTS 输入，只改变采样随机种子。</p>')
        blocks.append('<div class="players">')
        for i, c in enumerate(g['candidates']):
            audio_path = c['reconstructed_audio_path'] if branch == 'audio' else c['audio_path']
            audio, meta = player(audio_path)
            sample['candidates'].append({'candidate_id': c['candidate_id'], 'score': score(c, branch),
                                         'sft_selected': c['sft_selected'], **meta})
            blocks.append(f'<div><b>候选 {i+1}</b>{audio}</div>')
        blocks.append('</div><details><summary>听完后展开：奖励分数、输入 caption、生成音频打标</summary>')
        score_label = '属性重建分（0～1）' if branch == 'audio' else 'Captioner 重建 logprob（越大越好，非 0～1）'
        blocks.append(f'<p>{score_label}；同组最高减最低 = {m["span"]:.4f}</p><table><tr><th>候选</th><th>重建分</th><th>最终奖励</th><th>SFT 选中</th></tr>')
        for i, c in enumerate(g['candidates']):
            blocks.append(f'<tr><td>{i+1}</td><td>{score(c,branch):.6f}</td><td>{c["reward"]:.6f}</td><td>{"是" if c["sft_selected"] else "否"}</td></tr>')
        blocks.append('</table>')
        source_caption = g['reference_attributes'] if branch == 'audio' else g['source_caption']
        blocks.append('<h3>原始音频参考属性 / 源 caption</h3><pre>' + esc(json.dumps(source_caption, ensure_ascii=False, indent=2)) + '</pre>')
        if branch == 'audio':
            fields = sorted(g['candidates'][0]['attribute_reconstruction']['fields'])
            blocks.append('<h3>每个属性得分</h3><table><tr><th>属性</th><th>参考标签</th>' + ''.join(f'<th>候选{i+1}</th>' for i in range(4)) + '</tr>')
            for field in fields:
                values = [c['attribute_reconstruction']['fields'][field] for c in g['candidates']]
                style = ' class="diff"' if max(values)-min(values) > 1e-9 else ''
                blocks.append(f'<tr{style}><td>{field}</td><td>{esc(get(source_caption,field))}</td>' + ''.join(f'<td>{v:.3f}</td>' for v in values) + '</tr>')
            blocks.append('</table>')
        else:
            blocks.append('<h3>共同 TTS 输入</h3><pre>' + esc(json.dumps(g['request'], ensure_ascii=False, indent=2)) + '</pre>')
        for i, c in enumerate(g['candidates']):
            view = {k: c[k] for k in ('caption', 'generated_attributes', 'reward_components_raw', 'reward_components_normalized') if k in c}
            blocks.append(f'<details><summary>候选 {i+1} 详细记录</summary><pre>{esc(json.dumps(view,ensure_ascii=False,indent=2))}</pre></details>')
        blocks.append('</details></section>')
        sections.append(''.join(blocks))
        selected.append(sample)
    dump(OUT / 'listening_manifest.json', selected)
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V5 两轮训练：同组音频试听</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 20px;color:#182b40;background:#f3f5f8}section{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:22px;margin:24px 0}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px;margin:18px 0}audio{display:block;width:100%;margin:8px 0}.reference{max-width:480px}small{color:#51637a}summary{cursor:pointer;font-weight:600;padding:10px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px;font-size:13px}table{border-collapse:collapse;font-size:14px;width:100%;margin:16px 0}td,th{padding:8px;border:1px solid #cbd5e1;text-align:left;overflow-wrap:anywhere}.diff{background:#fff4db}</style>
<h1>V5 两轮训练：同组音频试听</h1><p>音频原始 WAV 已内嵌，保存此 HTML 后用浏览器打开即可离线播放，不需要服务器或 API。</p>
<p>建议先听原音和四个候选，再展开分数。audio-only 的四条输入 caption 可能不同（含转录文本）；caption-only 才是同一输入的四次 TTS 采样。</p>
<p>这里是训练期间保存的 rollout。第一轮使用第一轮开始时的模型；第二轮使用第一轮更新后的模型，不是第二轮最终 checkpoint 重新生成的音频。参考属性与生成标签均为当时打标结果，未进行人工复核。</p>
<p>这些组按分数跨度挑选用于试听，不能当作随机样本估计总体效果。重建分不包含独立的转录准确率项。</p>'''
    document += ''.join(sections) + '''<script>document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);</script></html>'''
    (OUT / 'listen.html').write_text(document)
    lines = ['# V5 两轮训练：同组重建分差异', '', '统计使用已提交的 round_000、round_001 奖励输出文件；未调用 API 或重新生成音频。', '',
             '## audio-only：Captioner GRPO 的属性重建分', '',
             '只比较具有真实、有效重建分的候选，不将未生成音频的候选按零分混入组内跨度。', '',
             '| 指标 | 第一轮 | 第二轮 |', '|---|---:|---:|']
    a, b = [report['rounds'][str(r)]['audio'] for r in (0, 1)]
    items = [('总组数', a['groups'], b['groups']), ('有效音频候选数 / 736', a['valid_candidates'], b['valid_candidates']),
             ('至少两个有效候选的组数', a['comparable_groups'], b['comparable_groups']),
             ('完整四候选组数', a['valid_count_histogram'].get(4,0), b['valid_count_histogram'].get(4,0)),
             ('有效候选平均重建分', a['candidate_score']['mean'], b['candidate_score']['mean']),
             ('同组最高−最低：平均值', a['span']['mean'], b['span']['mean']),
             ('同组最高−最低：中位数', a['span']['median'], b['span']['median']),
             ('同组第一名−第二名：中位数', a['top2_gap']['median'], b['top2_gap']['median']),
             ('最高−最低 ≤0.01 的组数', a['span_thresholds']['le_0.01'], b['span_thresholds']['le_0.01']),
             ('最高−最低 ≥0.10 的组数', a['span_thresholds']['ge_0.10'], b['span_thresholds']['ge_0.10'])]
    for title, v0, v1 in items:
        fmt = lambda v: f'{v:.4f}' if isinstance(v,float) else str(v)
        lines.append(f'| {title} | {fmt(v0)} | {fmt(v1)} |')
    lines += ['', '组间参考可评分属性数为 12～15；组内分母相同。绝大多数有效候选按 15 个属性计分，因此一个属性从 0 变 1 通常让总分变化约 0.0667。', '',
              '## caption-only：TTS GRPO 的 Captioner 重建 logprob', '',
              '此处是另一种量纲：当前 Captioner 对源 caption 属性值的宏平均 logprob，越大越好；不是 audio-only 的 0～1 属性匹配率，也不是 TTS SFT loss。', '',
              '| 指标 | 第一轮 | 第二轮 |', '|---|---:|---:|']
    for title, key in [('平均重建 logprob', 'candidate_score'), ('同组最高−最低：平均值', 'span')]:
        lines.append(f'| {title} | {report["rounds"]["0"]["caption"][key]["mean"]:.4f} | {report["rounds"]["1"]["caption"][key]["mean"]:.4f} |')
    lines += ['', '两个轮次各 196 组，每组 4 条有效候选。', '', '## 试听', '', '[离线试听页面](listen.html)（所有 WAV 原样内嵌；下载后用浏览器打开）。', '',
              '页面包含两轮 audio-only 各三组（小、中等、大分差），以及 caption-only 同一输入在两轮中的各四次采样。', '',
              '重点样本：第二轮 expressivespeech_266 的分数为 0.4333、0.4933、0.7667、0.2667；其四条中间 caption 的情绪都是 neutral，而打标器将第三条生成音频判断为 angry。它获得高分不能直接证明中间 caption 的情绪描述正确。', '',
              '第二轮 expressivespeech_321 的第一条中间转录变成 “This is an example sentence.”，仍获得 0.5333、为该组最高分。第二轮 expressivespeech_065 的四条音频时长约 10.72、4.16、9.20、10.16 秒，却都在 0.60～0.627 分；这是按属性平均而不单独检查内容的实际局限，不能把分数接近直接解释为听感接近。', '',
              '分差证明当前分数能够区分候选，不能单独证明排序符合人工听感，也不能区分 caption 变化、TTS 随机性和打标噪声。第二轮 rollout 使用第一轮更新后的模型。', '',
              '完整统计：statistics.json；所有组：all_groups.csv；试听原路径与 SHA-256：listening_manifest.json。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    for p in OUT.iterdir():
        if p.is_file():
            p.chmod(0o666)
    print(json.dumps({'report': str(OUT/'report.md'), 'listening_page': str(OUT/'listen.html'),
                      'page_mib': (OUT/'listen.html').stat().st_size / 1024**2,
                      'audio_groups': len(selected), 'audio_clips': sum(len(g['candidates']) + int('reference_audio' in g) for g in selected),
                      'audio_stats': {str(r): report['rounds'][str(r)]['audio'] for r in (0,1)}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
