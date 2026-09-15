#!/usr/bin/env python3
"""Rebuild portable tables using only result files included in this repository."""
import csv
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
def save_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    buf=io.StringIO();writer=csv.DictWriter(buf,fields);writer.writeheader();writer.writerows(rows)
    (ROOT/path).write_text(buf.getvalue())
def md_table(fields,rows):
    def fmt(x):return '—' if x is None else f'{x:.6f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']+
                     ['| '+' | '.join(fmt(x) for x in row)+' |' for row in rows])

def main():
    rows={}
    source='experiments/diagnostics/completed_caption_rounds_20260907/rounds.csv'
    for row in csv.DictReader((ROOT/source).open(encoding='utf-8-sig')):
        if row['trajectory']=='midasheng_v1' and row['round']=='base':
            rows[('base',0)]={'version':'base','training_round':0,'round_index':None,
                'et_spider':float(row['spider']),'et_fense':float(row['fense']),
                'paraspeechcaps':float(row['paraspeechcaps']),'stylecap':float(row['stylecap']),
                'source':source}
    assert ('base',0) in rows
    paths=['midasheng_v4_all_v5_round1_caption_4gpu_20260910_run01','midasheng_v5_round001_caption_3gpu_run01','v6_caption_all_rounds_run01']
    for name in paths:
        source=f'results/benchmarks/{name}/summary.json'
        for e in read(source)['entries']:
            if not e.get('complete',e.get('state')=='complete'):continue
            version='v6' if name.startswith('v6') else 'v5' if 'v5' in e.get('candidate','') else 'v4'
            n=e['training_round'];key=(version,n)
            row=rows.setdefault(key,{'version':version,'training_round':n,'round_index':e['round_index'],'source':source})
            result=e['result'];suite=e['suite'].lower()
            if suite=='emotiontalk':
                metrics=result['tasks']['overall']['metrics']
                for metric in ['spider','fense']:
                    row['et_'+metric]=metrics[metric]['value'] if metrics[metric]['status']=='ok' else None
            elif suite=='paraspeechcaps':row['paraspeechcaps']=result['final_score']
            elif suite=='stylecap':row['stylecap']=result['macro_average_accuracy']
    caption=sorted(rows.values(),key=lambda r:({'base':0,'v4':1,'v5':2,'v6':3}[r['version']],r['training_round']))
    save_csv('results/captioner_summary.csv',caption)
    table=md_table(['版本','训练轮次','ET SPIDEr','ET FENSE','ParaSpeechCaps','StyleCap'],
                   [[r['version'],r['training_round'],*[r.get(k) for k in ['et_spider','et_fense','paraspeechcaps','stylecap']]] for r in caption])
    (ROOT/'results/CAPTIONER.md').write_text('# Captioner benchmark\n\n'+table+'\n\n'
        '以上为原始分数，越高越好。EmotionTalk 取 overall；ParaSpeechCaps 为项目采用的六属性 content scheme A；StyleCap 为选择题宏平均准确率。\n\n'
        '训练轮次从 1 开始，代码中的 round_000 对应第 1 轮。base 表示未训练模型。V5 这里只汇总已有完整评测的两轮，不代表仅训练了两轮。V4 保留已完成评测轮次。\n\n'
        '来源逐行保存在 [captioner_summary.csv](captioner_summary.csv)，原始汇总保存在 [benchmarks](benchmarks/)。'
        '各指标口径不同，不跨列比较大小；小幅差异不能直接解释为统计显著提升。缺失指标不记为零。\n')
    source='results/tts_dsd/base_current_api.json';base=read(source)
    tts=[{'version':'base','training_round':0,'zh':base['metrics']['zh']['DSD']['percentage'],
          'en':base['metrics']['en']['DSD']['percentage'],'mean':base['bilingual_macro_average'],
          'scored':base['scored'],'source':source}]
    for i in (0,1):
        source=f'results/tts_dsd/v5_round{i:03d}.json';d=read(source)
        assert d['complete'] and d['scored']==2000
        tts.append({'version':'v5','training_round':i+1,'zh':d['metrics']['zh']['percentage'],
                    'en':d['metrics']['en']['percentage'],'mean':d['bilingual_percentage'],'scored':d['scored'],'source':source})
    for name in ['v6_round000_002_benchmarks_run01','v6_round003_benchmarks_run01']:
        source=f'results/benchmarks/{name}/summary.json'
        for e in read(source)['entries']:
            if e['suite'].lower()!='dsd' or not e['complete']:continue
            d=e['result'];assert d['complete'] and d['scored']==2000
            tts.append({'version':'v6','training_round':e['round']+1,'zh':d['metrics']['zh']['percentage'],
                        'en':d['metrics']['en']['percentage'],'mean':d['bilingual_percentage'],'scored':d['scored'],'source':source})
    save_csv('results/tts_dsd_summary.csv',tts)
    table=md_table(['版本','训练轮次','DSD 中文 (%)','DSD 英文 (%)','双语平均 (%)','已评分'],
                   [[r[k] for k in ['version','training_round','zh','en','mean','scored']] for r in tts])
    (ROOT/'results/TTS_DSD.md').write_text('# TTS · InstructTTSEval DSD\n\n'+table+'\n\n'
        '每轮中英文各 1,000 条，共 2,000 条。此表采用 Gemini-2.5-Pro audio API 评分，并用相同服务重新评分的 base（80.9%）作参照。\n\n'
        '历史 V5 JSON 中的 base_bilingual_percentage=81.65 是旧参照，保留原文件作为来源，但不用于这张表。'
        '本地评测的模型/接口配置与官方原始 preview 模型协议不完全相同，不宣称这是官方榜单成绩。\n\n'
        'V6 第 5–10 轮没有收录 DSD 评测，不补零、不推测。来源见 [tts_dsd_summary.csv](tts_dsd_summary.csv)。\n')
    print(json.dumps({'captioner_rows':len(caption),'tts_rows':len(tts)}))

if __name__=='__main__':main()
