"""Summarize existing Qwen2.5 evaluations; no inference or training."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
BENCH = OUT.parent.parent
ROOT = BENCH / 'qwen25_v1_v2_caption_eval_runs_20260907_run01'
source = json.loads((ROOT / 'summary_qwen25_v1_v2/results.json').read_text())
rows, details, checks, missing = {}, [], [], []
identities = {}
for entry in source['entries']:
    assert entry['state'] == 'complete'
    suite, label = entry['suite'], entry['trajectory'] + '_' + entry['label']
    folder = Path(entry['result_path'])
    row = rows.setdefault(label, {'version': entry['trajectory'], 'round': entry['round']})
    if suite == 'emotiontalk':
        raw = json.loads((folder / 'metrics_standard_public/scores.json').read_text())
        meta = json.loads((folder / 'run_metadata.json').read_text())['identity']
        predictions = folder / 'predictions.jsonl'
        expected = 7716
        for task, result in entry['result']['tasks'].items():
            assert result['count'] == raw['tasks'][task]['count'] == 1929
            for metric, val in result['metrics'].items():
                original = raw['tasks'][task]['metrics'][metric]
                assert val['status'] == original['status']
                corpus_key = {'bertscore': 'bert_score.f1', 'clapscore': 'clap_sim'}.get(metric, metric)
                assert val['value'] == original.get('corpus', {}).get(corpus_key)
                details.append({'version': entry['trajectory'], 'round': entry['round'],
                                'task': task, 'metric': metric, **val})
                if val['status'] != 'ok':
                    missing.append(f"{label}: {task}/{metric}")
                if task == 'overall':
                    row[metric] = val['value']
    elif suite == 'paraspeechcaps':
        raw = json.loads((folder / 'reports/summary.json').read_text())
        assert raw == entry['result']
        meta = json.loads((folder / 'outputs/run_metadata.json').read_text())['evaluation_identity']
        predictions, expected = folder / 'outputs/field_records.jsonl', 840
        assert raw['samples'] == 140
        row['para_score'] = raw['final_score']
        row['para_parse_rate'] = raw['all_six_fields_parse_rate']
    else:
        raw = json.loads((folder / 'evaluation_summary.json').read_text())
        assert raw == entry['result']
        meta = json.loads((folder / 'run_metadata.json').read_text())['identity']
        predictions, expected = folder / 'predictions.jsonl', 3112
        row['style_accuracy'] = raw['macro_average_accuracy']
        row.update({'style_' + k: v['accuracy'] for k, v in raw['tasks'].items()})
    with predictions.open() as f:
        actual = sum(bool(line.strip()) for line in f)
    assert actual == expected, (label, suite, actual)
    adapter = meta['adapter']
    assert 'Qwen2.5-Omni-3B' in adapter['base_model_name_or_path']
    assert f"round_{entry['round']:03d}/checkpoints/caption_final" in adapter['path']
    branch = 'DualISL_Train_RewardV2' if entry['trajectory'] == 'qwen25_rewardv2' else 'DualISL_Train'
    assert f'/{branch}/runs/' in adapter['path']
    identity = (adapter['path'], adapter['weights_sha256'], adapter['config_sha256'])
    assert identities.setdefault(label, identity) == identity
    checks.append({'checkpoint': label, 'suite': suite, 'rows': actual,
                   'result_path': str(folder), 'adapter_path': adapter['path'],
                   'recorded_weights_sha256': adapter['weights_sha256']})

assert len(checks) == 42 and len(rows) == 14
ordered = sorted(rows.values(), key=lambda r: (r['version'] != 'qwen25_v1', r['round']))
for filename, records in [('rounds.csv', ordered), ('emotiontalk_all_metrics.csv', details)]:
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (OUT / filename).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
(OUT / 'verification.json').write_text(json.dumps({'source': str(ROOT),
    'source_created_utc': source['created_utc'], 'checked_tasks': checks,
    'unavailable_metrics': missing,
    'note': 'Checked raw scores, line counts and recorded adapter identities across suites; did not rehash training weights or rerun inference.'}, ensure_ascii=False, indent=2) + '\n')

def name(r):
    return ('原版' if r['version'] == 'qwen25_v1' else 'RewardV2') + f" r{r['round']}"

def table(items):
    lines = ['| checkpoint | EmotionTalk SPIDEr | EmotionTalk FENSE | ParaSpeechCaps Scheme A | StyleCap 准确率 |',
             '|---|---:|---:|---:|---:|']
    for r in items:
        lines.append(f"| {name(r)} | {r['spider']:.6f} | {r['fense']:.6f} | {r['para_score']:.6f} | {100*r['style_accuracy']:.4f}% |")
    return '\n'.join(lines)

a, b, c = [rows[k] for k in ['qwen25_v1_r4', 'qwen25_rewardv2_r4', 'qwen25_rewardv2_r8']]
report = '''# Qwen2.5-Omni-3B 两版 Captioner 评测汇总

整理日期：2026-09-08。评测汇总生成于 2026-09-07 11:04 UTC（北京时间 19:04）。本次读取并核对现有结果，没有重新训练或推理。

## 版本与完成范围

实际两版是 **原版 DualISL + RewardV2**，已完成 **三套** Captioner benchmark。RewardV3 的 runs 目录目前只有 MiDasheng-7B，不是 Qwen2.5 的第二版。

| 分支 | 正式训练目录（相对 Caption） | 已有 commit 且已评测轮次 | full 任务 |
|---|---|---|---:|
| 原版 | `DualISL_Train/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_20260903_run01` | r0–r4 | 15 |
| RewardV2 | `DualISL_Train_RewardV2/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_reward_v2_20260903_run02` | r0–r8 | 27 |

r0 是第一轮训练后的 checkpoint，不是未训练 base；r4 是第五轮，r8 是第九轮。当前已提交范围仍为上述轮次，不能把计划十轮等同于已完成十轮。未核验远端训练进程状态。

| benchmark | 口径 | 每个 checkpoint 的规模 |
|---|---|---|
| EmotionTalk | standard_public；speaker/style/emotion/overall 四个独立任务 | 1,929 音频 × 4 = 7,716 预测 |
| ParaSpeechCaps | attr6 content-scheme-a-v1，六属性综合分 | 140 音频、840 属性记录 |
| StyleCap / PromptSpeech MCQ | gender/pitch/speaking_speed/volume 四任务宏平均准确率 | 3,112 题，每任务 778 题 |

下表的 EmotionTalk 使用 **overall 任务**，不是四任务平均；SPIDEr、FENSE 和 Para 分值保留原始尺度，Style 转为百分比，均越高越好。

## 相同轮次与最新已评测 checkpoint

'''
report += table([a, b, c]) + '\n\n'
report += (f"同为 r4，RewardV2 − 原版：SPIDEr {b['spider']-a['spider']:+.6f}，FENSE {b['fense']-a['fense']:+.6f}，"
           f"Para {b['para_score']-a['para_score']:+.6f}，Style {100*(b['style_accuracy']-a['style_accuracy']):+.4f} 个百分点。"
           'V2 的 FENSE 较高，另外三项略低，没有全面领先。相同轮次也不保证实际训练更新预算相同。\n\n')
report += (f"各版最新已评测 checkpoint 为原版 r4 与 V2 r8。V2 的 SPIDEr、FENSE、Para 较高，Style 较低；"
           '但 V2 多训练四轮，这个比较不能单独证明 reward 改动有效。\n\n## 全轮次结果\n\n')
report += table(ordered) + '\n\n## 各指标最高分（在本次已评测 checkpoint 内）\n\n'
report += '| 指标 | 原版最佳 | RewardV2 最佳 |\n|---|---|---|\n'
for key, title in [('spider','EmotionTalk SPIDEr'),('fense','EmotionTalk FENSE'),('para_score','ParaSpeechCaps'),('style_accuracy','StyleCap')]:
    best = [max((r for r in ordered if r['version'] == v), key=lambda r:r[key]) for v in ['qwen25_v1','qwen25_rewardv2']]
    cells = [f"{name(r)}：{r[key]:.6f}" if key != 'style_accuracy' else f"{name(r)}：{100*r[key]:.4f}%" for r in best]
    report += f"| {title} | {' | '.join(cells)} |\n"
report += '''
这些是逐指标在测试结果中取最大值，不是预先在验证集选定的统一最佳模型。整体变化幅度较小，未进行两版差值的统计显著性检验，不能据此宣称稳定提升。

## 完整性与解释边界

- 42/42 项 full 的原始评分与既有汇总一致，逐文件行数符合预期；同一 checkpoint 在三个 benchmark 中记录的 adapter 路径、配置哈希和权重哈希一致。本次没有重新计算训练权重哈希。
- ParaSpeechCaps 全部 14 个 checkpoint 的六字段解析率均为 100%。解析成功不等于属性正确。
- EmotionTalk 的 overall 七项指标均可用；BLEU-4 接近零，完整精度见 CSV。所有 14 个 checkpoint 的 style 任务 CLAPScore 不可用，V2 r8 的 speaker CLAPScore 也不可用，错误为文本张量长度不一致。因此 full 完成不等于所有子指标成功。
- 本批结果不含 Qwen2.5 未训练 base，也不含 TTS。不能用 r0 充当 base，或混入旧 Qwen3 分支的 base 结果，判断微调相对 base 的收益。
- 旧 README 的“尚未启动评测”和机制分析中“Qwen2.5 待补评”是历史快照；本报告以已经落盘的 42 项 full 结果为准。

## 数据文件

- [逐轮全部主要指标 CSV](rounds.csv)：EmotionTalk overall 七项、Para 总分/解析率、Style 总分及四子任务。
- [EmotionTalk 全任务指标 CSV](emotiontalk_all_metrics.csv)：保留 status、缺失值和错误原因。
- [核对记录与原始结果路径](verification.json)。
- [原始汇总](../../qwen25_v1_v2_caption_eval_runs_20260907_run01/summary_qwen25_v1_v2/results.json)。
'''
assert all(r['para_parse_rate'] == 1 for r in ordered)
(OUT / 'results.md').write_text(report)
print(table([a,b,c]))
print('Verified:', len(checks), 'full tasks;', len(rows), 'checkpoints; unavailable subtask metrics:', len(missing))
print('Report:', OUT / 'results.md')
