# TTS 主表来源核验（2026-09-09）

本文件仅核验已有结果；未启动模型、GPU、API 评测，也未修改任何历史结果。表中轮次指完成的训练轮数，代码 `r0/r1/r2` 分别对应第 1/2/3 轮。所有分支的生成模型均为 Qwen3-TTS-12Hz-1.7B-VoiceDesign，分支名称表示联合训练使用的 Captioner。

## EmergentTTS-Eval：推荐进入当前主表

| Captioner 分支 / checkpoint | 完成轮数 | WER (%) ↓ | WVMOS ↑ | Win Rate (%) ↑ | 本地 / judge 有效样本 |
|---|---:|---:|---:|---:|---|
| 共享 TTS Base | 0 | 12.6751994953 | 4.3030918306 | 45.8662613982 | 1,645 / 1,645 |
| Qwen3-Omni 原版 r0 | 1 | 12.5556372836 | 4.3205159810 | 45.9574468085 | 1,645 / 1,645 |
| Qwen3-Omni 原版 r1 | 2 | 12.3292920822 | 4.3210776264 | 48.1762917933 | 1,645 / 1,645 |
| Qwen3-Omni 原版 r2 | 3 | 14.8947328909 | 4.3144862392 | 46.2310030395 | 1,645 / 1,645 |
| MiDasheng-7B 原版 r0 | 1 | 13.8008899342 | 4.3159956692 | 46.2917933131 | 1,645 / 1,645 |
| MiDasheng-7B 原版 r1 | 2 | 12.8239560082 | 4.2815807487 | 44.1337386018 | 1,645 / 1,645 |
| MiDasheng-7B 原版 r2 | 3 | 14.0909968201 | 4.2783630261 | — | 1,645 / 1,643 |

除 MiDasheng r2 外的六行均读取真实最终 `evaluation-metrics.json`，同目录 `staged_completion.json` 均为 `status=completed`、`selected_samples=1645`；本次重新计算的 metrics 文件 SHA-256 与 completion 中所记值全部一致。六行 `eval/judger_parsing_failed_count` 均为 0，`skipped_samples` 均为 0。

### 精确文件定位

以下所有相对路径的根目录为：

`/data/L202500147/Caption/benchmark/EmergentTTS-Eval-public/qwen3_voice_design/runs/`

最终 metrics 的文件名统一为 `emergent-tts-eval_strong-prompting_evaluation-metrics.json`；completion 为同目录 `staged_completion.json`。

| 表中行 | 最终目录（相对于上述根目录） |
|---|---|
| Base | `qwen3_tts_vd_strong/staged_evaluation/final/` |
| Qwen3 r0 | `eval_qwen8_round0_full_20260830_run01_paid_evaluation/final/` |
| Qwen3 r1 | `eval_qwen8_round1_full_20260830_run01_paid_evaluation/final/` |
| Qwen3 r2 | `eval_qwen8_round2_full_20260831_run01_paid_evaluation/final/` |
| MiDasheng r0 | `eval_midasheng4_round0_full_20260830_run01_paid_evaluation/final/` |
| MiDasheng r1 | `eval_midasheng4_round1_full_20260831_run01_paid_evaluation/final/` |

Base 迁移后的 completion 内仍记旧服务器路径；本次使用上述当前共享文件系统中的真实文件核验哈希。

MiDasheng r2 没有最终 metrics 和 completion。本地数据来自：

`eval_midasheng4_round2_full_20260831_run01/staged_evaluation/local_metrics.jsonl`

该文件恰有 1,645 条、1,645 个唯一 ID，全部 `status=success`。上表 WER 和 WVMOS 分别为其 `wer` 和 `mos_score` 的算术平均，与当前正式汇总代码的平均方式一致。它们是完整本地评测的离线汇总，不是完整付费评测结束后的 final 产物。

Judge 数据来自：

`eval_midasheng4_round2_full_20260831_run01_paid_evaluation/judge_results.jsonl`

该文件有 1,643 条成功、1,643 个唯一 ID；按当前正式胜率公式离线计算为 43.8222763238%。因为缺 2 条且无最终完成产物，当前主表建议写 `—`。若以后选择报告这一部分结果，应只给 Win Rate 加 `1,643/1,645` 注，不能说该行的 WER/WVMOS 也只有 1,643 条。

### 协议及表注

- Benchmark 为 EmergentTTS-Eval strong prompting，目标样本数为 1,645。
- WER 使用 `openai/whisper-large-v3`，先计算每条 `(edit distance / reference length) × 100`，再对样本取算术平均；它不是全部词级计数合并的 corpus WER。来源：`/data/L202500147/Caption/benchmark/EmergentTTS-Eval-public/inference.py:109` 及 `qwen3_voice_design/staged_local_metrics.py:55`。
- WVMOS 是 Wav2Vec2MOS 自动预测分数，不能写成人工 MOS。来源：`qwen3_voice_design/staged_local_metrics.py:40`、`:99`、`:114`。
- 全部被核验的 judge metadata 均记录 `judge_model=gemini-2.5-pro`、`seed=42`、`selected_samples=1645`。
- Win Rate 的对手是 benchmark 官方 `gpt-4o-mini-tts-alloy` 基线音频，**不是上表中的 Qwen3-TTS Base**。候选与基线顺序随机化，胜/平/负计 1/0.5/0，再取平均并乘 100。来源：`/data/L202500147/Caption/benchmark/EmergentTTS-Eval-public/README.md:18`、`inference.py:134`；各 run 的 judge metadata 指向同一官方 `data/baseline_audios`。
- 评测入口 `run_round0_evaluations.sh` 仅配置 Qwen3 与 MiDasheng 的 r0–r2。当前权威 EmergentTTS `runs/` 中未发现 r3/r4/r9/r12、Qwen2.5、RewardV2/V3/V4 的后续完整结果。后续 Captioner benchmark 目录及 reports 中也未定位到对应 EmergentTTS 最终文件。因此第 5/10/13 轮等目前留空，不能拼接早期轮次 TTS 分数。
- Qwen3 r1 的这三项均优于共享 Base，但 r2 的 WER 回落；MiDasheng 分支没有同步稳定改善。表格支持早期 TTS 效果观察，不能据此宣称两侧稳定共同提升。

## 可选补充：现有 InstructTTSEval 摘要

以下仅只读检查现有七份 `summary.json` 的完成度与汇总字段，未进一步做逐条一致性或哈希核验，不纳入本次默认主表。所有 summary 均标记 `official=false`；不可写成官方原协议结果。

根目录：`/data/L202500147/Caption/benchmark/InstructTTSEval-public/qwen3_voice_design/runs/`。

每个分支文件为 `<branch>/full_bilingual_seed42/evaluation_gemini_2_5_pro/summary.json`。

| branch | complete | scored / expected | 双语 macro (%) |
|---|---|---:|---:|
| `base` | false | 5,998 / 6,000 | 75.5899 |
| `qwen8_r0` | true | 6,000 / 6,000 | 75.0000 |
| `qwen8_r1` | false | 5,999 / 6,000 | 75.1630 |
| `qwen8_r2` | false | 3,476 / 6,000 | 72.9283（覆盖不足，不建议横向比较） |
| `midasheng4_r0` | false | 0 / 6,000 | — |
| `midasheng4_r1` | false | 0 / 6,000 | — |
| `midasheng4_r2` | false | 0 / 6,000 | — |

摘要采用 `successful_judge_results_only`，所以不同缺失范围会改变分母和测试覆盖；目录名称中的 `full` 不代表评测完成。Base 和 Qwen3 r1 接近完整，但现有分数也未给出高于 Base 的总分。因此没有必要为了扩大主表添加这组尚未整齐完成的列。
