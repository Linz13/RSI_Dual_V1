# Qwen2.5-Omni-3B 两版 Captioner 评测汇总

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

| checkpoint | EmotionTalk SPIDEr | EmotionTalk FENSE | ParaSpeechCaps Scheme A | StyleCap 准确率 |
|---|---:|---:|---:|---:|
| 原版 r4 | 0.063098 | 0.796566 | 0.425261 | 40.1671% |
| RewardV2 r4 | 0.062943 | 0.804976 | 0.422952 | 39.8458% |
| RewardV2 r8 | 0.063703 | 0.801647 | 0.428751 | 40.0386% |

同为 r4，RewardV2 − 原版：SPIDEr -0.000155，FENSE +0.008410，Para -0.002309，Style -0.3213 个百分点。V2 的 FENSE 较高，另外三项略低，没有全面领先。相同轮次也不保证实际训练更新预算相同。

各版最新已评测 checkpoint 为原版 r4 与 V2 r8。V2 的 SPIDEr、FENSE、Para 较高，Style 较低；但 V2 多训练四轮，这个比较不能单独证明 reward 改动有效。

## 全轮次结果

| checkpoint | EmotionTalk SPIDEr | EmotionTalk FENSE | ParaSpeechCaps Scheme A | StyleCap 准确率 |
|---|---:|---:|---:|---:|
| 原版 r0 | 0.062170 | 0.803498 | 0.424894 | 39.7815% |
| 原版 r1 | 0.062596 | 0.803828 | 0.426409 | 39.9743% |
| 原版 r2 | 0.062966 | 0.798219 | 0.425713 | 40.0386% |
| 原版 r3 | 0.062884 | 0.795914 | 0.422682 | 40.2314% |
| 原版 r4 | 0.063098 | 0.796566 | 0.425261 | 40.1671% |
| RewardV2 r0 | 0.062438 | 0.804565 | 0.425588 | 40.0064% |
| RewardV2 r1 | 0.062223 | 0.804151 | 0.424646 | 39.8458% |
| RewardV2 r2 | 0.062339 | 0.803870 | 0.421700 | 39.9100% |
| RewardV2 r3 | 0.062712 | 0.804345 | 0.419306 | 39.9100% |
| RewardV2 r4 | 0.062943 | 0.804976 | 0.422952 | 39.8458% |
| RewardV2 r5 | 0.062876 | 0.804151 | 0.428088 | 39.8136% |
| RewardV2 r6 | 0.063146 | 0.803034 | 0.425199 | 40.0064% |
| RewardV2 r7 | 0.063275 | 0.802243 | 0.420411 | 40.1028% |
| RewardV2 r8 | 0.063703 | 0.801647 | 0.428751 | 40.0386% |

## 各指标最高分（在本次已评测 checkpoint 内）

| 指标 | 原版最佳 | RewardV2 最佳 |
|---|---|---|
| EmotionTalk SPIDEr | 原版 r4：0.063098 | RewardV2 r8：0.063703 |
| EmotionTalk FENSE | 原版 r1：0.803828 | RewardV2 r4：0.804976 |
| ParaSpeechCaps | 原版 r1：0.426409 | RewardV2 r8：0.428751 |
| StyleCap | 原版 r3：40.2314% | RewardV2 r7：40.1028% |

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
