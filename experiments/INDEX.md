# 实验索引

先确定版本，再看配置和结果。训练中的 reward、teacher-forcing loss、独立 benchmark 是不同测量，不能互相替代。

| 实验 | 要回答的问题 | 结果与证据 |
|---|---|---|
| V5 双循环训练 | 属性重建奖励下，两模型能否随迭代改善？ | [实验设置](v5/README.md)、[Captioner](../results/CAPTIONER.md)、[TTS](../results/TTS_DSD.md) |
| V6 四属性 audio-only | 缩小属性空间、移除另一个循环和 anchor 后如何？ | [实验设置](v6/README.md)、[十轮统计](v6/training/rounds.csv) |
| V6 前三轮机制诊断 | 强裁判为何不保证 Captioner 属性能力提升？ | [分析](diagnostics/v6_captioner_diagnosis_round000_002/README.md)、[结构化证据](diagnostics/v6_captioner_diagnosis_round000_002/analysis.json) |
| V5 loss 奖励探测 | TTS teacher-forcing loss 能否区分同组 caption？ | [报告](diagnostics/v5_loss_reward_probe_round001_run01/report.md)、[候选分数](diagnostics/v5_loss_reward_probe_round001_run01/candidates.csv) |
| SpeechBERTScore / cosine | 哪种音频相似度提供不同的候选排序？ | [报告](diagnostics/v5_speech_similarity_round001_run01/report.md)、[分数](diagnostics/v5_speech_similarity_round001_run01/candidates.json) |
| V5 同组音频与 reward | 同组候选差距和转录一致性如何？ | [报告](diagnostics/v5_round000_001_group_rewards/report.md)、[统计](diagnostics/v5_round000_001_group_rewards/statistics.json) |
| V5 实现审计 | 代码是否按预期产生训练数据和奖励？ | [训练审计](diagnostics/v5_training_audit_20260910/REVIEW.md)、[caption-only 检查](diagnostics/v5_caption_only_review/REVIEW.md) |
| 旧 likelihood 重建分 | 分数很低是否就说明奖励无效？ | [分数审计](diagnostics/reconstruction_score_audit_20260909_run01/analysis.md)、[历史 loss](diagnostics/tts_loss_reward_history_20260910/statistics.json) |
| 早期版本对照 | V1/V2/V3 修改机制及训练信号有何区别？ | [版本分析](diagnostics/version_mechanism_analysis_20260907/analysis.md)、[早期 benchmark](diagnostics/completed_caption_rounds_20260907/results.md) |
| 吞吐与格式诊断 | 慢在哪里、格式信号是否影响训练？ | [耗时审计](diagnostics/training_latency_audit_20260909/analysis.md)、[格式审计](diagnostics/format_reward_audit_20260910_run01/analysis.md) |

## 已有结果支持到什么程度

- V6 十轮 Captioner 的 EmotionTalk 指标有小幅变化，ParaSpeechCaps 并未相应改善；不支持“多轮训练必然全面提升”的结论。完整数字见总表。
- V5 loss 探测完成 704 次评分、其中 432 条候选。loss 权重 0.1 / 0.2 时，116 个可比较组没有改变第一名；0.3 时改变 3 组。参考指令与错误性别指令比较为 16:16，不足以仅通过提高权重就确认该信号有效。
- 音频相似度实验比较 432 对音频。不同指标确实产生不同排序；分差更大或与属性 reward 一致，不等于更接近人的听感。该实验没有证明某个指标已改善正式训练。
- V6 中间 caption 与最终音频的离线诊断使用固定模型标签，不是独立人工真值；不能把其 agreement 直接写成人工准确率。

## 记录口径

文件里的 `round_000` 是第 1 轮。第 i 轮采集使用轮初模型，采集指标属于更新前的行为，benchmark 属于该轮提交后的 checkpoint。API 使用量是日志记录量，计费重试和服务商价格会影响账单，历史美元估算不能当当前报价。

历史报告保留当时的发现和限制；不因为文件中出现“当前”“下一步”就把它当今天的状态。来源与版本关系以本索引、各 run 的 commit 和 provenance 为准。
