# V6 四属性 audio-only 实验

目标：只训练性别、音高、情绪类别、情绪强度与转录这组简化接口，检查单循环训练的效果。

- 从 Captioner/TTS base 开始，不接 V5 adapter。
- 源池 419 条，过滤后 411 条音频；group=8；十轮。
- 不使用 paired anchor、caption-only、Captioner SFT 或 TTS GRPO。
- 重建音频内容错误率 ≤10% 后才做四属性评分；总 reward 0.9 重建 + 0.1 格式。
- TTS SFT 从内容合格候选中选属性 top1，没有额外属性分门槛。

[原始配置](training/resolved_config.yaml) · [数据统计](data_report.json) · [十轮训练统计](training/rounds.csv) · [最终提交记录](training/latest.json)

已完成 10 轮，Captioner 共 30 项 benchmark（每轮三项）。TTS DSD 收录第 1–4 轮。完整结果见 [Captioner](../../results/CAPTIONER.md) 和 [TTS](../../results/TTS_DSD.md)。

每轮 `summary.json` 记录格式通过、内容合格、属性分、GRPO 可用组、SFT 选择数和批量回退。首轮经历 API 重试/恢复，`elapsed_seconds_this_invocation` 仅统计该次调用，不等于完整一轮墙钟耗时。

`examples/` 每轮保留前两组 Captioner rollout 文字样例，用于说明数据结构；不是抽样评测，也不含可播放音频。解释属性收益与错误归因时先读 [前三轮诊断](../diagnostics/v6_captioner_diagnosis_round000_002/README.md)。
