# V5 主实验

目标：保留两条循环，使用固定属性打标替代 audio-only 中原有 likelihood 重建奖励，同时保留 caption-only TTS GRPO。

- Captioner：MiDashengLM-7B；TTS：Qwen3-TTS VoiceDesign。
- paired / audio-only / caption-only 源记录：39 / 184 / 196；主 run 开启 paired anchor。
- audio-only reward：属性重建 0.9、格式 0.1，不做 ASR 内容门槛。
- 导出源码：`frameworks/v5`，对应 FastResume 修订；第 1 轮是此前 LabelRobust 实现。
- 本快照主 run 有 3 个已提交轮次；前 2 轮有完整收录的 Captioner 与 DSD 评测。

[原始配置](training/resolved_config.yaml) · [轮次统计](training/rounds.csv) · [最终提交记录](training/latest.json)

原 run 目录名中的 `10rounds` 是计划值，不表示此 V5 run 已完成十轮。权重未复制，commit 中保留 checkpoint 身份。V5 迁移改变的代码/配置记录位于仓库 `provenance/v5_migration/`；当前源码快照不应被解释为第 1 轮逐字相同的代码。

DSD 统一比较使用当前 API 重评的 base；旧结果里内嵌的 base 分数可能来自之前服务商，见 [TTS 总表说明](../../results/TTS_DSD.md)。
