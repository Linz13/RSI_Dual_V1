# 训练框架

## V5：完整双循环

每轮开始时固定当前 Captioner 和 TTS，先完成两个方向的候选生成与评分，再依次进行 Captioner GRPO、Captioner SFT、TTS GRPO、TTS SFT，提交两侧 checkpoint。

**audio-only：**真实音频 → Captioner 多个候选 caption → TTS 重建音频 → 固定打标器提取属性 → 与原音频参考属性比较。Captioner 的 reward 为 `0.9 × 属性重建分 + 0.1 × 格式分`。这一分支停用旧 likelihood / 冻结基模反事实语义奖励，也没有新增中间 caption 与参考标签的直接正确性奖励。

Caption 允许安全规范化和部分字段可用；TTS 只接收可用条件。原参考属性 unknown 不进入分母，生成属性 unknown 计零。同组按原参考固定比较字段。枚举精确匹配，非语言发声集合 F1，强调内容基于文本对齐，韵律/停顿由固定 GPT-5.5 文本评审给 0/0.5/1 分。V5 此分支不做 ASR 内容门槛。

按重建分选有效 top1，其 caption 作为条件、原始真实音频作为 GT，训练 TTS SFT；不拿重建音频作为这个 SFT 的 GT。

**caption-only：**原始 caption → TTS 采样音频 → 轮初 Captioner 对目标 caption 的概率/重建评分，沿用 V4 这一方向的奖励组合与校准。它不是 V6，也不是后来讨论过的“四属性生成 caption 后交给强裁判”的新方案。奖励用于 TTS GRPO；选中合成音频和原 caption 构成 Captioner SFT。

V5 主 run 默认保留 paired anchor，数据源为 paired 39、audio-only 184、caption-only 196。Anchor Captioner 只保留合成属性空间中的 caption 目标，每轮 39 条；TTS 39 条。数据角色记录不等同于独立测试集。

V5 源码导出自 FastResume。该修订将 TTS rollout batch 设为 1，避免先尝试批量失败再串行生成；audio-only synthesis 可多卡分配。它没有把 GRPO / SFT 改成其他优化算法。第 1 轮结果来自之前的 LabelRobust，实现对应关系见 `provenance/v5_migration/`。

## V6：四属性 audio-only

仅保留真实音频 → Captioner → TTS 重建音频。从两个 base 开始；每轮只做 Captioner GRPO 和 TTS SFT，不执行 caption-only、TTS GRPO、Captioner SFT 或 paired anchor。

Caption 是扁平五字段 JSON：`transcript`、`gender`、`pitch_level`、`emotion`、`emotion_intensity`。中间 caption 只做格式、类型和枚举处理，不与原参考答案比正确性。TTS 使用这个 caption 自己生成的 transcript 和可用属性，language 为 Auto，不传 reference audio。

重建音频先由 Qwen3-ASR 转录，与原音频可靠转录比较：确定性规范化后的中文 CER、英文 WER 或混合 token 编辑率 ≤ 0.10 才通过。内容不通过则重建分为 0，不再请求属性 API。

| 参考/重建属性 | 固定打标模型 |
|---|---|
| 性别、情绪强度 | Qwen3.5-Omni-Plus |
| 音高 | Gemini-3.1-Pro-Preview |
| 情绪类别 | emotion2vec-plus-large，本地 |

四属性等权精确匹配。原参考 unknown 排除，重建 unknown 计 0；同组分母不随候选改动。相邻音高或强度等级不给部分分。

格式不可解析时 `F=0`，可解析时 `F=0.5+0.5×合法字段数/5`；unknown 是合法格式，不代表事实正确。

`R = 0.9 × R_reconstruction + 0.1 × F`。同组奖励相同则跳过 Captioner GRPO。TTS SFT 在内容合格候选中按属性重建分选 top1，同分优先 ASR 错误率低者，再按候选序号；不额外设属性分 ≥0.75 等门槛。

源池 419 条，时长过滤后 411 条；十轮、group=8。配置中 Captioner/TTS 推理 batch=8 是请求值；是否真批量执行要看每轮 `caption_actual_batch_counts` 和 fallback 统计。首轮 Captioner 实际有串行回退，不能仅凭配置估算吞吐。

API 扩展的当前默认为 Qwen 2 worker / 30 RPM、Gemini 64 worker，带冷却与持续重试；不把传输失败填成 unknown 或零奖励。冻结配置中的共享 worker=4 与扩展实际并发要区分。

## 两个模型的 loss

- Captioner SFT：对目标 caption 文本 token 做 teacher forcing 的交叉熵 / 负对数似然。V5 有此阶段，V6 没有。
- TTS SFT：将原始 GT 音频编码为离散 codec token；条件为选中的 caption/transcript，teacher forcing 预测 GT token。现有实现为主码本 NLL 加 `0.3 ×` 子码本 NLL，具体掩码与归一化见 worker。
- GRPO：对采样轨迹计算策略概率比、组内相对优势、裁剪目标及 KL 项。Captioner 的轨迹是文本 token，TTS 的轨迹是音频 codec token。GRPO 不是普通 SFT 交叉熵。

TTS SFT 不直接计算两个 WAV 的波形相似度。loss 奖励和 SpeechBERTScore/cosine 是另做的诊断实验，没有据此宣称已加入这里的 V5/V6 正式 reward。
