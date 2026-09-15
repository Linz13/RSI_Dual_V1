# V5 caption-only / TTS GRPO 核查

读取当前 FastResume 源码、已完成 round_000 / round_001 的 caption rewards 及 resolved_config。未修改训练代码或配置，未运行 GPU 或新增 API 打标。用户新确定的四属性方案（性别、音高、情绪类别、情绪强度）尚未应用到这两轮。

使用当前 candidate_components / semantic_reward 对两轮共 1,568 个候选重新计算，准入标志全部一致，奖励最大绝对误差为 0。

## 实际流程

1. 每轮 196 条 source caption 投影到 P_syn，保留转录、语言和声音属性。
2. 轮初 TTS 对同一 caption 采样 4 条音频，保存 codec token 与 old/ref token logprob。
3. Whisper 计算内容匹配分；audio_health 检查时长、能量、削波等。
4. 轮初 MiDasheng 对固定 source caption 做 teacher-forcing likelihood 评分；不是自由生成 caption 后做属性准确率比较。只平均已知属性值的 token logprob，先字段内平均，再字段间宏平均。当前包含转录、语言；两轮各 608 个候选有 16 字段、176 个有 15 字段。
5. 每条候选各创建最多 3 个属性替换反事实，当前 MiDasheng 和冻结初始 MiDasheng 都对它们评分。反事实选择 key 包含 candidate_id，同组候选并未共用对照题。
6. 重建和反事实两项使用首轮统计固定标准化，再加冻结基模惩罚。
7. 有至少两个有效候选的组计算组内 advantage，TTS 重放实际采样 codec，以 clipped policy objective + KL 更新 LoRA。一轮一个 epoch。主码本和子码本均更新，子码本项权重 0.3；clip 0.2、KL 系数 0.02。
8. 组内最高奖励音频另外与原 source caption 配对，进入 Captioner cycle SFT。TTS cycle SFT 的伪配对来自 audio-only 分支，二者不同。

两条 loop 的采样、评分均在任何模型更新前完成。虽然更新顺序先 Captioner 后 TTS，TTS 使用的仍是轮初 Captioner 产生的奖励；下一轮才使用更新后的两个模型。

## 奖励

设 s 为 source caption 的字段宏平均 logprob，s_j^- 为反事实描述的分数。

```
m = s - log(mean(exp(s_j^-)))
z_s = (s - (-1.593519987483136)) / 0.31549831801364003
z_m = (m - 0.07109962275425263) / 0.09669614897659874
penalty = 0.25 * max(0, -m_base / 0.09669614897659874 - 0.5)
R = 0.5*z_s + 0.5*z_m - penalty
```

m_base 使用冻结初始 Captioner。分数不限制在 [0,1]。冻结奖励模型惩罚与 paired-anchor 监督样本、TTS GRPO 的 KL 参考策略是不同机制。

准入要求轨迹有效、音频路径存在于记录、audio_health 有限且 >=0.5、ASR 分数有限且 >=0、正例及反事实评分可用。这里 semantic_valid 不代表属性正确或转录正确。ASR 分数不作为加权奖励项，阈值为 0，不能拦截内容错配。转录仍在 likelihood 宏平均中，有间接影响。

## 已完成两轮的离线统计

| 统计 | round_000 | round_001 |
|---|---:|---:|
| 组数 | 196 | 196 |
| 候选 / 有效候选 / GRPO 候选 | 784 / 784 / 784 | 784 / 784 / 784 |
| 同组反事实集合不一致的组数 | 196 | 196 |
| 重建项组内加权跨度均值 | 0.228927 | 0.198044 |
| 反事实项组内加权跨度均值 | 0.913445 | 0.920487 |
| 最终 top1 与纯重建 logprob top1 不同 | 119 / 196 | 125 / 196 |
| 反事实聚合分高于目标描述的候选 | 175 | 169 |
| 冻结基模惩罚非零的候选 | 87 | 85 |
| 去掉冻结基模惩罚后 top1 改变的组数 | 0 | 0 |
| ASR score <0.8 的候选 | 183 | 200 |
| 被选入 Captioner SFT 且 ASR score <0.8 | 45 / 196 | 47 / 196 |

跨度是同组最大值减最小值，先乘 0.5 再跨组平均。反事实/重建跨度均值比约 3.99 / 4.65；不等于方差贡献率或准确率。top1 改变不能直接证明变好或变差，惩罚不改变 top1 也不等于对 advantage 完全无影响。

ASR score 为 1−截断至 [0,1] 的错误率；英语按词，中文按字符（中文当前未统一去标点）。因此 <0.8 是待核查信号，不是人工确认的错误标签。

## 具体例子

round_001 的 `random440_expressivespeech_164`：候选 0 的反事实包括语速 slow、性别 female、音量 medium；候选 2 则包括性别 female、音高 low、音色 nasal。不同音频的 margin 同时受不同对照题影响。

`random440_deear_060` 的 source caption 指定 language=Chinese、accent=Mandarin，transcript 却为 `I didn't put it anywhere Shut yet.`。候选 3 的 ASR 是 `我还没有放到任何地方的地方`、ASR score=0，仍被选入 Captioner SFT。这里只确认输入/ASR记录冲突，未人工试听确认真实内容；不把这个例子归结为 TTS 单独失败。

## 判断与建议

- 优先问题是奖励的可比性和数据准入，现有检查没有单凭上述现象证明 GRPO 优化器实现错误。
- 若保留反事实，同组所有音频应使用相同反事实字段、替换值和评分口径；四属性方案可以固定一个共享的小对照集合。
- Captioner likelihood 不等于感知准确率，可能受文本先验、正确前缀和字段间关联影响。需要验证它对四属性音频的排序是否符合独立评审；不能由自由生成准确率低直接推导 likelihood 完全无效。
- 恢复两个分支的内容检查，并清理源 caption 的语言/转录/指令冲突；SFT top1 当前没有绝对质量准入，低质量组也能产生伪配对。
- 四属性变更应同步调整提示、TTS 指令、likelihood 字段掩码、反事实、SFT 目标和校准。旧 15/16 字段的标准化统计不能直接沿用。
- 外部强打标可先用作 likelihood 排序的独立检查；完全替换 caption-only 奖励将改变 TTS 对可训练 Captioner 的依赖，应与论文目标一起决定。

源码入口：`dual_isl_train/orchestrator_v4.py::_collect_caption_only`、`orchestrator_v5.py::_score_round`、`workers/qwen3_captioner.py::synth_value_macro_logprob`（MiDasheng 继承）、`rewards.py::semantic_reward/score_groups`、`workers/qwen_voice_design.py::grpo_update`。

数据入口：`DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01/round_00{0,1}/rewards/round_00{0,1}_caption_rewards.output.jsonl`。
