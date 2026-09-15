# MiDasheng 原版、RewardV2、RewardV3 效果差异分析

日期：2026-09-07。基于交接文档、三个框架 README/实现、实际训练日志、V2/V3 summary/calibration 和逐样本 benchmark 输出。没有启动训练或 GPU 评测，没有修改训练代码、阈值、checkpoint 或权限。此前暂缓的消融实验仍未执行。

## 结论与边界

**在这次 MiDasheng-7B 的三套 Captioner 评测中，原版整体领先；V3 的准入改进已在训练过程指标上生效，但尚未转化为整体更好的 benchmark。**

不能据此认定所有模型上原版框架都最好，也不能把结果解释成后续改动全部无效。Qwen3 原版此前基本持平，Qwen2.5 两分支仍待补评；此处没有评估三版 TTS 最终能力。原版与 V2/V3 也不是只改变一个 reward 项的等预算对照。

最有证据的解释是：V2 的严格准入和置信 gate 大幅削减了循环语义监督；V3 修复了其中的准入瓶颈，却保留了几乎关闭 cycle-SFT 的 gate，同时新语义 reward 的排序能力尚不足以支撑预期的高置信选择。原版则持续接收更多伪配对并执行更多 SFT 更新。**这是有日志支撑的机制解释，不是已经通过消融分离出的因果贡献。**

## 1. 先做相同轮数比较

固定比较 r9，即三版都训练完十轮，不以原版 r12 对比 V3 r9。SPIDEr/FENSE 使用 EmotionTalk overall 原始分值，Para 为 Scheme A 综合分，Style 为宏平均准确率。三个分支共享同一个 MiDasheng base，指标均越高越好。

| 模型 | SPIDEr | FENSE | ParaSpeechCaps | StyleCap |
|---|---:|---:|---:|---:|
| base | 0.053341 | 0.871648 | 0.233767 | 0.404563 |
| 原版 r9 | 0.059113 | 0.883743 | 0.312299 | 0.412918 |
| V2 r9 | 0.055701 | 0.876766 | 0.239420 | 0.407455 |
| V3 r9 | 0.054345 | 0.872371 | 0.240994 | 0.400707 |

原版优势不只来自多训练三轮。V3 相比 V2 也不是每项都更差：Para 稍高，另外三项较低；不能以微小的 Para 差值推断确定改善。三版 r9 的 base/model、backend、attention 和评测 protocol 身份一致；已完成表格此前核验了 adapter 对应关系。

为区分测试样本波动，本次对已有预测做 20,000 次成对 bootstrap，随机种子 20260907，没有重新推理：

| r9 差值 | Para 点估计 | Para 95% 区间 | Style 点估计 | Style 95% 区间 |
|---|---:|---|---:|---|
| 原版 − base | +0.07853 | [0.06193, 0.09565] | +0.00835 | [-0.01653, 0.03285] |
| V2 − base | +0.00565 | [-0.00125, 0.01294] | +0.00289 | [-0.00285, 0.00842] |
| V3 − base | +0.00723 | [-0.00140, 0.01615] | -0.00386 | [-0.00904, 0.00098] |
| 原版 − V3 | +0.07130 | [0.05407, 0.08938] | +0.01221 | [-0.00984, 0.03342] |
| V3 − V2 | +0.00157 | [-0.00466, 0.00790] | -0.00675 | [-0.01086, -0.00226] |

Para 按 140 个样本成对重采样；Style 按 38 个 speaker 整组重采样，保留一个 speaker 内的四任务和多音频相关性。区间只反映这些固定 checkpoint 在当前测试集重采样下的不确定性，不包含训练随机种子方差。Para 未进一步按 speaker/source 聚类；没有进行多重比较校正或 EmotionTalk 复合指标重算。原版在 Para 的优势较稳健，不能把所有小幅变化都视作同等可靠的能力结论。

## 2. V3 的改动确实起效了

以下来自 V2/V3 十轮真实 summary，非历史文本模拟：

| Captioner 训练过程指标，r0–r9 合计 | V2 | V3 |
|---|---:|---:|
| 满足代码完整语义条件的候选 / 7,360 | 86 | 1,646 |
| 使用语义 GRPO 的组 / 1,840 | 1 | 431 |
| schema curriculum 组 / 1,840 | 1,839 | 1,409 |
| 语义组中非零 advantage 组 | 旧 summary 未记录该计数 | 431 |
| round-0 audio-only z-score 的标定候选数 | 2 | 136 |

V3 语义组占比为 23.42%，V2 为 0.054%。V3 r0 原始 schema-valid 为 96/736，对应 V2 r0 仅 2/736；V3 r9 原始有效为 147/736、安全准入为 203/736。已确认实际 rollout prompt 含完整 JSON 示例。

因此，说“V3 只改了代码但训练根本没用上”是不符合日志的。安全规范化和新 prompt 的组合明显增加了候选覆盖，并产生了非零语义 advantage；但两项改动同时存在，不能分离各自贡献。raw-valid 不等于属性判断正确，semantic-valid 也仅表示通过代码准入和评分条件，不是人工语义正确率。

标定从 2 个候选增到 136 个也是实际改善。V2 用两个候选估计均值/标准差并冻结十轮，标尺容易受这两个样本影响。V3 改善了样本数量，但未自动证明标尺可靠或 reward 排序正确。

## 3. 原版实际接受的训练信号多得多

原版部分 summary/manifest 受共享存储 0600 权限限制，本次从可读的所有 rank 的 `caption_final/training_metrics/rank_*/metrics.jsonl` 统计非 padding 的真实 SFT 记录；按 epoch/step 检查无重复记录，并识别 `anchor::` 前缀。V2/V3 用相同方法统计，再与 summary 的 cycle 数量交叉核对。

| 前十轮 r0–r9 | 原版 | V2 | V3 |
|---|---:|---:|---:|
| Captioner paired-anchor SFT 记录 | 780 | 780 | 780 |
| Captioner cycle-SFT 记录 | 1,960 | 94 | 109 |
| Captioner SFT 总记录 | 2,740 | 874 | 889 |
| Captioner SFT optimizer 更新步 | 452 | 112 | 114 |
| Captioner GRPO 参与组 | 1,432（原混合 reward） | 1,840（仅 1 组语义） | 1,840（431 组语义） |

这些是训练记录暴露次数，不是独立数据数量：paired 是同一批 39 条音频，每轮两个目标；cycle 是固定 caption-only 池每轮重新生成音频。

原版每轮向 Captioner 提供 196 条循环伪配对，加 78 条 paired anchors，持续十轮。V2/V3 r0–r8 每轮都只有 78 条 paired anchors，直到 r9 才分别增加 94/109 条循环伪配对。

“同为 GRPO 1 epoch + SFT 1 epoch”因此不代表优化预算相同：一个版本每轮 274 条 SFT，另两个版本前九轮每轮只有 78 条。原版最初三轮 world size=4，每轮 69 个 SFT 更新步；随后 world size=8，每轮 35 步。V2/V3 从首轮起 world size=8，前九轮每轮仅 10 步，最后为 22/24 步。当前 worker 每个同步 batch 执行一次 optimizer.step；4 卡与 8 卡本身也构成有效 batch 与优化路径差异。

原版 1,432 组使用的是“output quality + reconstruction”混合奖励，不能标作与 V3 严格语义组完全同口径的正确标签组。其宽松规范化可能丢失信息，更多伪配对也不代表每条都准确。但原版更多数据和更新是事实，足以反驳“只改奖励公式，其余训练量完全一致”的解释。

## 4. V3 为什么没恢复循环 SFT

V3 只改变安全准入和 prompt，没有改变每轮 confidence gate。实际十轮如下：

| 轮次 | Captioner 语义 GRPO 组 / 184 | Captioner cycle 配对 | TTS cycle 配对 |
|---|---:|---:|---:|
| r0 | 34 | 0 | 0 |
| r1 | 32 | 0 | 0 |
| r2 | 44 | 0 | 0 |
| r3 | 35 | 0 | 0 |
| r4 | 40 | 0 | 0 |
| r5 | 43 | 0 | 0 |
| r6 | 51 | 0 | 0 |
| r7 | 48 | 0 | 0 |
| r8 | 49 | 0 | 0 |
| r9 | 55 | 109 | 0 |

注意方向：audio-only 的 `sft_selected` 进入 TTS，caption-only 的 `sft_selected` 进入 Captioner，不能看反。

- audio-only 十轮均 `no_high_precision_operating_point`，阈值是 null。即使候选本身已经合法、甚至是本组最高 reward，也因 `threshold_not_calibrated` 无法进入 TTS cycle-SFT。
- caption-only 前九轮同样没有可用阈值，直到 r9 才校准成功，接收 109 条进入 Captioner。
- V3 仍有 76.58% 的 Captioner 组使用格式奖励；格式奖励主要学习可解析性和 schema 完整性，不直接区分同为合法 JSON 的正确/错误情绪。
- TTS 每轮 196 个语义 GRPO 组仍在更新，因此不能说整个系统没有语义 RL；只是循环监督通路高度不对称。Captioner 为 TTS 打分的过程不会反向更新 Captioner，它只有在自己的 GRPO/SFT 阶段才得到更新。

## 5. 门控失败背后：奖励区分度不足

本次直接重算每轮校准输出的 AUROC：对 39 个 paired 正例和 39 个构造负例，计算正例分数高于负例的比例，平分计 0.5。另计算在 recall≥0.25 时，所有现有分数阈值可达到的最高经验 precision。

| V3 轮次 | audio-only 综合分 AUROC | audio-only 最高 precision | caption-only 综合分 AUROC | caption-only 最高 precision |
|---|---:|---:|---:|---:|
| r0 | 0.552 | 0.565 | 0.798 | 0.826 |
| r1 | 0.574 | 0.700 | 0.765 | 0.765 |
| r2 | 0.599 | 0.667 | 0.784 | 0.846 |
| r3 | 0.571 | 0.625 | 0.751 | 0.778 |
| r4 | 0.468 | 0.527 | 0.814 | 0.867 |
| r5 | 0.554 | 0.611 | 0.762 | 0.778 |
| r6 | 0.552 | 0.636 | 0.822 | 0.867 |
| r7 | 0.663 | 0.750 | 0.727 | 0.769 |
| r8 | 0.627 | 0.769 | 0.774 | 0.826 |
| r9 | 0.572 | 0.640 | 0.897 | 1.000 |

gate 要求 precision≥0.90 且 recall≥0.25，因此上表直接解释了 19/20 个方向-轮次为何无法校准。这不只是“有很多好数据被一个略高阈值挡住”：现有评分和负例定义在这批数据上根本不能满足选取要求。r9 caption-only 的 1.000 是另一可用阈值的最高 precision；实际代码优先最大 recall，选的是 precision=0.92、recall=23/39，对应 threshold=0.678728，二者并不矛盾。

AUROC 约 0.5 表示在该批校准样本上排序区分度很弱，不等于模型所有语义能力随机，也不等于线上组内四候选排序的精确准确率。校准正负例复用了 paired 训练样本，数量小；每轮更换部分反事实，故曲线不是固定测试集上的学习曲线。尚需独立标注数据检验。

代码还存在一个值得单独诊断的口径差异：校准负例只用原真值作唯一对照，正例及线上候选用最多三个替代项的 LogMeanExp 作对照。这可能影响统一绝对阈值的可迁移性，当前数据不能单独量化其贡献。

## 6. 为什么增加反事实 reward 仍可能没有涨分

原版是 `0.5*z(output_quality)+0.5*z(reconstruction)`；V2/V3 改成 `0.5*z(reconstruction)+0.5*z(counterfactual)-anchor_penalty`，同时改变准入、格式回退与 cycle-SFT。它们替换了原来一部分信号，并非只在原机制上额外添加一条保证正确的监督。

反事实项考察“原 caption 是否比少数改写 caption 更能解释音频”。如果改写的负例容易、字段选择不同，或者 TTS 对某些属性不敏感，较高 margin 不保证原 caption 就符合真实音频。冻结 base anchor 只能提供另一个模型的判断；它对超出容忍区的负 margin 扣分，不是外部真值。

TTS 实际 reconstruction 是 teacher-forcing 下 main codec 平均 logprob 加 0.3 倍 sub codec 平均 logprob；Captioner 评分是完整 JSON 上有效 value tokens 的字段宏平均。这些实现减少了部分格式干扰，但其分数不直接等于独立听觉属性准确率。是否受真实 codec 前缀、文本先验、难度或负例集合影响，需要匹配内容的对照；本次没有证据证明发生了 reward hacking。

group 内标准化让有效的微小分数差形成非零 advantage。因此 431 组非零 advantage 证明有更新信号，并不证明信号方向可靠。可能提高能力、可能主要改变回答分布，也可能互相抵消，不能仅从组数判断。

原版连续奖励中还有 ASR/content quality，V2/V3 将其改成硬门槛，实际 min_asr_score 仍为 0。代码先把有限 ASR 分数截断到 [0,1]，所以这道门槛不排除有限的低 ASR 分数。V3 r9 的 203 个 semantic-valid audio-only 候选中，有 56 个 ASR score<0.5；它们是准入候选，不全是被选 SFT 样本。该分数由自动 ASR 估计，不能直接认定音频/文本全部错误，但说明“更严格准入”等同“所有方面质量更高”不成立。

## 7. benchmark 提升也需要拆开解释

StyleCap r9 分任务结果：

| 模型 | gender 正确 / 778 | pitch 正确 / 778 | speed 正确 / 778 | volume 正确 / 778 |
|---|---:|---:|---:|---:|
| base | 483 | 279 | 263 | 234 |
| 原版 | 506 | 282 | 263 | 234 |
| V2 | 494 | 277 | 263 | 234 |
| V3 | 478 | 272 | 263 | 234 |

**base 和三个 r9 checkpoint 的 speed/volume 都是 777/778 题回答 B。** 原版的 StyleCap 涨分主要来自 gender/pitch，不能写成语速、音量都学好了。原版 gender 回答 B 的次数从 base 的 635 增到 752，类别偏好改变参与了总分变化。现有预测不能单独证明这种改变来自更好的听觉辨别。

Para 的主要变化也集中在部分字段。原版 r9 的 accent 从 base 的 0.02857 升至 0.27143，pitch 从 0.02857 升至 0.10714；但原版 situational-traits 的逐样本 F1 约 0.08269，反而低于 V2 的 0.10057 和 V3 的 0.09148。原版不是每个子能力都最好。

Para accent 的 ground truth 有 124/140 条为 american。原版预测 unknown 从 base 的 134 降至 87，american 从 4 增至 39。因此总分增长至少同时伴随拒答减少和类别分布变化，尚不能全部归因于精细口音识别。Para 六字段 parse rate=1.0 只说明解析成功，unknown 同样可以被合法解析。

这些发现不否定原版实测得分优势，而是限制“训练框架更能听懂所有副语言属性”的结论。EmotionTalk 的文本相似度/复合指标也不应直接当作属性分类正确率；CLAPScore unavailable 继续保留为缺失。

## 8. 怎样继续最有信息量

当前优先级应是查清实际损失信号和评分能力，而不是直接再增加轮数或盲目降低 gate。以下只是诊断建议，未执行此前暂缓的消融或额外 GPU 任务。

1. **用已有中间 checkpoint 区分 GRPO 与 SFT。** 在固定、独立且可重复的小验证集比较相邻轮起点、`caption_after_grpo`、`caption_final`，尤其 r0/r6/r9。若 after-GRPO 无改善而 after-SFT 明显变化，能直接缩小解释范围；当前只评 final 无法分离两阶段。
2. **先验明白 reward 能否排对，再选择 gate。** 使用未参与训练的同 transcript、不同属性正负对，按字段计算排序正确率和 precision/coverage；正负例采用对称的比较集合。不能用 benchmark 测试集调门槛，也不能把校准集 90% precision 当作线上伪配对 90% 正确。
3. **检查 benchmark 的音频依赖及回答偏好。** 对 StyleCap 的 speed/volume、Para 的 accent/pitch，先看混淆矩阵、unknown 比例和多数类基线；如后续做音频置换/同内容不同属性诊断，保持 prompt 和内容匹配，区别听觉识别与输出偏好。
4. **以后允许做消融时，匹配训练预算再分离改动。** 相同 MiDasheng base、GPU world size、paired 重复次数和优化配置，从小规模比较 paired-only、原版完整、原版加安全准入、再逐项加入 counterfactual/gate。gate 改变数量是其真实效果，但需另设记录或匹配样本/更新预算的对照，才能区别“筛得更准”和“训练得更少”。不要一次同时改变多项再归因于单个公式。

保留 V3 的安全规范化与审计能力是有依据的；是否保留当前 counterfactual 权重、绝对 gate 和格式回退比例，需要以上证据来决定。最终更好的方案可能组合不同版本的有效部分，而不必在原版和 V3 之间整体二选一。

## 9. 数据与代码定位

- [机器可复核证据：汇总、完整来源、AUROC、bootstrap、预测分布](evidence.json)
- [三版前十轮实际训练量 CSV](training_signal.csv)
- [只读审计脚本](audit.py)：使用共享 `qwen3-captioner/bin/python` 执行；只在本报告目录生成派生证据。
- [已完成逐轮 benchmark 表](../completed_caption_rounds_20260907/results.md)
- [V2 旧审阅](../../../DualISL_Train_RewardV2/reports/research_review_20260905.md)
- [原版 reward](../../../DualISL_Train/dual_isl_train/rewards.py)、[V3 reward/gate](../../../DualISL_Train_RewardV3/dual_isl_train/rewards.py)、[V3 反事实校准构造](../../../DualISL_Train_RewardV3/dual_isl_train/orchestrator.py)

三版 MiDasheng 实际 adapter config 均为同一 base、r=4、alpha=8、dropout=0 和相同 target modules；不要把交接文档里 Qwen 通用的 r=8 表项套到 MiDasheng。现有证据不支持把版本差异主要归咎于 LoRA rank 不同。

限制：原版十轮部分 summary/manifest 未获读取权限，原版精确 SFT/GRPO 数量来自可读的 rank 日志；未核验远端进程，不判断其他训练是否停止。未检查三版训练集与所有 benchmark 的音频级重复、未做多随机种子或新人工标注，不能下无泄漏、因果归因或普遍最优的结论。
