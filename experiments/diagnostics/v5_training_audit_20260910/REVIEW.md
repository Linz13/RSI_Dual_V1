# V5 最新训练实现与真实运行审计

审计时间：2026-09-10 11:48（北京时间）。只读取训练产物、运行 CPU 检查；未修改运行中的代码、配置、checkpoint，也未调用打标 API。

结论：核心闭环、奖励计算、SFT 目标、paired anchor 和真实参数更新符合已确认方案。没有发现需要立即废弃第一轮或中止训练的奖励算错、目标错配、漏训/重复训练证据。但 **Captioner 的真实批量生成没有实现预期加速，第一轮后尚未观察到格式能力提升，“无声音指令也能获奖并入选 SFT”的宽松准入后果已经出现**。另有一处原始 schema 有效率统计残留问题。训练完成和权重变化不等于 benchmark 效果提升。

## 1. 核查的是哪个版本

- 当前生效源码：`/data/L202500147/Caption/DualISL_Train_RewardV5_FastResume`。
- 训练目录仍在 `DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01`。
- 源码实现摘要：`e081f357623773e0c26ab6081a1477d1b2af992d6e378ca708c8010b0f7a8ca7`。与当前配置组成的锁和 `run_state.json` 一致，外部打标资源摘要也一致。
- 第一轮 `round_000` 是 LabelRobust 完成的；FastResume 从第二轮 `round_001` 接着训练。两种版本不能混在一起评价提速。
- 共 10 轮，8 卡，paired anchor 开启；第一轮两模型从基模开始，没有 warmstart。
- 第一轮两个最终 checkpoint 的完整摘要重新核验通过。第二轮加载它们时的 adapter audit 显示权重精确匹配。

## 2. 逐项对照需求

| 已确认要求 | 代码及实际结果 |
|---|---|
| audio-only 用生成音频属性和原音频属性比较 | 已实现；457 个完成重建的候选全部使用缓存评审结果重新逐字段计算，与存储值完全一致 |
| 不奖励中间 caption 与 GT 属性直接匹配 | 已实现；audio-only 最终只含重建和格式两项 |
| 0.9 重建 + 0.1 格式 | 已实现，736 候选的总分、组内 advantage、skip、top1 选择均复算一致；权重固定，没有自动衰减日程 |
| 停用 audio-only 旧 likelihood、反事实、冻结基模奖励和校准 | 已实现；旧配置中的 audio-only counterfactual_weight 不被该分支读取，校准文件仅有 caption_only |
| 部分合法属性也能进入 TTS | 已实现；合法字段逐个提取，大小写及空白安全规范化；非法、缺失、unknown 不成为声音指令 |
| 不擅自修补 GRPO 原始生成轨迹 | 已实现；训练保留原始 sampled_token_ids 和 old/ref 概率，规范化对象只构造 TTS 请求 |
| 转录不计重建分，不检查转录是否正确 | audio-only 已实现；转录仍用于 TTS text、语速和强调位置对齐。caption-only 保留 V4 分支，仍会计算旧 critics/ASR 诊断，其 ASR 阈值为 0 |
| 原音频 unknown 不计分，生成音频 unknown 不能缩小分母 | 已实现；按原音频已知字段固定分母（本轮 12～15 项），生成 unknown 对应 0 分 |
| 评审请求失败不能冒充低分音频 | 已实现；基础设施或完整响应失败保持 pending；合法完整响应中的个别非法字段按已确认规则重试后降 unknown。本轮最终远程缓存没有字段降级记录 |
| audio-only 选重建分最高 caption 做 TTS SFT | 已实现；选中 176 条，与逐组重建 top1 一致，不按含格式项的总分选 |
| TTS SFT 的 GT 是原音频 | 已实现；176 条伪配对均指向原音频 codec，另有 39 条 paired；未用重建音频自我监督 |
| Captioner SFT 用原 caption-only caption 作 GT | 已实现；196 条使用获选合成音频作输入、对应 source_caption 作目标 |
| 删除原始大 schema Captioner anchor | 已实现；Captioner 只有 39 条 P_syn anchor，TTS 39 条，不再有 78 条 Captioner anchor |
| 两个 GRPO 暂时不改 | 与 V4 做 AST 方法对照：两个 grpo_update 及关键 loss/backward 方法相同。FastResume 相比 LabelRobust 仅 3 个核心文件变化，未修改奖励与更新方法 |
| 本地打标避免和训练模型争抢显存 | 已实现；合成过程中只重叠远程 API；合成进程退出后依次运行本地专家，各阶段释放模型；没有 StepAudio 参与 |

原音频 reference 本轮来自同一固定自动打标流水线及缓存；`reference_labels_path` 为空，**不是读取人工逐条修正后的标签文件**。

## 3. 当前固定打标方案

| 属性 | 打标来源 | 重建比较 |
|---|---|---|
| 转录 | Qwen3-ASR | 不单独计分、不做内容门槛 |
| 语言 | Gemini-3.1-Pro-Preview | 枚举一致 |
| 性别、年龄、音色 | Qwen3.5-Omni-Plus | 各字段枚举一致 |
| 口音 | 英语 XLSR / 中文 VoxLect | 枚举一致；其他语言 unknown |
| 语速 | Brouhaha + 英语 G2P / 中文 pypinyin | 枚举一致 |
| 音高 | Gemini-3.1-Pro-Preview | 枚举一致 |
| 音量 | 原 RMS 阈值规则 | 枚举一致 |
| 情绪 | emotion2vec-plus-large | 枚举一致 |
| 情绪强度、强调等级 | Qwen3.5-Omni-Plus | 各字段枚举一致 |
| 强调词句 | Qwen3.5-Omni-Plus | 对齐原/生成转录中的词字位置后集合 F1 |
| 韵律、停顿 | Gemini-3.1-Pro-Preview | GPT-5.5 固定 rubric，0 / 0.5 / 1 |
| 非语言发声 | Qwen3.5-Omni-Plus | 事件集合 F1 |

共 16 个叶子字段，转录以外 15 个评分字段；强调等级与强调词句分开计分。本轮原缓存包含 Gemini/Qwen35 各 641 份响应（184 原音频 + 457 重建音频）及 387 份可复用文本评审，记录的返回文本模型均为 gpt-5.5。打标没有调用 Qwen3-Captioner、Kimi、StepAudio，也没有多数投票和人工补标。

## 4. 第一轮确实训练了什么

| 阶段 | 真正参与的记录 | 每卡同步优化步数 | 平均 loss | 阶段耗时 |
|---|---:|---:|---:|---:|
| Captioner GRPO | 176 组 | 22 | 0.000072 | 38.2 分钟 |
| Captioner SFT | 196 cycle + 39 anchor = 235 | 30 | 1.0613 | 3.3 分钟 |
| TTS GRPO | 196 组 | 25 | -0.000209 | 69.3 分钟 |
| TTS SFT | 176 cycle + 39 anchor = 215 | 27 | 3.5758 | 1.9 分钟 |

跨卡汇总真实 sample_ids 和输入/应训练组逐一对应，没有遗漏或重复真实记录。不能把 8 卡的同步步数相加当作全局优化步数；不足整批的 rank 使用零权重占位，真实记录计数不包含占位。

四阶段 loss 有限，全部 rank 的参数签名发生变化。TTS 更新前后显式跨卡权重检查为全相等；Captioner 是显式梯度 all-reduce、各 rank 报告的更新后参数签名相同，但其输出没有 TTS 那样的逐张量跨卡验证报告，不把签名相同描述为逐张量证明。

TTS SFT 实际目标为 `-mean(main_logprob) - 0.3 * mean(sub_logprob)`，即原音频 codec 的 teacher-forcing token NLL；不计算两段 waveform 的逐点相似度。Captioner SFT 为目标 JSON token 的交叉熵，音频/提示部分不作为标签。GRPO loss 接近 0 属于组内优势中心化时可出现的现象，不能据此认为没有训练；应看非零梯度、参数变化及独立评测。

## 5. 重建奖励实际分布

- 736 个 caption 候选，457 个进入音频重建与打标；176/184 组有有效 cycle 样本。
- 完成重建者均分 **0.5746**，中位数 **0.5833**，25%～75% 分位 **0.4508～0.7000**，范围 **0.1333～0.9333**。
- 176 个 top1 均分 **0.6475**，最低 **0.25**。低分 top1 仍会进入 SFT，符合已确认的相对 top1 规则；“入选”不代表达到独立高质量门槛。
- 同组有效候选的 422 对非平分比较中，仅 4 对被格式项逆转重建排序（约 0.95%）。第一轮没有发现格式项普遍压过重建项。
- 8 个完全没有可用重建的组，格式分也没有形成可用差异，GRPO 全部跳过；本轮没有单靠格式项更新的组。
- 匹配较高：语言 0.965、年龄 0.869、性别 0.825；较低：音量 0.357、情绪 0.361、韵律 0.388。不能用整体均分代替各属性改进。
- 原音频 unknown 主要为音高 13、韵律 12、停顿 11、语速 7；重建音频对应 unknown 为 46、40、41、6。这些 uncertain 标签依规则处理，没有导致整轮中断。

这些数值是训练用裁判的回路一致性，不能独立证明 caption 正确或模型变强，更不能仅凭分数不是很低判定奖励设计有效。

## 6. 需要关注的发现

### A. Captioner batch=4 实际未生效（效率问题，优先级中）

源码位置：[midasheng_captioner_candidate.py:194](/data/L202500147/Caption/DualISL_Train_RewardV5_FastResume/scripts/midasheng_captioner_candidate.py:194)。批量行为概率与单条回放概率不一致时，当前 rank 后续全部串行。

第一轮和第二轮各 736 个最终候选全部 `generation_batch_size=1`，原因均为 `trajectory_mismatch`。不是 8 卡没用，而是每卡上的 4 候选没有并行生成；日志仅 8 个组有 `batch_fallback` 标记，不能误读为只有 8 组串行。代价与校验被保留，未发现错误批量轨迹被强行训练。

处理建议：若保持 GRPO 不动，先承认 Captioner 目前是“8 卡分片、卡内串行”；要真正实现 batch=4 需在独立 GPU 验证中解决生成/回放数值路径一致性，不能直接放宽阈值或覆盖 old_logprob。

### B. 第一轮后尚未观察到格式改善（效果风险，优先级中）

| 同一 184 条音频的采样 | 第一轮更新前 | 第一轮更新后、第二轮采样 |
|---|---:|---:|
| 候选总数 | 736 | 736 |
| 可解析 JSON | 457（62.1%） | 434（59.0%） |
| 能构造 TTS 请求且轨迹有效 | 457（62.1%） | 432（58.7%） |
| 平均格式分 | 0.5761 | 0.5383 |
| 按 V5 准入口径重算完整 schema | 101 | 98 |

第一轮不可用的 279 条中只有 6 条达到生成长度上限，其余 273 条已经 EOS，但 JSON 有语法/结构错误。第二轮也只有 6 条 length。**主要问题不是 max_new_tokens 太小，单纯加长生成不能针对性解决。**

这是不同随机种子的训练采样，不是固定种子的受控前后测，不能据此断言模型已经退化；但还没有证据说格式奖励达到了“先把格式能力拉起来”的预期。当前 0.9/0.1 是固定权重，没有随轮次/格式能力调整。格式合法的 unknown 本就能获得格式分，这一项不评价信息真实性或充足性。

建议等当前第一轮 benchmark 出来，再决定是否增加短暂的 P_syn 格式监督预热或调整训练安排；不建议只把格式系数加大就期待解决所有错误。

### C. 没有声音指令也能入选 SFT，已实际发生（设计后果，优先级中）

准入位置：[partial_caption.py:137](/data/L202500147/Caption/DualISL_Train_RewardV5_FastResume/dual_isl_train/partial_caption.py:137)；选择位置：[attribute_reward.py:130](/data/L202500147/Caption/DualISL_Train_RewardV5_FastResume/dual_isl_train/attribute_reward.py:130)。准入只要求可用 transcript，没有声音属性数量下限，SFT 只看重建 top1。

第一轮有 **22/457** 个合成请求的 instruct 最终仅为 `Speak naturally in a neutral voice.`；它们重建均分 **0.4986**、最高 **0.7667**，其中 **11 个被选中**进入 TTS SFT（176 个 cycle 样本的 6.25%）。第二轮此类请求有 **34/432** 个。

这不证明这 11 条在听感或所有标签上错误；它说明当前奖励只能验证 TTS 输出匹配，不能保证中间 caption 提供了足够声音信息。用户此前担心的机制确实存在，并非假设一写出来就会自动消失。

它符合此前“不设最少声学属性数量”的宽松规则，不是分数算错。如果要收紧，建议优先考虑 **至少一个可渲染的有效声音属性**（排除 transcript/language、空列表和 unknown），尤其可作为 TTS SFT 的候选资格条件；这仍然不需要加入中间 caption 对 GT 的直接监督奖励。但该规则会改变实验定义，应在明确版本边界实施，不能与已完成第一轮混称同一训练设置。

### D. 原始 schema 计数有 1 条旧标志残留（统计 bug，优先级低）

[partial_caption.py:97](/data/L202500147/Caption/DualISL_Train_RewardV5_FastResume/dual_isl_train/partial_caption.py:97) 的解析失败分支没有返回 `raw_schema_valid=False`。上游使用 `c.update(admit(...))`，所以旧 parser 留下的 True 可能不被覆盖。

第一轮 `random440_expressivespeech_031::1` 恰好触发：V5 判断 JSON 不可解析，仍残留旧的 `raw_schema_valid=True`。因此看板/原 summary 显示 102 条，按 V5 重新统计是 101 条。

该条没有进入合成，未被选 SFT，重建/格式奖励均未错误增加。影响的是 schema 有效率展示。修复方式是在 admit 初始化结果中明确设置 `raw_schema_valid=False`，同步明确采用哪个 parser 口径。当前未改动运行源码。

### E. 旧版测试入口没有完全清理（维护问题，优先级低）

V5 专项 **160/160 通过**。另外主动运行整个继承的 `tests` 目录时，总计 272 passed、40 failed、20 errors、3 skipped：失败集中于 V2/V4 配置被 V5 的 version 校验拒绝，以及没有迁移来的旧 `.recovery/midasheng_parserfix_v1` fixture；并非这 60 项都证明 V5 正在算错。旧版测试仍不适合直接作为 V5 全量验收入口，应清理/迁移测试配置。完整失败记录予以保留，没有宣称全仓测试通过。

## 7. 最新提速的实测

| 阶段 | 第一轮 LabelRobust | 第二轮 FastResume |
|---|---:|---:|
| audio-only TTS 合成 | 457 条，40.4 分钟 | 432 条，6.9 分钟 |
| 实际分布 | 原单进程合成 | 8 个 rank，各 52 或 56 条 |
| 合成 batch | 保留原规则 | 432 条全部 batch=4 |

第二轮合成结果的 ID 集合、条数、request 与 generation_seed 均和输入一致；全局 batch 成员保留，GPU 峰值约 6.83 GiB。已实测的这一段明显提速，但样本、文本和模型权重不同，不能声称严格 5.8 倍受控加速，更不能推广为整轮加速倍数。

caption-only TTS rollout 已配置为直接串行，避免第一轮 193/196 组先尝试批量再重做的浪费；在本次审计时第二轮尚未到该阶段，只有代码/配置和 CPU dispatch 测试证据，尚无第二轮 GPU 耗时。

远程打标/文本评审各 4 worker；本地专家逐阶段执行。TTS 合成加快后远程打标可能不再被充分隐藏，整轮时长还要等本轮完整阶段记录。Qwen3-ASR 虽设置 max_inference_batch_size=4，当前调用仍逐条 transcribe，也不能把此配置当作四音频真实 batch 的证明。

## 8. 当前建议

保留正在运行的实验和已开始的第一轮 benchmark，无须因本次发现重训第一轮。先用独立评测看第一轮是否带来收益，同时观察第二轮完整耗时。下一版优先解决格式能力验证和纯默认声音请求的准入；Captioner 真正 batch=4 属于需要单独验证的推理一致性工作。

本次依据是代码、配置、全量第一轮离线复算、部分第二轮真实产物和 CPU 测试；没有人工听完所有音频，也没有对第一轮声学内容正确率给出独立证明。

审计数据：[audit.json](audit.json)；审计脚本：[audit.py](audit.py)；V5 测试：[V5_TESTS.log](V5_TESTS.log)；含旧测试的记录：[CPU_TESTS.log](CPU_TESTS.log)。
