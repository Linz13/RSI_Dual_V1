# 打标方案与结果解释

这里解释随包代码的实际行为。模型—属性分工以 `source/Experiment/labeling2/pipeline.py` 为准，schema 以 `target_schema.py` / `target_schema.json` 为准。

## 属性分工

简称：G = Gemini (`gemini-3.1-pro-preview`)；Q = Qwen3.5 (`qwen3.5-omni-plus`)；C = Qwen3-Captioner；K = Kimi-Audio；S = Step-Audio-R1.1。

| 属性 | 来源 | 合并方式 |
|---|---|---|
| semantic_content.language | G | 单来源 |
| semantic_content.topic | G | 单来源 |
| semantic_content.intent | Q | 单来源 |
| speaker_profile.gender | Q、C | 2/2 |
| speaker_profile.age | Q、C | 2/2 |
| speaker_profile.timbre | Q | 单来源 |
| speaker_profile.accent | 英文 XLSR / 中文 VoxLect | 按语言选择专家 |
| paralinguistic.speaking_rate | 英/中文语速专家 | 计算 + 阈值 |
| paralinguistic.pitch_level | G | 单来源 |
| paralinguistic.volume_level | RMS 能量 | TextrolSpeech 阈值 |
| paralinguistic.emotion | emotion2vec-plus-large | 取最高分类分数并映射标签 |
| paralinguistic.emotion_intensity | Q、C | 2/2 |
| paralinguistic.emphasis.level | Q、C、K | 2/3 |
| paralinguistic.emphasis.emphasized_text | Q、C、K | 文本裁决，2/3；level=none/unknown 时直接 [] |
| paralinguistic.prosody | G、Q、C | 文本语义裁决，2/3 |
| paralinguistic.pause | G | 单来源 |
| paralinguistic.nonverbal_vocalization | Q、C、S、K | 每种事件 3/4 |
| environment.background_sound_events | G、Q、S | 文本语义裁决，2/3 |
| environment.recording_quality | Q、C | 2/2 |
| environment.acoustic_scene | G、Q | 文本语义裁决，2/2 |

此外，当前代码会把输入 manifest 的 `transcript` 复制到最终 `semantic_content.transcript`（没有则为空字符串）。它不是新增的模型预测字段，语速 fallback 的 Whisper 转写也不会因此自动成为该字段的 GT。

## Prompt 与输入

`prompts.py` 的 `build_prompt()` 按模型分配的字段生成嵌套 JSON 要求、枚举值和字段描述。`reference/prompts/` 保存打包时五份完整模型 prompt 和开放裁决 system prompt。

通用音频模型只接收音频和标注指令，不接收数据集 metadata 或现成转写。多说话人音频按主要说话人标注，没有执行 diarization。Kimi 另有短输出、去重和停止约束。输出非法时有字段级验证与修复提示；保留有效字段，记录 raw response、parse_status、field_errors 和 attempt。

`manifest.jsonl` 每行一个样本，必须有唯一 `sample_id` 和存在的 `audio_path`。可选 `dataset`、`metadata`、`transcript`、`language_hint`。相对音频路径按 manifest 所在目录解析。

## 封闭类别共识

实现：`consensus.py`。

- 参与来源数固定为分工表的来源数；失败/缺失不会降低门槛。
- 2 个模型要求 2 票，3 个要求 2 票，4 个要求 3 票。
- 标量按规范化后的字符串计票，未达门槛返回 `unknown`。
- 非语言发声事件逐项计票，每个模型对同一事件最多投一票；保留达到门槛的事件。
- `none` 与实际事件互斥。封闭字段中的 `unknown` 可以形成“模型都不确定”的一致结果，但它仍不是已知属性值；不要仅看 passed 就当作标签有信息。

## 开放字段语义裁决

实现：`open_resolution.py`。初步 `finalize` 给场景、背景音、韵律及需要裁决的重读词句保留候选，后续 `resolve-open` 才处理。

历史使用 `gpt-5.6-sol`，只看候选文本，不听音频。候选以稳定的匿名 ID 排序提供；裁决模型先找核心含义一致的候选集合，再为每条保留的描述提供足够数量的 supporting IDs。无效/unknown 候选视作不可用，不能降低 required_support。

- 场景：只保留两者共享的具体场景含义。
- 背景音：同义事件聚类，每个事件至少两个候选支持。
- 韵律：只保留节奏、语调、分句方面的共同描述；不能重复语速、音高档位、音量、重读、停顿等独立字段。
- 重读词句：至少两者支持的重叠原文片段，允许大小写/标点归一化，不能改写。代码进一步检查片段能否在支持者候选中找到。

裁决器响应还有结构和支持数量校验。语义支持关系本身仍依赖裁决模型判断，不是程序能够证明的事实。冲突返回 `unknown`、`["unknown"]` 或 `[]` 并进入复核队列。

## 专家计算规则

实现：`expert_worker.py`。

音量：`librosa.load` 后计算 `librosa.feature.rms(y=y).mean()`。小于 0.03331899 为 low，大于 0.05054203 为 high，边界和中间为 medium。这是录音信号能量分档，不是校准声压级。

英语语速：已有转写优先，否则 Whisper turbo；G2P `eng -> eng-ipa` 后实际用 `len(output_string)` 计数，除以 Brouhaha VAD 的语音时长。小于 11.477411477411476 为 slow，大于 19.129019129019127 为 fast，中间及边界为 moderate。不要把这个计数解释成严格分词后的 IPA 音素数。

中文语速：提取汉字，用 pypinyin 非空声母数量 + 非空韵母数量作单位数，除以同样的 VAD 语音时长。小于 4 为 slow，大于 10 为 fast，中间为 moderate。代码将其注明为受论文启发的部署近似方案。

情绪类别取 emotion2vec 最高分；情绪强度来自 Q/C 共识，两者没有用分类置信度绑定。英文口音标签由 checkpoint 输出映射；中文八类标签顺序与 VoxLect 分类头一致。

专家阶段优先使用 Gemini 语言结果；如果没有合法的语言字段，调度可用 language_hint 兜底。但最终 accent/rate 合并按最终 Gemini language 选择专家，所以完整运行应先完成 Gemini，不能把 language_hint 当作最终语言标签替代。

## 输出与续跑

- `raw_predictions/`：各通用模型追加式预测和重试记录。
- `expert_predictions/`：专家结果与 evidence，包含数值、阈值、转写和 VAD 信息。
- `state/manifest_snapshot.jsonl`：当前 run 不可混用的样本快照。
- `final/labels.jsonl`：初步合并标签。
- `final/provenance.jsonl`：初步来源和投票详情。
- `final/open_candidates.jsonl`：开放字段候选。
- `open_resolution/`：裁决输入状态、原始响应和 decisions。
- `final/labels_open_resolved.jsonl`：开放字段裁决后的标签。
- `final/provenance_open_resolved.jsonl`：最终每字段来源。
- `final/review_queue_open_resolved.jsonl`：最终仍需复核的字段。

`--resume` 依靠既有预测和状态跳过已成功的任务，不等于允许更改输入样本。裁决阶段校验输入哈希、URL 和模型。变更音频路径、样本集合或裁决输入后不能无条件复用旧状态。

`open_resolved` 表示执行过裁决，不表示所有字段有确定答案。`["none"]` 是无事件判断，`["unknown"]` 是未能确定，不能互换。

历史 300 条运行包含英文 150、中文 150，最终仍有 259 条样本带复核事项；这是理解共识筛选强度的参考。当前源码与历史生成时源码并非承诺完全一致。完整统计与两条示例见 `reference/historical_summary.json` 和 `reference/historical_examples.json`。
