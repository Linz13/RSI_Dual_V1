# Overleaf 临时主表材料（2026-09-09）

用途：给当前论文草稿提供可以直接粘贴的临时主表。仅整理已有结果，不重训、不重新推理、不访问付费 judge、不改写历史结果。

## 最新选择：单张 MiDasheng 原版双侧主表

配套简短正文见 `midasheng_joint_every3_results.tex`（包含实验设置、表格引用和结果解释）。将它与 `midasheng_joint_every3_table.tex` 一起上传 Overleaf，以 `\input{midasheng_joint_every3_results}` 替换初稿 `Current Research Progress` 开头标签为 `tab:caption_tts_results` 的整个旧表；正文文件已引入新表，不要重复插入。旧表的其他引用应改为 `tab:midasheng-joint-every3`。初稿末尾 “One real bidirectional round is complete” 的 Status 也需要按所述版本更新；原版的九轮结果不代表最新 V4 已完成验证。

正文与表注已进一步精简：表头 `Composite` 对应原 `Scheme A`，分数不变；完整指标定义需在论文评测设置中说明。移出表注的复现信息保留在此：前三轮使用四卡，之后使用八卡；EmotionTalk 为 overall-caption 任务、`standard_public` profile；已报告的本地 TTS 指标均覆盖 1,645 条，第 3 轮 judge 仅成功 1,643/1,645 条，因此胜率未报告。

用户已将方案收敛为一张表，仅展示 MiDasheng-7B 原版，每完成三轮取一个 checkpoint，并在同一行对齐 Captioner 和 TTS。当前应使用 **`midasheng_joint_every3_table.tex`**，上传后用 `\input{midasheng_joint_every3_table}` 引入，或直接粘贴正文。

用户后续明确取消第十二轮；当前行固定为 Base、完成第 3/6/9 轮，对应代码中的 Base、r2/r5/r8。该选择按固定间隔取值，不按指标最高点选择。Captioner 的四项数值来自下述 CSV；TTS Base 和第三轮来自 `tts_sources.md`。第三轮 WER/WVMOS 完整、Win Rate 不完整，因此只将该胜率留空；第 6/9 轮 TTS 暂留空。补评入口见 `../../README_MIDASHENG_TTS_LOCAL.md`；只补两轮本地 WER/WVMOS，不调用 API。

2026-09-09 用户后续停止八卡评测，改用 GPU 4–7；当前补评入口及新的四卡输出目录见 `../../README_MIDASHENG_TTS_LOCAL_4GPU.md`。后续填表应读取该四卡输出的完整本地 summary，不把原八卡 partial 当完成结果。

数值舍入与来源匹配、LaTeX 括号/环境配对已检查；本机未安装 LaTeX 编译器，未编译检查实际排版。之前两个分表保留为历史备选，不再作为当前推荐。

## 之前的双表备选

1. `captioner_main_table.tex`：Captioner 主表，展示原版在三个 backbone 上的结果，并显式标注 Qwen2.5 RewardV2 变体。
2. `tts_main_table.tex`：同一双模型实验的 TTS 侧外部评测，缺失结果用 `--`；不同表允许覆盖不同轮次，但不能把不同轮次写成同一对 checkpoint。

两个文件直接粘贴到正文，或上传后分别用 `\input{captioner_main_table}` 和 `\input{tts_main_table}` 引入。依赖 `booktabs`、`graphicx`，用户提供的草稿已经加载。不要在两个文件中再添加 documentclass 或 begin/end document。本机未找到 pdflatex/tectonic，因此未执行 LaTeX 编译。

## Captioner 行选择

- 三个 backbone 原版均展示已完成第五轮（代码 r4），便于按相同轮次观察；这不是跨模型等算力对照。
- MiDasheng 增加第十轮（代码 r9）和当前汇总中最后的第十三轮（代码 r12），同时保留后期 Para 小幅回落，避免只展示测试最高点。
- Qwen2.5 RewardV2 展示第五轮及本批最后已评测的第九轮（代码 r8），明确标注版本。r0 已经训练过一轮，不能充当 Base。
- 没有在不同指标中拼接各自最佳 checkpoint；不使用最佳值粗体暗示统计显著性。
- MiDasheng RewardV2/V3 的比较适合独立版本分析，未塞入这两张以跨模型表现、双侧能力为主的表。原始结果仍完整保留在下面来源中。
- V4 尚缺完整评测，不报告为已验证的最终方法；如果论文最终方法定为 V4，应在有结果后明确区分 V4 与原版基线。

## Captioner 数据来源与显示规则

- `../completed_caption_rounds_20260907/rounds.csv`：MiDasheng、Qwen3。`sources.json` 给出原始结果路径，`results.md` 为先前完整核验报告。
- `../qwen25_v1_rewardv2_results_20260908/rounds.csv`：Qwen2.5。`verification.json` 给出本批 42 项 full 的来源与核验记录。
- Para/SPIDEr/FENSE 保留原始尺度，显示四位小数；StyleCap 乘以 100，显示两位小数并标 `%`。所有缺失使用 `--`，不计为 0。
- 未在本次重新核验所有训练权重或远端进程。所选轮次来自已有已核验报告，不将计划轮数视为完成状态。

## TTS 数据来源与显示规则

`tts_sources.md` 记录此次实际读取的 EmergentTTS 最终分数、完成文件和来源。六个完成 case 使用已有全量 1,645 条的 final metrics；MiDasheng 第三轮只填完整本地 WER/WVMOS，judge 尚缺两条，Win Rate 留空。

WER 显示两位小数并标 `%`，WVMOS 显示三位小数，Win Rate 乘以 100 后显示两位小数。Qwen3 第三轮 WER 原值约 14.8947，按原始精度舍入为 **14.89**，用户早期初稿的 14.90 不沿用。Qwen3 第一、二轮 WVMOS 都显示 4.321，但原值不同；不把舍入后的相同值解释为完全相等。

Win Rate 的对手是 benchmark 的 `gpt-4o-mini-tts-alloy`，不是未训练的 Qwen3-TTS Base；ties 计 0.5，judge metadata 为 Gemini-2.5-Pro。共享 Base 只列一次。当前结果不能支持两侧模型都持续改善。

## 解释范围

当前主要正向证据是原版 MiDasheng 的 ParaSpeechCaps 提升。Qwen3 基本持平，Qwen2.5 缺 Base，不能声称所有 backbone 均明确改善。EmotionTalk 使用中文参考与预测，但默认 SPIDEr/FENSE 组件缺少中文适配验证，仅作为带注释的辅助指标。上述表不能独立证明提升来自双向机制而非 paired-SFT 等训练成分。

原版与 V2/V3 不是严格单因素、等更新预算消融。当前主表没有多训练种子统计；不要添加显著性星号。若只放一张表，优先 Captioner 表；TTS 表以后续结果逐步补齐。
