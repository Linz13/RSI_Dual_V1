# 当前训练耗时与小模型未明显提速的诊断

2026-09-09；读取真实完成阶段、rollout 轨迹、TTS GRPO 心跳及源码。未启动 GPU、修改训练实现/配置或停止任务。当前两条 V4 run 的 37 项源码身份均匹配。

## 结论

现有实现存在明确的吞吐瓶颈，四小时不能简单解释为模型参数规模导致的必然耗时。但当前记录也不足以认定存在运行死锁、GPU 未启用或错误地重复训练整轮。耗时集中在生成及逐 token/逐帧的 GRPO 回放，而非 SFT。

本次没有远端 GPU profiler/利用率时间线，不能定量区分 GPU kernel、CPU 调度、内存带宽、通信和文件系统成本。已定位计算路径，尚未实测替代实现的提速与数值等价性。

## 实际阶段时间

以下为成功 stage 的 elapsed_seconds，包含子进程启动/加载/执行/退出；不包括所有初始化、未计时 bookkeeping、故障重跑或人为暂停。只统计有 commit 的轮次；本次未重新验证训练权重字节。

| 阶段 | MiDasheng-7B V4，8 卡，完成第 2–4 轮均值（分钟） | Qwen2.5-Omni-3B V4，4 卡，完成第 2 轮（分钟） |
|---|---:|---:|
| TTS GRPO | 72.92 | 116.88 |
| TTS rollout（含生成及概率回放） | 47.61 | 82.59 |
| Captioner GRPO | 39.80 | 62.92 |
| Captioner rollout（含生成及概率回放） | 36.13 | 64.83 |
| 两侧 SFT 合计 | 6.37 | 4.77 |
| 其余阶段 | 25.81 | 29.30 |
| 合计 | 228.64（3.81 小时） | 361.28（6.02 小时） |

MiDasheng V4 第 1–4 轮分别为 258.90、239.37、239.34、207.23 分钟；Qwen 四卡第 1–2 轮为 393.04、361.28 分钟。第一轮 reference 复用及历史长度分配有回退，不宜单独用于预测后续轮速。

历史 Qwen2.5-Omni-3B V2 八卡九轮平均 253.11 分钟（4.22 小时），证实小参数模型在八卡上也可能接近四小时。V2/V4 奖励、实现、生成轨迹等不同；该事实不是跨模型或跨版本的严格吞吐实验。原版 Qwen 八卡阶段文件本账号 PermissionError，未使用其计时。

## 为什么参数变小没有明显缩短整轮

1. **两侧模型一起工作。** 换 Captioner 没有换 TTS。MiDasheng V4 后三轮中，TTS rollout + GRPO 已占 120.53 分钟、总时长的 52.71%。即便其他阶段理想化地全部消失，这两个阶段仍在；这是固定当前工作量的示意下界，并非换模型后的时长预测。
2. **当前小模型只用了四卡。** 第 2 轮 Captioner 每 rank 分到 46 个源组，而 MiDasheng 八卡为 23 个。两个 run 有效 batch 和 optimizer 步数也不同，不能只比较模型参数量。
3. **单卡计算颗粒过小。** 配置中 rollout_batch_size 和 replay_batch_size 均为 1。MiDasheng generate_batch 实际对 seeds 循环逐条调用 generate；其 sampled_token_logprobs_batch 同样逐轨迹执行。Qwen 代码有微批接口，但当前配置为 1。
4. **输出长度及实现路径也决定时间。** MiDasheng、当前 Qwen 四卡、历史 Qwen V2 八卡第 2 轮生成的 Captioner token 均值分别为 165.64、149.05、164.21，并没有随参数量缩成几分之一。前两个 run 的 EOS 完整候选为 732/736、731/736，不支持“普遍生成到 384 上限”解释全部耗时。

## 源码中的主要瓶颈

### TTS GRPO 使用逐帧、逐子码本回放

`dual_isl_train/workers/qwen_voice_design.py:352` 的 `_incremental_trajectory_impl` 在每个 frame 内顺序计算 15 个子码本分布：1 次 predictor 前缀调用加 14 次增量调用。外层 GRPO 每个候选调用此函数并单独 backward（816 行起）。这产生大量小前向调用和很大的增量 autograd 图。

MiDasheng V4 第 2 轮共有 196×4=784 个 TTS 候选、64,814 个 codec frame；单次回放全部候选对应 972,210 次 predictor 调用（不是 GPU kernel 数；GRPO padding/跳过可改变实际数）。当前生成后还有 policy replay，GRPO 更新时再回放，生成与更新成本都存在。

当前 TTS GRPO 已有长度分桶和候选级反向，非末候选使用 DDP no_sync；这些不是尚未实现的优化。第 2 轮 8 个 rank 的 step 时长累积均值为 73.22 分钟，接近整个 stage 的 75.24 分钟，说明单靠优化模型加载不会解决该阶段大头。

心跳中前向/构造 loss 区间占 36.90%，反向/同步区间占 63.10%。这是所有 rank 的主机时间区间汇总，含日志与等待，不能说“63% 都是通信”或纯反向 kernel。

同文件 477 行已有 `_trajectory_impl`，能对已知 codec 序列用 teacher forcing 计算概率，并调用 `forward_sub_talker_finetune`。它是值得对照的替代路径，但当前 GRPO 未使用，不能直接替换后声称等价。需核查同一轨迹、conditioning、main/sub mask、generation_steps、概率处理、loss、LoRA 梯度、一步更新和峰值显存。并行计算已知动作序列的 likelihood 本身不等于把 GRPO 换成 SFT；是否保持当前实现的目标和梯度仍须验证。

### Captioner 同样使用原生生成循环回放

`scripts/midasheng_captioner_candidate.py:245` 与 `dual_isl_train/workers/qwen3_captioner.py:490` 使用 generate 强制逐 token 重放已采样 ID；GRPO 每 16 token backward 并 detach KV。这种实现保护了原轨迹和显存，但吞吐有限；不能把改 chunk、取消 detach 或整序列反向当作无需验证的无损开关。

`dual_isl_train/workers/qwen3_captioner.py:706` 的 rollout 仍依次执行 policy replay 与 reference replay。第 2 轮两条 V4 run 的 736 个 Captioner 候选记录的两者最大概率差均为 0，且轮初 adapter 审计精确匹配。可评估复用，但需独立确认身份、模式及概率等价，不能在参数已更新后复用。

### 已做过的优化及剩余开销

TTS reference replay 复用和 rollout 历史长度分配已生效。MiDasheng 第 2 轮独立 reference 验证 8 次、复用 776 次；Qwen 四卡验证 4 次、复用 780 次。不要重复安排这两项已完成工作。

MiDasheng 生成 callback 每 token 两次 CPU 标量读取，TTS 候选后 CUDA synchronize、共享文件心跳写入也是开销候选，但当前没有 profiler 证据说明它们占主要比例。

调度器的 audio-only collection、caption-only collection、Captioner 更新和 TTS 更新按顺序执行，各阶段本身有多卡并行。这不是“只用了一张卡”；也不能未经计时就认为把一台八卡切成两个四卡阶段会更快。

## 建议的下一步

### 补充：当前实现与批量训练惯例（同日问答核验）

“双向生成→评分→GRPO→循环SFT”的研究设计不要求每卡单样本、逐轨迹串行；这些属于实现选择。跨卡数据并行已存在，单序列自回归生成的时间依赖也真实存在，但不同候选可批处理，已知轨迹的 likelihood 计算应评估并行/分块路径。

交接文档历史段记录早期 Captioner batch 2/4 在约78.6 GiB OOM，随后 H100 配置统一固定每rank batch1；TTS为限制计算图峰值采用逐候选backward。这些是可核实的历史背景，不等于已证明当前所有模型和阶段都只能batch1。MiDasheng V4当前完成四轮的Captioner GRPO峰值约27.5 GiB，TTS GRPO各轮最大峰值37.22–39.55 GiB；第二轮rollout峰值Captioner约17.8、TTS约5.2 GiB。这些是PyTorch allocated峰值，不包含全部驱动/通信分配，且未覆盖未来全部长轨迹。可据此安排batch2/4的实测，不可按80/当前峰值直接推定可用batch。

公开实现对照：

- [Qwen3-TTS官方README](https://github.com/QwenLM/Qwen3-TTS/blob/main/README.md)有VoiceDesign批量生成示例，说明底层模型并非只能单条生成。当前worker额外记录main/sub采样概率、变长轨迹和种子，且回放显式要求batch维为1；普通批量音频生成接口不能直接替代完整RL rollout。
- [Qwen3-TTS官方SFT脚本](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/sft_12hz.py)使用DataLoader(batch_size)、批量序列forward与forward_sub_talker_finetune，默认batch2、梯度累积4。这是Base模型SFT示例，不是当前VoiceDesign GRPO的已验证替代方案。
- [Hugging Face TRL GRPO文档](https://huggingface.co/docs/trl/grpo_trainer)分别提供训练batch、生成batch、梯度累积与vLLM生成选项。说明批量/微批是标准可用手段，不能据此保证自定义16-codebook语音GRPO可直接套用TRL或vLLM。

建议固定group_size=4及optimizer更新频率，在group内比较microbatch1/2/4，保持每个候选的有效长度mask和原损失归一化。梯度累积可保持有效训练batch，但仅增加累积次数不会让单卡计算自动变成向量化batch。训练正确性与实现吞吐应分别验收；显存有余量是优化依据，目标是吞吐和反馈周期，而非单纯把显存填满。

优先用独立单 H100 短测试诊断 TTS GRPO：固定短/中/长各一组、每组四候选，比对当前增量回放与并行/分块 likelihood 路径的概率、loss、LoRA 梯度、一步更新、峰值显存及预热后耗时。发现算法差异必须记录并解决，不能只放宽误差门槛。通过后再做数步八卡同步检查，最后才接入正式训练。无须为组件优化重跑十轮。

随后评估 Captioner 条件满足时的 reference 回放复用、真正的回放批处理/只读输入复用，以及 profiler 显示的 CPU 同步开销。不要先花主要精力削减占约 2.8% 的 SFT 或不足十分钟的诊断校准。

工程提速与研究筛选同时进行：预先固定较小、按来源/时长分层的开发池，保留四候选并跑 1–2 轮，用于排错、吞吐与方向诊断；完整数据用于方法定型后的主要对照。开发池时长先实测，不能保证按样本比例线性缩短，也不能把短程结果当最终收敛结论。每个已提交 checkpoint 可以独立评测，无需等十轮全部结束。

当前没有实测依据承诺四小时降至一小时。例如仅假设 TTS GRPO 加速 4 倍，其余不变，整轮也只是从 228.64 降至约 173.96 分钟（2.90 小时）；进一步降低需处理多个主要阶段。这是敏感性算例，不是性能结果。

## 复核材料

- [逐轮阶段表](stage_times.md)
- [数值、来源 SHA-256 与身份校验](evidence.json)
- [可重跑 CPU 审计脚本](audit.py)：`python3 audit.py`，仅生成本目录报告，不加载模型。
