# V4 提速可行性：第一步离线分析

日期：2026-09-08。用户授权分析历史代码与结果，不启动 H100 测试、不修改训练实现或配置。

**结论：最有依据的两个优化候选是“轮初等价 reference 回放复用”和“TTS rollout 按预计工作量分配完整组”。目前能确认重复路径和负载问题，尚不能确认实际提速。批量推理也值得研究，但需要改 worker，不能靠增大现有 batch 配置完成。**

## 1. 范围与方法

- 历史来源：V3 权威 run `midasheng_7b_reward_v3_10rounds_20260905_run01`，r0–r9。
- 读取 180 个有耗时的完成 stage、20 个 rollout 输出及各卡元数据、80 个 TTS GRPO rank 心跳文件。
- 覆盖 Captioner 7,360 个候选与 TTS 7,840 个候选；直接比较已保存 old/reference 概率数组，并核对记录的 generation/policy replay 与 policy/reference 最大误差。
- 汇总 8,000 次 TTS GRPO 候选事件区间，含 DDP padding 的重复计算。心跳共 52,087 条，不调用模型、不重新计算 likelihood。
- V4 与 V3 的两个 worker、MiDasheng worker、分布式调度、telemetry、adapter 审计六个文件逐字节一致。因此这些性能路径仍适用于 V4；V4 更多 cycle-SFT 引入的耗时没有实测。
- 保存文件哈希、逐轮逐卡/逐组数量、可复算 Python 脚本。只使用标准库，不加载 torch/CUDA，不访问远端进程。

## 2. 时间花在哪里

180 个 stage 的 `elapsed_seconds` 合计，平均每轮 **253.53 分钟**。这是包括 worker 启动、加载、执行和退出的计时，不是纯 GPU kernel 时间，也没有涵盖所有 run 初始化/CPU bookkeeping。

| 阶段 | 平均分钟/轮 | 含义 |
|---|---:|---|
| TTS rollout | 83.92 | 生成、保存生成概率、音频解码/写盘、policy replay、reference replay等 |
| TTS GRPO | 67.03 | 原 codec 轨迹逐帧回放、候选反向、DDP/optimizer 等 |
| Captioner GRPO | 38.90 | 原 token 轨迹的原生回放及分块反向等 |
| Captioner rollout | 35.20 | 生成、policy replay、reference replay、解析等 |
| 其余 | 28.48 | 重建评分、gate 诊断、SFT、reload等 |

四个主阶段占 88.77%。两侧 SFT 合计平均约 4.84 分钟，gate 的四个模型校准 stage 合计约 8.93 分钟。去优化 gate 阈值搜索或微调 SFT，不是当前最主要的速度来源。

## 3. 优化候选一：有条件复用轮初 reference 回放

### 已确认的代码事实

`orchestrator._collect_audio_only` 与 `_collect_caption_only` 在 rollout 时都传入同一个 `checkpoint_in` 和 `target_checkpoint`，即本轮尚未更新的起点。此处的 reference 是用于 GRPO/KL 的轮初参考策略，**不是永久冻结的原始 base reward anchor**。

两条 rollout 的实际路径：

```text
Captioner：生成并保留 old_token_logprobs
        → 对原 token 做 policy replay（验证生成概率）
        → 对同一原 token 再做 reference replay（存储 reference 概率）

TTS：生成 codec，保留 main/sub 的实际采样概率
   → 解码并保存音频
   → incremental policy replay
   → incremental reference replay
```

r1–r9 两类 rollout 的汇总元数据都记录 policy/reference adapter 精确相同：Captioner 检查 396 个张量、TTS 检查 462 个张量，`exact=true`、`max_abs_diff=0`。这是汇总中的 adapter 审计记录，未重新加载权重做 GPU 审计。

### 已确认的历史数值

| 候选 | 十轮数量 | 记录的生成/policy replay最大误差 | 记录的policy/reference最大误差 | 保存的old/reference数组最大差异 |
|---|---:|---:|---:|---:|
| Captioner | 7,360 | 0 | 0 | 0 |
| TTS | 7,840 | main/sub 均0 | main/sub 均0 | main/sub 均0 |

r0 尚无 checkpoint adapter 对审计，采用新 LoRA/base 对照，但历史概率也全部相同。Captioner 7,352/7,360 个轨迹通过完整轨迹条件；8 个未通过不影响上述已保存概率差异统计，不能将概率相等写成所有候选都有效或语义正确。

### 建议的改法边界（未实施）

只在 rollout 的已确认等价条件下，让同一次 policy replay 的结果同时提供 reference 概率。**保留真实生成 old probabilities，保留 generation/replay 一致性检查。** 这能避免第二次完整 replay，而不是把所有概率无条件设为相同。

必须检查当前/reference 身份、adapter 数值和模式、eval/dropout、dtype、同一输入及概率处理规则；r0 应单独确认初始化 adapter 与 base 输出等价。不能仅凭路径字符串相同无条件走快速路径，也不能把“由复用构造出的零差异”冒充独立测得的 replay 审计，应记录独立的复用标志与验证来源。

不能复用于 GRPO 更新之后的当前 policy，也不能省掉用于 reward 的冻结 `C_0/T_0` 评分或跨 checkpoint 缓存模型输出。

### 能推断多少速度收益

假设 rollout 为 `生成G + 两次等价回放2R + 其他O`，复用后约为 `G + R + O`。可节省的是一次 R；**目前没有 G/R/O 的实测分解，不能宣称阶段提速三分之一，也不能承诺省下多少分钟。**

TTS 每次 incremental replay 的每个 codec frame 调用 code predictor 15 次。十轮共产生 645,567 帧，两次 replay 在代码层面对应约 19,367,010 次 predictor 调用（并非 GPU kernel 数），因此该路径值得优先测量。

代码定位：[TTS rollout](../../../DualISL_Train_RewardV4/dual_isl_train/workers/qwen_voice_design.py:621)、[TTS incremental replay](../../../DualISL_Train_RewardV4/dual_isl_train/workers/qwen_voice_design.py:352)、[Captioner rollout](../../../DualISL_Train_RewardV4/dual_isl_train/workers/qwen3_captioner.py:706)。

## 4. 优化候选二：TTS rollout 的完整组负载分配

### 已确认事实

当前 `shard_indexed_rows` 按完整 group 轮流分卡，组数接近不代表计算量接近。每个 rank 完成全部本地组后才汇合，旧日志没有逐组完成时间，不能直接测出其他卡等待了多少分钟。

TTS r9 的实际分配：

| rank | 分配组数 | 四候选合计codec帧数 |
|---|---:|---:|
| 0 | 25 | 8,394 |
| 1 | 25 | 8,921 |
| 2 | 25 | 7,137 |
| 3 | 25 | 7,990 |
| 4 | 24 | 7,022 |
| 5 | 24 | 11,108 |
| 6 | 24 | 5,518 |
| 7 | 24 | 8,876 |

最重卡的帧数是最轻卡的约两倍、平均值的 1.368 倍。十轮最重卡/平均帧数为 **1.341–1.398**，问题持续存在。

### 只用前一轮信息的模拟

对 r1–r9，用上一轮同一源 group 的四候选总帧数作成本估计，将预计较长的组优先分给预计总负载最轻的卡；组不拆分。再用本轮真实帧数检验这种分配，没有用本轮未来长度来作分配决定。

- 九轮中，最重卡的总帧数相对原分配下降 **22.66%–26.05%**，平均 **24.43%**。
- r9 模拟各卡帧数：8021/8212/7849/8171/8015/8062/8497/8139，最大值从 11108 降为 8497，下降 **23.51%**。
- r0 没有上一轮长度，不能直接使用这种估计；可另研究文本长度估计或完整组动态队列，本次未验证。

**这仅是工作量指标变化，不是实际提速。** 序列长度、prefix、硬件/I/O速度、随机数和数值变化都会影响真实时间。不能将24.43%直接乘以84分钟当作收益承诺。

优先改推理组分配，保持每组4候选、原候选seed、原始输出索引；不要同时改变GRPO训练batch的组成和更新顺序。分卡改变仍需H100验证候选/概率可复现性。

Captioner rollout 按 token 数计算的最重卡/平均值仅1.044–1.078；套用同样上一轮长度策略，九轮平均最大负载下降仅0.41%，部分轮反而增加。因此这不是Captioner当前的优先优化。

TTS GRPO **已经有长度分桶**；其逐同步步帧数最大/平均的累计比，十轮平均约1.112。这个代价指标也不是实际等待时间，不应把新增rollout均衡与现有GRPO分桶混为一谈。

代码定位：[推理分片与已有GRPO分桶](../../../DualISL_Train_RewardV4/dual_isl_train/distributed.py:91)。

## 5. 批量推理：有可能，但不是简单调配置

### TTS

本地 qwen_tts 底层 `generate` 有批量输入、padding、逐样本EOS切分实现，说明底层有可利用的批处理结构。但现有 worker 还有明确单样本限制：

- `rollout` 给底层传单元素list，并取 `codes_list[0]`；每候选单独全局设seed。
- main/sub生成概率提取函数要求logits第一维为1；sub trace按frame缓存。
- incremental和teacher replay都显式要求`codes.shape[0]==1`。
- 变长候选必须准确对应各自main/sub采样概率、EOS与有效帧mask，不能将padding作为训练token。

优先顺序建议：先等价replay复用，再尝试只读replay批处理，最后考虑生成批处理。replay不重新采样，较容易限定比较范围；但仍需改mask/cache等实现并核验数值。批量生成还必须处理每候选独立随机数、提前EOS和采样概率，不能只依赖一次全局seed。

### Captioner

MiDasheng 的 `generate_batch` 内部仍 `for seed in seeds` 单条生成；`sampled_token_logprobs_batch` 也逐序列回放。单纯增加`rollout_batch_size/replay_batch_size`不会得到真正的向量化batch，还可能触发原有契约校验。

`_inputs(audio,prompt)`在一组4候选生成与两次回放中共调用12次（4+4+4）。它重复执行processor聊天模板处理与设备搬运，可研究只读预处理结果复用；旧日志不能证明每次都实际从磁盘重读，也没有预处理耗时。TTS一组的相同request在外层tokenize一次、两次×4候选replay的conditioning中又tokenize8次，也可研究纯token输入缓存。

缓存边界必须明确：只读token/声学输入可研究复用；训练中可训练audio projector/模型embedding的激活不应跨optimizer更新或直接detach复用，否则改变梯度。生成、policy和reference如果使用不同adapter，模型相关cache也不可盲目共享。

r9历史rollout每卡PyTorch峰值：Captioner17.66–18.01 GiB、TTS4.62–5.24 GiB，支持评估batch，但不代表总显存占用、SM利用率或可安全扩大训练batch。

代码定位：[TTS概率提取](../../../DualISL_Train_RewardV4/dual_isl_train/workers/qwen_voice_design.py:26)、[MiDasheng生成](../../../DualISL_Train_RewardV4/scripts/midasheng_captioner_candidate.py:171)、[MiDasheng回放](../../../DualISL_Train_RewardV4/scripts/midasheng_captioner_candidate.py:245)。底层qwen_tts源码绝对路径及hash见 additional_code_sources.json。

## 6. GRPO与日志：可以看清一部分，不能据此武断优化

TTS现有心跳能把候选过程粗分成：

- candidate_start → candidate_loss_ready：前向回放、loss计算与CPU日志区间，累计占约36.21%。
- candidate_loss_ready → candidate_complete：反向、可能的DDP等待、显式CUDA同步与日志区间，累计占约63.79%。

两者是80个rank-round合计的**主机时间区间比例，不是全局墙钟比例或纯kernel耗时，更不能说64%全是通信**。末候选负责DDP同步，所以上述第二段包括等待；不能从日志精确分离通信与计算。

每轮每rank的25个step累计平均3921.26秒（65.35分钟），接近67.03分钟的整个TTS GRPO阶段。对这一阶段而言，主要时间确实在更新循环内；剩余约1.68分钟也包含多种开销，不能全部叫模型加载时间。

十轮52,087条心跳由共享存储上的append_jsonl逐次打开/关闭文件；`DUALISL_SHARED_WRITABLE=1`时还会逐次chmod。MiDasheng生成callback逐token将采样ID/概率转到CPU。两者都是可见的开销候选，但没有I/O或CPU/CUDA profiler，尚不能量化贡献。

可考虑持久化日志句柄、文件首次创建时设权限、保持必要刷盘频率，以及减少不必要的token级CPU同步；必须保留实时心跳与故障定位能力，不直接关闭日志。

MiDasheng GRPO每16个token做反向并detach KV cache。直接改变chunk大小或改成整段teacher forcing可能改变梯度路径；TTS去掉逐候选反向可能恢复历史OOM风险。这些不属于当前优先的等价优化。

## 7. 下一阶段最小验证方案（仅方案，尚未执行）

### 单H100：先测TTS等价replay复用

1. 用一致的代码/环境/已审计checkpoint，在相同原request和codec轨迹上分别计时policy/reference replay；同时测生成、解码/写盘和tokenization。
2. 区分冷启动与预热后重复测量，采用CUDA event或同步边界计时，明确记录同步开销；保留原代码基线。
3. 确认模式/dtype/adapter及概率处理一致后，验证共享结果与原两次replay结果；既有轨迹门槛为`5e-4`，但不能只凭过门槛就声称逐位等价，还应报告最大差异、GRPO loss与参数更新差异。
4. 改动实现应只应用在新隔离性能测试中，不改写旧checkpoint或V4训练目录。V4标准worker拒绝直接加载V3 checkpoint版本，未来测试需正确使用V3兼容基线/专用审计方案，不能为测试擅自改checkpoint版本标记。

可定位的r9 TTS代表组（每组保持4候选）：

| 范围 | group ID | 四候选总codec帧数 |
|---|---|---:|
| 约10%分位 | random440_expressivespeech_322 | 80 |
| 中位 | random440_expressivespeech_066 | 245 |
| 约90%分位 | random440_expressivespeech_117 | 668 |
| 最长 | random440_expressivespeech_018 | 1873 |

四组用于组件正确性与长度覆盖，不代表整池吞吐。样本源路径已保存在evidence.json的representative_groups。

### 8×H100：再测TTS推理分配与集成

- 用相同代表池/完整候选seed比较原round-robin与按上一轮帧数估计的完整组分配。需每卡多组并包含长样本，记录每组起止、各rank工作结束和barrier到达/退出。
- 单独比较原版、只复用replay、只换推理分配、两者组合，防止不知道收益来源；这是性能对比方案，不是当前启动的训练消融。
- 验证全局ID覆盖、每组4候选、概率/轨迹正确性、显存、总墙钟、异常恢复与双final完整smoke，再决定是否纳入后续训练。

## 8. 尚缺的测量与建议优先级

| 优先级 | 候选 | 已有依据 | 必须补的证据 |
|---|---|---|---|
| 1 | 等价reference replay复用 | 十轮概率差异0；r1–r9 adapter精确相同 | 单卡同条件计时、复用分支数值/更新验证 |
| 2 | TTS rollout按预计帧数分组分卡 | 每轮最重卡负载高34%–40%；上一轮长度模拟有效 | 8卡rank时间与整体墙钟、seed和输出验证 |
| 3 | 只读预处理复用/回放batch | 单条重复路径明确、生成显存有空间 | 子阶段profile、mask/概率与显存检查 |
| 4 | 生成batch、底层GRPO回放改写 | 底层batch结构存在，现worker单条限制明确 | 较大实现工作、采样/梯度/同步等价性 |

其他方向如常驻模型可研究，但当前没有独立加载时间，不应排在已确认热点之前。8卡已经参与单阶段数据并行，硬拆4+4并行两种模型不保证更快。减少候选、epoch、序列长度、reward或anchor计算会改变实验，不作为本次工程加速建议。

**本次交付是离线证据和可检验的优化方案，训练代码/配置未改变，GPU测试未执行，也没有给出未经测量的固定提速承诺。**

## 复算文件

- [主审计脚本](audit.py)：读取旧run并只写本目录派生数据。
- [分配模拟脚本](analyze_load.py)：读取派生CSV，无模型调用。
- [完整证据与来源hash](evidence.json)、[额外代码来源hash](additional_code_sources.json)。
- [stage耗时](stage_times.csv)、[逐卡rollout负载](rollout_rank_loads.csv)、[逐组成本](rollout_group_costs.csv)、[TTS候选主机时间区间](tts_grpo_candidate_intervals.csv)、[分配模拟](load_simulations.json)。

```bash
cd /data/L202500147/Caption/benchmark/reports/v4_performance_feasibility_20260908
PYTHONDONTWRITEBYTECODE=1 /data/L202500147/miniconda3/envs/qwen3-tts/bin/python audit.py
PYTHONDONTWRITEBYTECODE=1 /data/L202500147/miniconda3/envs/qwen3-tts/bin/python analyze_load.py
```
