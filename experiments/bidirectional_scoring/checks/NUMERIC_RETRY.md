# 首次模型试运行：数值检查失败与 v2 重试

## 已观察到的结果

用户于 2026-09-15 执行旧 `smoke01` 的 A、B。两边权重均已加载，均在第一个样本的数值检查阶段停止。两个 `items` 文件都是 `invalid`，没有有效候选评分。人工标注不需要因此重做。

| 检查项：最大目标 token log-probability 差 | A | B |
|---|---:|---:|
| 完全相同输入重复计算 | 0 | 0 |
| 尾部增加 3 个被 mask 的 padding | 0.2190570831 | 0.0823287666 |
| 首帧独立前缀＋逐码本路径 | 0.0546903610 | 不适用 |
| 改变未来音频帧，对此前分数的影响 | 0 | 不适用 |
| 各描述 token 只提供其前缀 | 不适用 | 0 |

重复计算阈值为 `1e-5`，替代路径阈值为 `0.05`。A 的 padding 和首帧独立路径超限，B 的 padding 超限。原程序因 `model.audit()` 先抛异常，未能单独生成 `numeric_audit.json`；数值仍保存在日志、`status.json` 和无效条目的异常文本中。

原始依据：

- `runs/smoke01/logs/score-a_2026-09-15T09-50-46.776439+00-00.log`
- `runs/smoke01/logs/score-b_2026-09-15T09-52-22.417666+00-00.log`
- `runs/smoke01/scores/{a,b}/v1/status.json` 与 `items/{A001,B001}.json`
- A 环境 Torch `2.11.0+cu126`；B 环境 Torch `2.6.0+cu124`；均为 BF16、Transformers `4.57.3`、自动 SDPA 后端。

`rope_scaling` 提示不是本次抛出 `ValueError` 的位置；日志显示模型在该提示之后完成加载并进入数值检查。不能从该提示单独推断根因。

## 源码支持的排查方向与未证实之处

已核对本机安装源码：

- `qwen-tts/.../modeling_qwen3_tts.py`：`Qwen3TTSTalkerForConditionalGeneration.forward` 每次无缓存前向根据 mask 重算位置；`get_rope_index` 对有效 token 使用累计位置，右侧 padding 不改变已有有效位置。独立首帧与主评分使用的主码本及此前子码本对齐关系一致；尚未发现这里有明确的 off-by-one 证据。
- `transformers/masking_utils.py` 中 `_ignore_causal_mask_sdpa`、`sdpa_mask` 允许无 padding 时省略显式 mask；含 padding 时采用显式 mask。
- `transformers/integrations/sdpa_attention.py` 根据 mask 决定 `is_causal`、GQA 路径，然后交给 PyTorch 选择 SDPA 实现。
- MiDasheng 的 `modeling_midashenglm.py:592` 将处理后的 embedding、mask、position_ids 传给 decoder；没有把 mask 丢弃。音频编码器还直接调用 PyTorch SDPA，因此只改 decoder 的 `attn_implementation` 不能覆盖所有注意力调用。

这些实现说明，两份数学上应一致的输入可能进入不同的计算路径。[PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html) 说明批处理形状、低精度归约可能导致不同结果；[SDPA 接口文档](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html) 提供固定后端的上下文接口。

**目前尚未证明这些超限差异完全来自 BF16／自动后端，也不能证明掩码存在错误。**v1 没有逐 token 对照向量或 GPU kernel 记录，无法从一个最大值推断平均分漂移或候选排序会不会改变。本次失败不是 A、B 研究假设的实验结论。

## 已实现的修复与诊断

- 默认固定 SDPA `MATH` 后端，覆盖评分、独立前缀、子码本和音频编码；退出上下文后恢复后端选择开关。
- 在用户启动的独立模型进程中关闭 TF32、BF16 GEMM 低精度归约、SDPA math 低精度归约。模型仍以 BF16 加载，不改权重、安装环境或标注。
- **原 `0.05`／`1e-5` 阈值不变。**失败仍禁止候选评分；通过这个检查也不证明微小风格分差可靠。
- 先保存完整检查文件，再判断是否停止；发生计算异常同样保存失败状态和 traceback。已有失败检查不会因为文件存在而被当成通过。
- 保存各路径目标 token 分数、最大误差、平均绝对误差、有符号平均分漂移及设备／后端信息。
- 新增相同形状下改变被 mask 的 padding 内容的检查；A 另在同一个 talker hidden 下比较逐码本与并行子码本结果，帮助定位差异来源。
- `run_smoke.sh` 默认用 `v2`，保留 `v1` 全部失败记录。两个方向独立，检查通过后才继续原来的 4 条／4 组。

## 已做的轻量验证

- `numeric_retry_unit_tests.log`：18 项标准库测试通过，包括保存失败检查、拒绝用失败检查解锁评分、异常落盘、后端与标签参数传递。
- `numeric_retry_tts_cpu.json`、`numeric_retry_caption_cpu.json`：在两个已有环境使用 CPU 小张量与恒定输出替身验证后端 API、上下文、adapter 的调用路径和 padding 基本运算。均 `cuda_initialized=false`、`models_loaded=false`。这些替身结果不是模型评分，更不证明真实模型数值检查已通过。
- 旧 A／B 的非执行输入检查通过。修复前后旧清单、人工标注、日志与 v1 结果哈希保持一致；基线在 `numeric_retry_baseline.json`。

## 用户重试

在共享存储模型服务器运行，末尾 `0` 替换为可用 GPU 编号：

```bash
cd /data/L202500147/Caption/BidirectionalScoringPilot
bash run_smoke.sh 0
```

日志在 `runs/smoke01/logs/`；完整检查在 `runs/smoke01/scores/{a,b}/v2/numeric_audit.json`；评分成功后汇总到 `runs/smoke01/reports/v2/`。真实 GPU 数值差异是否消除，以用户重试结果为准。新标注页面可以继续使用，无需重启。
