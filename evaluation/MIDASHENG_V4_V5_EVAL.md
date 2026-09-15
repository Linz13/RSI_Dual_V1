# MiDasheng V4 / V5 四卡评测

评测 Captioner：V5 `round_000`，以及 V4 运行中已提交的全部轮次。
本次已核实 V4 `round_000`～`round_006`，共 8 个 checkpoint、24 项全量评测。
冻结清单、checkpoint 哈希、代码及数据摘要保存于输出目录的 `inventory.json`；续跑不自动纳入新增轮次。

## 在四卡服务器运行

先在四卡训练终端 Ctrl+C，等训练子进程退出，用 `nvidia-smi` 确认卡已释放。
以下命令使用物理卡 4、5、6、7；若服务器显示编号 0～3，把 `EVAL_GPUS` 改为 `0,1,2,3`。
建议在 tmux 会话中运行：

```bash
cd /data/L202500147/Caption/benchmark
EVAL_GPUS=4,5,6,7 bash run_midasheng_v4_v5_benchmarks_4gpu.sh run
```

前三个任务是 V5 的三个 benchmark，第四张卡同时开始 V4。空闲卡自动领取后续任务。
每张卡同一时间只执行一个任务；每项先跑独立的小规模验证，通过后立即进行全量评测，无须等其他模型验证完毕。
模型推理退出后再启动指标评分，避免同卡模型叠加占用显存。
所有打分使用现有本地评测资源，无须调用训练用的远程打标服务。

中断后执行相同命令即可续跑。完整且身份匹配的结果跳过，未完成推理使用各 benchmark 的 resume 功能。
Ctrl+C 会停止本次评测启动的子进程，保留已有结果。
某任务失败会记录日志，其他任务继续；命令最终以非零状态退出，修复后重跑同一命令。

```bash
# 仅做 CPU 预检，不占 GPU
bash run_midasheng_v4_v5_benchmarks_4gpu.sh check
# 查询完成数量并更新汇总
bash run_midasheng_v4_v5_benchmarks_4gpu.sh status
```

默认结果目录：
`/data/L202500147/Caption/benchmark/midasheng_v4_all_v5_round1_caption_4gpu_20260910_run01`

- `summary.md` / `summary.csv` / `summary.json`：每项完成后更新；V5 无须等 V4 完成即可查看。
- `logs/full_*midasheng_rewardv5_r0.log`：V5 三项进度，包含小规模验证及全量运行。
- `inventory.json`：冻结模型清单；r0 代表第一轮。
- `*/full/*/launcher_timing.json`：该任务本次调用耗时，可能包含小规模验证或续跑部分。

## 评测口径

| Benchmark | 每个 checkpoint 的全量规模 | 口径 |
|---|---:|---|
| EmotionTalk | 1,929 音频 × 4 任务 = 7,716 条预测 | standard_public；batch 4，max_new_tokens 128 |
| ParaSpeechCaps | 140 音频 × 6 属性 = 840 字段 | 现有 Scheme A attr6 |
| StyleCap / PromptSpeech | 3,112 道题 | 现有 speaker-open MCQ；batch 4 |

全部加载 `caption_final` LoRA，基模为 MiDashengLM-7B-1021-BF16，attention 为 sdpa。
EmotionTalk 必须成功计算 SPIDEr 和 FENSE；其他不可用指标保留原因，不能当作 0。
该 benchmark 沿用历史 standard_public 口径，不能宣称精确复现论文表格。
V4 各轮与 V5 第一轮展示训练轨迹；等轮次对比使用 V4 第一轮与 V5 第一轮。

## 耗时估计

参考 `later_caption_eval_runs_20260902_run01` 中 MiDasheng RewardV3 r0～r9 的同规模历史记录：

- EmotionTalk：模型加载后至评分文件写完约 25.2～31.4 分钟，另需模型加载。
- StyleCap：`wall_seconds_this_invocation` 约 3.6～4.4 分钟。
- ParaSpeechCaps：`wall_seconds` 约 7.0～8.1 分钟。

四张 H100 上，计入小规模验证、模型加载和调度，预计 **V5 三项约 35～50 分钟完成**，
**所有 8 个 checkpoint 合计预留 2～3 小时**。
这不是新 checkpoint 的实测；输出长度、CPU 配额及共享存储拥堵都可能延长耗时。
当前节点已做 CPU 预检和调度测试，实际 GPU 验证由上述 run 命令执行。
