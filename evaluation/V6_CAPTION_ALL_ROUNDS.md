# V6 十轮 Captioner 评测

训练目录：`/data/L202500147/Caption/DualRSI_Train_V6_AudioOnly4Attr/runs/midasheng_v6_8gpu_run01`。
默认固定评测 `round_000` 至 `round_009`（第 1–10 轮），校验每轮 commit 与 Captioner 权重哈希。

本入口仅依赖 Captioner 及本地指标环境，不加载 TTS 模型、不读取 API 密钥、不调用付费 API。
评测保持原来的三个 benchmark：EmotionTalk standard_public、ParaSpeechCaps Scheme A attr6、StyleCap speaker-open MCQ。

自动检查原目录 `v6_round000_002_benchmarks_run01` 和 `v6_round003_benchmarks_run01`。
只有 checkpoint、基模、数据/代码、batch、完整预测数量及主指标均通过校验的结果才会复用。
旧结果保持原位置，不覆盖、不复制、不创建指向旧目录的可写任务链接；来源及文件哈希记录在新 inventory 中。
新跑完的结果也会在重启时跳过；失败或中断的任务使用原有推理断点续跑机制。

## 8 卡运行

在 GPU 服务器的 tmux 终端执行：

```bash
cd /data/L202500147/Caption/benchmark
bash run_v6_caption_all_rounds.sh run \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpu-memory-gib 26 \
  --batch-size 4
```

26 GiB 是每个 GPU 进程的 PyTorch 分配预算，适用于此前每卡约 32 GiB 空闲的场景；另需 CUDA 等开销。
每卡同时最多运行一个 benchmark 任务；跨轮次和 benchmark 使用八卡任务队列。先开始耗时更长的 EmotionTalk。
EmotionTalk 和 StyleCap batch=4；ParaSpeechCaps 沿用原独立字段推理接口（单条），不虚报批量。
一个任务分配一张卡，不是把每个模型拆到八卡。部分任务完成后，末尾可能有卡空闲。
`run` 自动对待评测任务先做 smoke 再做完整评测，已复用的任务不重复 smoke。

```bash
# 只做 CPU 检查，不启动 GPU 推理。
bash run_v6_caption_all_rounds.sh check

# 可选：只做缺失任务的小规模 GPU 验证。
bash run_v6_caption_all_rounds.sh smoke --gpus 0,1,2,3,4,5,6,7 --gpu-memory-gib 26 --batch-size 4

# 退出评测入口后刷新状态；运行时直接查看 summary.md 和 logs。
bash run_v6_caption_all_rounds.sh status
```

输出：`/data/L202500147/Caption/benchmark/v6_caption_all_rounds_run01/`。

- `summary.md`：十轮 Captioner 主表，按之前展示要求不显示 StyleCap。
- `summary.csv`、`summary.json`：包括 StyleCap 的完整汇总，JSON 包含复用来源及各子任务分数。
- `captioner/logs/`：新任务的独立日志。
- `inventory.json`：固定 checkpoint、代码/数据/批次身份及旧结果指纹。
- `cpu_preflight.json`：CPU 校验和待完成任务统计。

中断后重复原命令即可；不更换同一输出目录的 batch、轮次、代码或 checkpoint。
可以更换 GPU 编号或卡数，它们只影响独立任务调度；同一输出根目录只允许一个入口运行。
如仅需部分轮次，可在新输出目录使用 `--rounds 4,5,6,7,8,9 --output-root ...`，参数为零起算。

本地开发验证仅使用 CPU，GPU 吞吐和显存仍以运行服务器为准。
